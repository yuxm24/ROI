"""Megatron-Core FSDP (HSDP) backend.

Independent NCCL communicators for all-gather and reduce-scatter, persistent
double buffers and NCCL user-buffer registration.  The shard axis of the
``(replicate, shard, tp)`` mesh may be any size that divides the world.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.tensor import DTensor

from zvla.core import BaseStrategy
from zvla.engines.utils import initialize_overwatch
from zvla.utils.checkpoint_io import (
    assert_scheduler_resume_compatible,
    torch_load_checkpoint,
)

overwatch = initialize_overwatch(__name__)


_EXPECTED_MCORE = "0.19.0"
_EXPECTED_MCORE_FSDP_API = "0.6.0"
_DEFAULT_PREFETCH_ELEMENTS = 327_313_408


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if hasattr(value, "to_local") else value


class _MCoreUnevenSavePlanner(DefaultSavePlanner):
    """Resolve MCore uneven-DTensor writes to their actual local shard.

    MCore replaces the write-item offsets with its true, potentially uneven
    partition metadata. PyTorch's default resolver subsequently looks that
    offset up using the standard *even* DTensor partition rule; with torch
    2.9 this can return an empty or full shard while the metadata describes a
    different tensor. Each MCore DTensor emits exactly one local write item,
    so resolving it directly to ``to_local()`` preserves the advertised
    payload without changing the model's sharding layout.
    """

    def __init__(self) -> None:
        super().__init__()
        self._local_snapshots: Dict[int, torch.Tensor] = {}
        self._snapshots_installed = False

    def install_local_snapshots(
        self,
        snapshots: Mapping[int, torch.Tensor],
    ) -> None:
        if self._snapshots_installed:
            raise RuntimeError("DCP local snapshots were installed more than once")
        self._local_snapshots = dict(snapshots)
        self._snapshots_installed = True

    def lookup_object(self, index):
        value = self.state_dict[index.fqn]
        if isinstance(value, DTensor):
            snapshot = self._local_snapshots.get(id(value))
            if snapshot is None:
                raise RuntimeError(
                    f"missing staged uneven-DTensor payload for {index.fqn!r}")
            expected = tuple(value.to_local().shape)
            if tuple(snapshot.shape) != expected:
                raise RuntimeError(
                    f"staged payload shape {tuple(snapshot.shape)} differs "
                    f"from local shard {expected} for {index.fqn!r}")
            return snapshot
        return super().lookup_object(index)


class _MCoreUnevenLoadPlanner(DefaultLoadPlanner):
    """Load MCore uneven-DTensor chunks into their actual local shard."""

    def lookup_tensor(self, index):
        value = self.state_dict[index.fqn]
        if isinstance(value, DTensor):
            return value.to_local()
        return super().lookup_tensor(index)


class _MCoreUnevenStager:
    """Stage exact local shards while retaining DTensors for DCP planning.

    PyTorch's generic 2.9 stager deep-copies a DTensor using its global shape,
    which changes the local payload of an uneven MCore partition. This stager
    leaves only the lightweight DTensor descriptors in the planned state and
    gives the save planner immutable CPU copies of their real local tensors.
    Plain tensors and containers are copied normally, so the returned state is
    safe while training overlaps the filesystem write.
    """

    _synchronize_after_execute = True

    def __init__(self, planner: _MCoreUnevenSavePlanner) -> None:
        self._planner = planner

    @property
    def should_synchronize_after_execute(self) -> bool:
        return True

    @staticmethod
    def _stage_value(value: Any, snapshots: Dict[int, torch.Tensor]) -> Any:
        if isinstance(value, DTensor):
            local = value.to_local().detach()
            snapshots[id(value)] = local.to(
                device="cpu", copy=True).contiguous()
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", copy=True).contiguous()
        if isinstance(value, dict):
            return {
                key: _MCoreUnevenStager._stage_value(item, snapshots)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                _MCoreUnevenStager._stage_value(item, snapshots)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                _MCoreUnevenStager._stage_value(item, snapshots)
                for item in value
            )
        if value is None or isinstance(
                value, (bool, int, float, str, bytes, torch.dtype, torch.device)):
            return value
        raise TypeError(
            f"unsupported value in sealed DCP state: {type(value).__name__}")

    def stage(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        snapshots: Dict[int, torch.Tensor] = {}
        staged = self._stage_value(state_dict, snapshots)
        self._planner.install_local_snapshots(snapshots)
        return staged

    def synchronize_staging(self) -> None:
        return None


class MegatronFSDPStrategy(BaseStrategy):
    """Megatron-Core v0.19 HSDP with independent AG/RS communication."""

    def __init__(
        self,
        device_id: int = 0,
        mixed_precision: Optional[str] = "bf16",
        reduce_in_full_precision: bool = False,
        sharding_strategy: str = "hybrid_shard",
        gradient_checkpointing: bool = False,
        cpu_offload: bool = False,
        sync_module_states: bool = True,
        resume_from_checkpoint: bool = False,
        device_mesh=None,
        megatron_prefetch_elements: int = _DEFAULT_PREFETCH_ELEMENTS,
        megatron_nccl_ub: bool = True,
        megatron_double_buffer: bool = True,
        megatron_manual_registration: bool = True,
        megatron_ubr_registration_mode: str = "local",
        megatron_high_priority_ag: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(device_id=device_id)
        self._mixed_precision = mixed_precision
        self._mp_dtype = torch.bfloat16
        self.resume_from_checkpoint = bool(resume_from_checkpoint)
        self.sync_module_states = bool(sync_module_states)
        self.device_mesh = device_mesh
        self.prefetch_elements = int(megatron_prefetch_elements)
        self.ubr_registration_mode = str(megatron_ubr_registration_mode)
        self.high_priority_ag = bool(megatron_high_priority_ag)
        self.sharding_strategy = sharding_strategy
        self._checkpoint_publish_poll_steps = int(os.environ.get(
            "ZVLA_CHECKPOINT_PUBLISH_POLL_STEPS", "50"))

        self._api = None
        self._ag_group = None
        self._training_group = None
        self._status_group = None
        self._writer_control_group = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        # Deliberately process-local: loading Adam/global_step must not imply
        # this process exercised the lazy communication buffers.
        self._optimizer_steps_this_process = 0
        self._buffers_registered = False
        self._pending_registration_step: Optional[int] = None
        self._registration_global_step: Optional[int] = None
        self._registered_allocation_signature = None
        self._registered_pool_segments = None
        self._pending_future: Optional[Future] = None
        self._pending_checkpoint: Optional[Dict[str, Any]] = None
        self._pending_state: Optional[Dict[str, Any]] = None
        self._registered_pool = None
        self._registered_groups = ()

    @property
    def mixed_precision_enabled(self) -> bool:
        return True

    @property
    def mixed_precision_dtype(self) -> torch.dtype:
        return self._mp_dtype

    def _load_and_validate_mcore(self):
        if self._api is not None:
            return self._api
        from megatron.core.distributed.fsdp.src import megatron_fsdp as api

        self._api = api
        return api

    def _create_process_groups(self) -> None:
        world = dist.get_world_size()
        shard_size = int(self.device_mesh.size(1))
        if world % shard_size:
            raise RuntimeError("world size is not divisible by the shard size")
        rank = dist.get_rank()
        options = dist.ProcessGroupNCCL.Options()
        options.is_high_priority_stream = self.high_priority_ag
        selected = None
        for node in range(world // shard_size):
            ranks = list(range(node * shard_size, (node + 1) * shard_size))
            group = dist.new_group(
                ranks=ranks,
                backend="nccl",
                pg_options=options,
                group_desc=f"zvla_mfsdp_ag_node_{node}",
            )
            if rank in ranks:
                selected = group
        if selected is None:
            raise RuntimeError("rank did not join an independent AG group")
        rs_group = self.device_mesh.get_group("shard")
        if selected is rs_group:
            raise RuntimeError("AG and RS unexpectedly share one communicator")
        if dist.get_process_group_ranks(selected) != dist.get_process_group_ranks(
                rs_group):
            raise RuntimeError("AG and RS communicators have different members")
        self._ag_group = selected

        # MCore owns WORLD's collective ordering while its forward/backward
        # hooks overlap parameter all-gathers and gradient reduce-scatters.
        # Keep every application-level tensor collective (finite guard,
        # telemetry, dataloader agreement) on an independent full-world NCCL
        # communicator so a framework-internal sequence number can never be
        # consumed by a trainer metric.
        self._training_group = dist.new_group(
            backend="nccl",
            group_desc="zvla_mfsdp_global_training",
        )

        self._status_group = dist.new_group(
            backend="gloo",
            group_desc="zvla_mfsdp_global_status",
        )
        writer_ranks = list(range(shard_size))
        self._writer_control_group = dist.new_group(
            ranks=writer_ranks,
            backend="gloo",
            group_desc="zvla_mfsdp_dcp_writer",
        )


    @staticmethod
    def _move_buffers(module: nn.Module, device: torch.device) -> None:
        for child in module.modules():
            for name, value in tuple(child._buffers.items()):
                if value is not None:
                    child._buffers[name] = value.to(device=device)

    @staticmethod
    def _prepare_compute_parameters(model: nn.Module) -> None:
        """Create BF16 compute weights; FP32 conditioners stay FP32."""
        from zvla.models.zerith_va.primitives import FP32Linear

        fp32_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, FP32Linear)
            for parameter in module.parameters(recurse=False)
        }
        for parameter in model.parameters():
            if (not parameter.is_floating_point()
                    or id(parameter) in fp32_ids):
                continue
            parameter.data = parameter.data.to(dtype=torch.bfloat16)

    @staticmethod
    def _conditioner_bucket_size(model: nn.Module) -> int:
        """Return a bucket boundary that isolates each FP32 conditioner.

        MCore groups all non-unit parameters with the same dtype and gradient
        policy before it installs module hooks.  The model has separate Wan and
        Reader FP32 time conditioners whose gradients become ready at very
        different points in the branched graph.  Combining both in one bucket
        violates MCore's bucket-completion invariant.  Splitting at the largest
        complete conditioner keeps each mathematical module intact while the
        heavy Wan/Reader blocks alone use the two-buffer unit pool.
        """
        from zvla.models.zerith_va.primitives import (
            FP32Linear,
            SinusoidalConditioner,
        )

        sizes = []
        owned = set()
        all_fp32_linear = set()
        for module in model.modules():
            if isinstance(module, FP32Linear):
                all_fp32_linear.update(
                    id(parameter)
                    for parameter in module.parameters(recurse=False))
            if not isinstance(module, SinusoidalConditioner):
                continue
            parameters = [
                parameter
                for child in module.modules()
                if isinstance(child, FP32Linear)
                for parameter in child.parameters(recurse=False)
            ]
            if not parameters:
                raise RuntimeError("SinusoidalConditioner has no FP32Linear parameters")
            overlap = owned.intersection(id(parameter) for parameter in parameters)
            if overlap:
                raise RuntimeError("FP32Linear belongs to multiple conditioners")
            owned.update(id(parameter) for parameter in parameters)
            sizes.append(sum(parameter.numel() for parameter in parameters))
        if not sizes or owned != all_fp32_linear:
            raise RuntimeError(
                "every FP32Linear must belong to exactly one conditioner")
        return max(sizes)

    def setup(self, model: nn.Module) -> nn.Module:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Megatron FSDP requires torchrun/NCCL")
        torch.cuda.set_device(self.device_id)
        self._load_and_validate_mcore()
        self._create_process_groups()

        self._root_model = model
        coordinate = self.device_mesh.get_coordinate()
        if coordinate is None:
            raise RuntimeError("rank is outside the configured device mesh")
        load_pretrained = not self.resume_from_checkpoint and int(
            coordinate[0]) == 0
        if load_pretrained and hasattr(model, "load_pretrained_weights"):
            model.load_pretrained_weights()
        if dist.is_initialized():
            dist.barrier(group=self._status_group)
        if hasattr(model, "freeze_backbones"):
            model.freeze_backbones()
        self._prepare_compute_parameters(model)
        self._move_buffers(model, self.training_device)
        conditioner_bucket_size = self._conditioner_bucket_size(model)
        fsdp_unit_modules = [
            "zvla.models.zerith_va.primitives.WanCompatibleBlock",
            "zvla.models.zerith_va.policy_reader.MixedReaderLayer",
        ]

        api = self._api
        ddp_config = api.DistributedDataParallelConfig(
            data_parallel_sharding_strategy="optim_grads_params",
            outer_dp_sharding_strategy="no_shard",
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=False,
            bucket_size=conditioner_bucket_size,
            fsdp_all_gather_in_start_param_sync=True,
            suggested_communication_unit_size=self.prefetch_elements,
            nccl_ub=True,
            fsdp_double_buffer=True,
            # Non-unit root parameters also receive persistent storage.  A
            # dynamic allocator fallback would make those buckets ineligible
            # for UBR and violate the sealed communication contract.
            fsdp_db_use_persist_buf_on_alloc_fail=True,
            # ``local`` uses conventional ncclCommRegister; ``symmetric``
            # uses NCCL window registration where the platform supports it.
            disable_symmetric_registration=(
                self.ubr_registration_mode == "local"),
            fsdp_manual_registration=True,
            megatron_fsdp_main_params_dtype=torch.float32,
            megatron_fsdp_main_grads_dtype=torch.float32,
            megatron_fsdp_grad_comm_dtype=torch.bfloat16,
            megatron_fsdp_use_decoupled_grad=False,
            megatron_fsdp_cuda_graph_mode=False,
            megatron_fsdp_enable_fine_grained_param_gather=False,
            # Wan and Reader blocks are intentionally asymmetric.  MCore's
            # fixed allocator would select only the largest equal-size family
            # and silently route the other family through dynamic storage.
            # MaxPool is still a persistent two-buffer allocator, but covers
            # every declared FSDP unit and remains eligible for UBR.
            megatron_fsdp_max_pool_double_buffer=True,
        )
        policy = api.MixedPrecisionPolicy(
            main_params_dtype=torch.float32,
            main_grads_dtype=torch.float32,
            grad_comm_dtype=torch.bfloat16,
        )
        index = api.FSDPDistributedIndex(
            device_mesh=self.device_mesh,
            dp_shard_dim="shard",
            dp_outer_dim="replicate",
            tp_dim="tp",
            hybrid_fsdp_group=dist.group.WORLD,
            fsdp_group_ag=self._ag_group,
        )
        wrapped = api.MegatronFSDP(
            module=model,
            dist_index=index,
            ddp_config=ddp_config,
            mixed_precision_policy=policy,
            fsdp_unit_modules=fsdp_unit_modules,
            disable_bucketing=False,
            device=None,
            calculate_per_token_loss=False,
            init_model_with_meta_device=False,
            sync_model_each_microbatch=False,
            report_nan_in_param_grad=False,
        )
        from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
            preprocess_state_dict_for_uneven_dtensor,
        )

        def prepare_uneven_state_dict(
            module,
            state_dict,
            prefix,
            local_metadata,
        ) -> None:
            del module, prefix, local_metadata
            preprocess_state_dict_for_uneven_dtensor(state_dict)

        wrapped.register_state_dict_post_hook(prepare_uneven_state_dict)
        self._arm_final_microbatch_sync(wrapped)
        self._wrapped_model = wrapped
        self._validate_double_buffer_contract(wrapped)
        self._validate_conditioner_bucket_boundaries(wrapped)
        if wrapped.suggested_RS_queue_capacity != self.prefetch_elements:
            raise RuntimeError(
                "MCore changed the validated reduce-scatter queue capacity")
        if wrapped.suggested_AG_prefetch_size != self.prefetch_elements // 2:
            raise RuntimeError(
                "MCore changed the validated one-unit all-gather prefetch size")
        if (wrapped.side_stream_for_buffer_copy_and_grad_accum.cuda_stream
                == wrapped.side_stream_for_param_gather.cuda_stream):
            raise RuntimeError(
                "Megatron FSDP AG and RS unexpectedly share one CUDA stream")
        if not self.resume_from_checkpoint:
            self._sync_fresh_hsdp_state(wrapped)
        dist.barrier(group=self._status_group)
        return wrapped

    @staticmethod
    def _arm_final_microbatch_sync(wrapped: nn.Module) -> None:
        """Mark the default one-batch optimizer cycle as replica-synchronous.

        MCore deliberately leaves ``is_last_microbatch`` unset when automatic
        synchronization is disabled.  In HSDP that still launches node-local
        reduce-scatter, but skips the outer replica all-reduce.  Zerith VA trains
        without gradient accumulation, so every backward is the final
        microbatch.  The MCore ``no_sync()`` context temporarily clears this
        flag and restores it for a future final microbatch if accumulation is
        ever exercised by a diagnostic.
        """
        if bool(getattr(wrapped, "model_auto_sync", False)):
            raise RuntimeError(
                "Megatron HSDP must keep model_auto_sync disabled so replica "
                "communication can overlap compute")
        wrapped.is_last_microbatch = True

    def _validate_double_buffer_contract(self, wrapped: nn.Module) -> None:
        buffer = wrapped.param_and_grad_buffer
        config = buffer.ddp_config
        if not (config.nccl_ub and config.fsdp_double_buffer
                and config.fsdp_manual_registration):
            raise RuntimeError("MCore disabled a mandatory UBR/double-buffer flag")
        expected_disabled = self.ubr_registration_mode == "local"
        if bool(config.disable_symmetric_registration) != expected_disabled:
            raise RuntimeError("MCore UBR registration mode differs from config")
        allocators = (
            buffer.weight_alloc,
            buffer.transpose_weight_alloc,
            buffer.main_grad_alloc,
            buffer.hsdp_grad_comm_alloc,
        )
        if any(allocator is None or
               allocator.__class__.__name__ != "MaxPoolAllocator"
               for allocator in allocators):
            raise RuntimeError(
                "MCore did not construct persistent MaxPool double buffers")
        expected_units = {
            group.fsdp_unit_id
            for group in buffer.parameter_groups
            if group.fsdp_unit_id is not None
        }
        if not expected_units:
            raise RuntimeError("MCore found no heavy Wan/Reader FSDP units")
        for allocator in allocators:
            if not allocator.fallback_to_persistent_buffer:
                raise RuntimeError(
                    f"{allocator.name} permits dynamic allocation fallback")
            actual_units = set(allocator.fsdp_double_buffer_units)
            if actual_units != expected_units:
                raise RuntimeError(
                    f"{allocator.name} does not persist every heavy FSDP "
                    f"unit: missing={sorted(expected_units - actual_units)}")
        if buffer.already_registered:
            raise RuntimeError("manual UBR was unexpectedly registered at startup")

    def _validate_conditioner_bucket_boundaries(self, wrapped: nn.Module) -> None:
        """Fail if MCore merges two hook-distinct FP32 conditioners."""
        from zvla.models.zerith_va.primitives import (
            FP32Linear,
            SinusoidalConditioner,
        )

        conditioner_names = set()
        for name, module in self._root_model.named_modules():
            if not isinstance(module, SinusoidalConditioner):
                continue
            if any(isinstance(child, FP32Linear)
                   for child in module.modules()):
                conditioner_names.add(name)
        buffer = wrapped.param_and_grad_buffer
        observed = set()
        for group in buffer.parameter_groups:
            group_owners = {
                owner
                for parameter in group.params
                for owner in conditioner_names
                if buffer.param_to_name[parameter].startswith(owner + ".")
                and ".time_" in buffer.param_to_name[parameter]
            }
            if len(group_owners) > 1:
                raise RuntimeError(
                    "MCore merged hook-distinct FP32 conditioners into one "
                    f"bucket: {sorted(group_owners)}")
            if group_owners and group.fsdp_unit_id is not None:
                raise RuntimeError(
                    "FP32 conditioner unexpectedly entered the heavy-unit pool")
            observed.update(group_owners)
        if observed != conditioner_names:
            raise RuntimeError("MCore omitted an FP32 conditioner bucket")

    def _sync_fresh_hsdp_state(self, wrapped: nn.Module) -> None:
        replica_group = self.device_mesh.get_group("replicate")
        local_values = []
        with torch.no_grad():
            for parameter in self._root_model.parameters():
                local_values.append(_local_tensor(parameter.data))
            for buffer in self._root_model.buffers():
                local_values.append(_local_tensor(buffer.data))
            if local_values:
                dist._broadcast_coalesced(
                    replica_group,
                    local_values,
                    256 << 20,
                    0,
                )
            wrapped.install_optimized_model_weights()

        payload = None
        if self.is_rank_zero():
            payload = {
                "teacher_provenance": getattr(
                    self._root_model, "teacher_provenance", None),
                "initialization_report": getattr(
                    self._root_model, "initialization_report", None),
            }
        values = [payload]
        dist.broadcast_object_list(
            values,
            src=0,
            group=self._status_group,
        )
        payload = values[0]
        for name, value in payload.items():
            if hasattr(self._root_model, name):
                setattr(self._root_model, name, value)
        for module in self._root_model.modules():
            marker = getattr(module, "mark_official_weights_loaded", None)
            if callable(marker):
                marker()
        if hasattr(self._root_model, "_initialization_complete"):
            self._root_model._initialization_complete = True

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        del model
        if self._root_model is None:
            raise RuntimeError("strategy.setup() must precede unwrap_model()")
        return self._root_model

    @staticmethod
    def _optimizer_samples(
        optimizer: torch.optim.Optimizer,
    ) -> list[torch.Tensor]:
        samples = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                value = _local_tensor(parameter.detach()).reshape(-1)
                if value.numel() == 0:
                    samples.append(torch.empty(0, dtype=value.dtype))
                    continue
                indices = sorted({0, value.numel() // 2, value.numel() - 1})
                samples.append(value[indices].cpu().clone())
        return samples

    def setup_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        if model is not self._wrapped_model:
            raise RuntimeError("optimizer must be built from the wrapped model")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("Megatron FSDP supports the pinned AdamW equation only")
        # MCore v0.19 retrieves the first optimizer parameter from the first
        # mapping entry.  Canonicalize that entry explicitly instead of
        # depending on application-specific group-key insertion order.
        optimizer.param_groups[:] = [
            {
                "params": group["params"],
                **{key: value for key, value in group.items()
                   if key != "params"},
            }
            for group in optimizer.param_groups
        ]
        before = self._optimizer_samples(optimizer)
        hyperparameters = [
            (group["lr"], group["weight_decay"])
            for group in optimizer.param_groups
        ]
        for group in optimizer.param_groups:
            group["lr"] = 0.0
            group["weight_decay"] = 0.0
        optimizer = self._api.fully_shard_optimizer(
            optimizer,
            preproc_state_dict_for_dcp_ckpt=False,
        )
        for group, (learning_rate, weight_decay) in zip(
                optimizer.param_groups, hyperparameters):
            group["lr"] = learning_rate
            group["weight_decay"] = weight_decay
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        self._wrapped_model.zero_grad_buffer()
        after = self._optimizer_samples(optimizer)
        if len(before) != len(after) or any(
                not torch.equal(left, right)
                for left, right in zip(before, after)):
            raise RuntimeError(
                "MCore optimizer initialization changed parameter values")
        self._optimizer = optimizer
        # MCore's MaxPool communication buffers are lazy.  In particular,
        # installing FP32 main shards does not materialize all weight/gradient
        # buckets.  Manual UBR therefore has to wait until one complete real
        # iteration has exercised forward, backward, Adam, weight install and
        # zero_grad.  This mirrors the pinned MCore v0.19 training loop.
        if self._buffers_registered or self._pending_registration_step is not None:
            raise RuntimeError("optimizer setup found stale UBR lifecycle state")
        return optimizer

    def finish_gradient_sync(self, model: nn.Module) -> None:
        if model is not self._wrapped_model:
            raise RuntimeError("gradient sync received the wrong model")
        if not bool(getattr(self._wrapped_model, "is_last_microbatch", False)):
            raise RuntimeError(
                "optimizer boundary reached before the final MCore microbatch; "
                "replica gradients would be unsynchronized")
        self._wrapped_model.finish_grad_sync()

    def optimizer_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if model is not self._wrapped_model or optimizer is not self._optimizer:
            raise RuntimeError("optimizer step received an unowned object")
        optimizer.step(
            sync_grad_before_optimizer_step=False,
            install_optimized_model_weights=False,
        )
        self._optimizer_steps_this_process += 1

    def after_optimizer_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        root_model: nn.Module,
        global_step: int,
    ) -> None:
        if model is not self._wrapped_model or optimizer is not self._optimizer:
            raise RuntimeError("post-step received an unowned object")
        root_model.after_optimizer_step(global_step)
        self._wrapped_model.install_optimized_model_weights()
        self._maybe_finish_pending_checkpoint(global_step)
        if not self._buffers_registered:
            if self._pending_registration_step is not None:
                raise RuntimeError(
                    "a second optimizer step reached the unregistered UBR pool")
            self._pending_registration_step = int(global_step)

    @staticmethod
    def _communication_allocation_signature(buffer) -> tuple:
        """Return the MaxPool allocation plan that UBR is about to seal."""
        rows = []
        for label, attribute in (
            ("weight", "weight_alloc"),
            ("transpose_weight", "transpose_weight_alloc"),
            ("main_grad", "main_grad_alloc"),
            ("hsdp_grad_comm", "hsdp_grad_comm_alloc"),
        ):
            allocator = getattr(buffer, attribute)
            tracker = getattr(allocator, "allocation_tracker", None)
            if not isinstance(tracker, dict):
                raise RuntimeError(
                    f"MCore {label} allocator has no allocation tracker")
            entries = []
            for key, size in tracker.items():
                if not (isinstance(key, tuple) and len(key) == 2):
                    raise RuntimeError(
                        f"MCore {label} allocation key changed: {key!r}")
                name, dtype = key
                entries.append((str(name), str(dtype), int(size)))
            rows.append((label, tuple(sorted(entries))))
        signature = tuple(rows)
        if not any(entries for _, entries in signature):
            raise RuntimeError(
                "the first training iteration materialized no MaxPool buffers")
        return signature

    @staticmethod
    def _pool_segment_signature(pool) -> tuple:
        """Capture only immutable CUDA memory-pool segment coordinates."""
        rows = []
        for segment in pool.snapshot():
            required = {
                "device", "address", "total_size", "segment_type",
                "segment_pool_id",
            }
            if not required.issubset(segment):
                raise RuntimeError(
                    "torch CUDA MemPool snapshot schema changed")
            rows.append((
                int(segment["device"]),
                int(segment["address"]),
                int(segment["total_size"]),
                str(segment["segment_type"]),
                tuple(int(value) for value in segment["segment_pool_id"]),
            ))
        signature = tuple(sorted(rows))
        if not signature:
            raise RuntimeError(
                "the MCore NCCL memory pool is empty after the first iteration")
        return signature

    def _assert_registered_allocations_stable(self) -> None:
        """Reject communication buffers first seen after manual registration."""
        if not self._buffers_registered:
            return
        buffer = self._wrapped_model.param_and_grad_buffer
        if not buffer.already_registered:
            raise RuntimeError("local UBR state diverged from MCore")
        current = self._communication_allocation_signature(buffer)
        if current != self._registered_allocation_signature:
            raise RuntimeError(
                "MCore allocated or grew a communication buffer after UBR "
                "registration; the sealed model execution path changed")

    @staticmethod
    def _register_pool_with_backend(backend, pool, mode: str) -> None:
        """Use exactly one requested ProcessGroupNCCL registration API."""
        if mode == "symmetric":
            backend.register_mem_pool(pool, symm=True)
            return
        if mode == "local":
            backend.register_mem_pool(pool)
            return
        raise ValueError(f"unsupported UBR registration mode: {mode!r}")

    def _unique_group_backends(self, groups) -> tuple:
        # Singleton mesh axes may reuse WORLD. PyTorch registers a pool once
        # per backend, so registering/unregistering it must use that identity.
        backends = []
        for group in groups:
            backend = group._get_backend(self.training_device)
            if all(backend is not existing for existing in backends):
                backends.append(backend)
        return tuple(backends)

    def _register_user_buffers_strict(self, global_step: int) -> None:
        """Register the MCore pool in the explicitly configured UBR mode.

        No exception-driven downgrade is permitted.  ``local`` calls the
        conventional registration API directly; ``symmetric`` calls the NCCL
        window-registration API directly.  Either mode must succeed on every
        configured communicator or the run is aborted collectively.
        """
        buffer = self._wrapped_model.param_and_grad_buffer
        if self._buffers_registered:
            raise RuntimeError("NCCL user buffers were registered more than once")
        if buffer.already_registered:
            raise RuntimeError("NCCL user buffers were registered more than once")
        if self._pending_registration_step != int(global_step):
            raise RuntimeError("UBR registration did not follow one completed step")
        from megatron.core.distributed.fsdp.src.megatron_fsdp import (
            param_and_grad_buffer as buffer_module,
        )

        pool = buffer_module.NCCL_MEMORY_POOL
        if pool is None:
            raise RuntimeError("MCore NCCL memory pool was not initialized")
        torch.cuda.synchronize(self.training_device)
        dist.barrier(group=self._status_group)
        allocation_signature = self._communication_allocation_signature(buffer)
        pool_segments = self._pool_segment_signature(pool)
        for backend in self._unique_group_backends(buffer.ubr_groups):
            self._register_pool_with_backend(
                backend, pool, self.ubr_registration_mode)
        torch.cuda.synchronize(self.training_device)
        dist.barrier(group=self._status_group)
        buffer.already_registered = True
        self._registered_pool = pool
        self._registered_groups = tuple(buffer.ubr_groups)
        self._registered_allocation_signature = allocation_signature
        self._registered_pool_segments = pool_segments
        self._registration_global_step = int(global_step)
        self._pending_registration_step = None
        self._buffers_registered = True
        if self.is_rank_zero():
            overwatch.info(
                "Registered the fully materialized MCore NCCL pool after "
                f"training step {global_step} with "
                f"{self.ubr_registration_mode} UBR")

    def zero_grad(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if model is not self._wrapped_model:
            raise RuntimeError("zero_grad received the wrong model")
        optimizer.zero_grad(set_to_none=True, zero_grad_buffer=True)
        if self._pending_registration_step is not None:
            self._register_user_buffers_strict(
                self._pending_registration_step)
        else:
            self._assert_registered_allocations_stable()

    def _optimizer_gradients(self) -> list[torch.Tensor]:
        if self._optimizer is None:
            raise RuntimeError("optimizer is not initialized")
        return [
            _local_tensor(parameter.grad)
            for group in self._optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]

    @staticmethod
    def _apply_gradient_clip(
        gradients: list[torch.Tensor],
        total: torch.Tensor,
        max_norm: float,
    ) -> None:
        if not gradients:
            return
        coefficient = torch.clamp(
            float(max_norm) / (total + 1e-6), max=1.0)
        torch._foreach_mul_(gradients, coefficient)

    @torch.no_grad()
    def clip_grad_norm(
        self,
        model: nn.Module,
        max_norm: float,
        *,
        precomputed_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del model, precomputed_norm
        gradients = self._optimizer_gradients()
        if not gradients:
            return torch.zeros((), device=self.training_device,
                               dtype=torch.float32)
        norms = torch._foreach_norm(gradients, 2, dtype=torch.float32)
        squared = torch.stack(norms).square().sum()
        self.reduce_sharded_gradient_tensor(squared)
        total = squared.sqrt()
        if torch.isfinite(total):
            self._apply_gradient_clip(gradients, total, max_norm)
        return total

    @torch.no_grad()
    def clip_grad_norm_and_assert_finite(
        self,
        model: nn.Module,
        max_norm: float,
        finite_values: torch.Tensor,
        *,
        precomputed_norm: Optional[torch.Tensor] = None,
        label: str,
    ) -> torch.Tensor:
        """Fuse the finite guard into one exact HSDP norm collective.

        HSDP leaves one identical FP32 main shard on every replica.  Summing
        local shard squares over WORLD and dividing by the replica count
        therefore produces the same norm as a shard-group reduction.  A
        local non-finite loss is encoded as NaN before that SUM, making the
        failure collective-consistent without a second scalar all-reduce.
        """
        if model is not self._wrapped_model:
            raise RuntimeError("gradient guard received the wrong model")
        gradients = self._optimizer_gradients()
        # Never let observability data drive optimizer mathematics.  The
        # backend recomputes this scalar from the same FP32 main-shard
        # gradients AdamW consumes, even when a caller supplies a telemetry
        # norm.  The WORLD SUM below is already required to propagate a
        # finite-loss failure collectively, so this adds no communication.
        del precomputed_norm
        if gradients:
            norms = torch._foreach_norm(gradients, 2, dtype=torch.float32)
            square = torch.stack(norms).square().sum()
        else:
            square = torch.zeros(
                (), device=self.training_device, dtype=torch.float32)
        post_reduce_divisor = float(self.device_mesh.size(0))
        local_failed = (~torch.isfinite(
            finite_values.detach())).any() | (~torch.isfinite(square))
        square = torch.where(
            local_failed,
            torch.full_like(square, float("nan")),
            square,
        )
        self.global_all_reduce(square, op=dist.ReduceOp.SUM)
        if post_reduce_divisor != 1.0:
            square.div_(post_reduce_divisor)
        total = square.sqrt()
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError(
                f"non-finite {label} detected collectively; optimizer state "
                "was not updated")
        self._apply_gradient_clip(gradients, total, max_norm)
        return total

    def reduce_sharded_gradient_tensor(self, tensor: torch.Tensor) -> None:
        group = self.device_mesh.get_group("shard")
        if dist.get_world_size(group=group) > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

    def gradient_shard_world_size(self) -> int:
        return dist.get_world_size(group=self.device_mesh.get_group("shard"))

    def global_all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp,
    ) -> None:
        if self.world_size() > 1:
            if not self._is_process_group(self._training_group):
                raise RuntimeError(
                    "Megatron application communicator is not initialized")
            dist.all_reduce(tensor, op=op, group=self._training_group)

    def global_all_gather(
        self,
        output_tensors: list[torch.Tensor],
        tensor: torch.Tensor,
    ) -> None:
        if self.world_size() > 1:
            if not self._is_process_group(self._training_group):
                raise RuntimeError(
                    "Megatron application communicator is not initialized")
            dist.all_gather(
                output_tensors, tensor, group=self._training_group)

    def global_all_gather_into_tensor(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> None:
        if self.world_size() > 1:
            if not self._is_process_group(self._training_group):
                raise RuntimeError(
                    "Megatron application communicator is not initialized")
            dist.all_gather_into_tensor(
                output_tensor, input_tensor, group=self._training_group)

    def global_all_gather_object(self, output_objects: list, obj: Any) -> None:
        if self.world_size() > 1:
            if not self._is_process_group(self._status_group):
                raise RuntimeError(
                    "Megatron status communicator is not initialized")
            dist.all_gather_object(
                output_objects, obj, group=self._status_group)

    def no_sync_context(self, model: nn.Module):
        if model is not self._wrapped_model:
            raise RuntimeError("no_sync received the wrong model")
        return self._wrapped_model.no_sync()

    def is_rank_zero(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier(group=self._status_group)

    def broadcast_from_rank0(self, obj: Any) -> Any:
        values = [obj if self.is_rank_zero() else None]
        dist.broadcast_object_list(values, src=0, group=self._status_group)
        return values[0]

    def _is_writer(self) -> bool:
        coordinate = self.device_mesh.get_coordinate()
        if coordinate is None:
            raise RuntimeError("rank is outside the device mesh")
        return int(coordinate[0]) == 0

    @staticmethod
    def _initialize_nonempty_adam_state(
        optimizer: torch.optim.Optimizer,
    ) -> None:
        for group in optimizer.param_groups:
            fused = bool(group.get("fused", False))
            capturable = bool(group.get("capturable", False))
            for parameter in group["params"]:
                if parameter in optimizer.state and optimizer.state[parameter]:
                    continue
                if _local_tensor(parameter).numel() == 0:
                    continue
                step_device = parameter.device if fused or capturable else "cpu"
                state = {
                    "step": torch.zeros((), dtype=torch.float32,
                                        device=step_device),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
                if bool(group.get("amsgrad", False)):
                    state["max_exp_avg_sq"] = torch.zeros_like(parameter)
                optimizer.state[parameter] = state

    def _preprocess_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
        state_dict: Dict[str, Any],
    ) -> None:
        """Coordinate uneven Adam shards without inventing scalar state.

        Empty parameter shards must participate in MCore's per-DTensor metadata
        collectives, but must not contribute a conflicting Adam ``step`` value.
        This mirrors the pinned MCore hook using the node-local shard group and
        a disposable structural copy: mock DTensors exist only long enough to
        establish the global chunk layout, while DCP saves/loads the original
        optimizer dictionary.
        """
        from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
            preprocess_state_dict_for_uneven_dtensor,
        )

        local_keys = sorted({
            key
            for state in optimizer.state.values()
            for key, value in state.items()
            if isinstance(value, DTensor)
        })
        shard_group = self.device_mesh.get_group("shard")
        gathered_keys = [None] * dist.get_world_size(shard_group)
        dist.all_gather_object(gathered_keys, local_keys, group=shard_group)
        dtensor_keys = sorted({
            key for keys in gathered_keys for key in keys
        })
        if not dtensor_keys:
            raise RuntimeError("Adam state has no distributed tensor entries")

        def shallow_copy(value):
            if isinstance(value, dict):
                return {key: shallow_copy(item)
                        for key, item in value.items()}
            return value

        extended = shallow_copy(state_dict)
        parameter_indices: Dict[int, int] = {}
        next_index = 0
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                identity = id(parameter)
                if identity not in parameter_indices:
                    parameter_indices[identity] = next_index
                    next_index += 1
                if parameter in optimizer.state or not isinstance(
                        parameter, DTensor):
                    continue
                parameter_index = parameter_indices[identity]
                target = extended["state"].setdefault(parameter_index, {})
                for key in dtensor_keys:
                    target[key] = DTensor.from_local(
                        local_tensor=torch.empty(
                            0, dtype=parameter.dtype, device=parameter.device),
                        device_mesh=parameter.device_mesh,
                        placements=parameter.placements,
                        shape=parameter.shape,
                        stride=parameter.stride(),
                    )
        preprocess_state_dict_for_uneven_dtensor(extended)

    def _checkpoint_state(
        self,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, Any]:
        from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
            preprocess_state_dict_for_uneven_dtensor,
        )

        model_state = self._root_model.checkpoint_state_dict()
        preprocess_state_dict_for_uneven_dtensor(model_state)
        state: Dict[str, Any] = {"model": model_state}
        if optimizer is not None:
            self._initialize_nonempty_adam_state(optimizer)
            optimizer_state = optimizer.state_dict()
            self._preprocess_optimizer_state(optimizer, optimizer_state)
            state["optimizer"] = optimizer_state
        return state

    def _collect_status_errors(self, local_error: Optional[str]) -> list[str]:
        values = [None for _ in range(self.world_size())]
        dist.all_gather_object(values, local_error, group=self._status_group)
        return [value for value in values if value is not None]

    def _maybe_finish_pending_checkpoint(self, global_step: int) -> None:
        """Publish a completed asynchronous DCP save at a safe point.

        ``async_save`` stages the checkpoint before returning and leaves only
        the CFS write in flight.  Every rank polls on the same optimizer-step
        cadence.  A tiny Gloo reduction proves that all eight writer futures
        are complete before entering the existing collective finalizer.
        """
        if self._pending_checkpoint is None:
            return
        if int(global_step) % self._checkpoint_publish_poll_steps:
            return
        local_ready = True
        if self._is_writer():
            local_ready = (
                self._pending_future is not None
                and self._pending_future.done()
            )
        ready = torch.tensor(int(local_ready), dtype=torch.int32)
        dist.all_reduce(
            ready,
            op=dist.ReduceOp.MIN,
            group=self._status_group,
        )
        if int(ready.item()) == 1:
            self._finish_pending_checkpoint()

    def _finish_pending_checkpoint(self) -> None:
        if self._pending_checkpoint is None:
            return
        local_error = None
        if self._is_writer():
            try:
                self._pending_future.result()
            except BaseException as exc:  # collective propagation below
                local_error = f"rank {self.rank()}: {type(exc).__name__}: {exc}"
        errors = self._collect_status_errors(local_error)
        pending = self._pending_checkpoint
        if errors:
            self._pending_future = None
            self._pending_checkpoint = None
            self._pending_state = None
            raise RuntimeError(
                "asynchronous Megatron DCP save failed: " + "; ".join(errors))

        publish_error = None
        if self.is_rank_zero():
            try:
                incomplete = Path(pending["incomplete"])
                final = Path(pending["final"])
                metadata = dict(pending["metadata"])
                metadata_temp = incomplete / "metadata.pt.incomplete"
                torch.save(metadata, metadata_temp)
                os.replace(metadata_temp, incomplete / "metadata.pt")
                complete_temp = incomplete / ".complete.incomplete"
                with complete_temp.open("w", encoding="utf-8") as handle:
                    handle.write("ok\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(complete_temp, incomplete / ".complete")
                os.replace(incomplete, final)
                self._update_latest_link(final.parent, final)
                self._cleanup_old_sharded_checkpoints(
                    final.parent, int(pending["max_keep"]))
            except (OSError, RuntimeError, ValueError) as exc:
                publish_error = f"{type(exc).__name__}: {exc}"
        publish_error = self.broadcast_from_rank0(publish_error)
        self._pending_future = None
        self._pending_checkpoint = None
        self._pending_state = None
        if publish_error is not None:
            raise RuntimeError(
                f"failed to atomically publish Megatron checkpoint: "
                f"{publish_error}")
        dist.barrier(group=self._status_group)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        global_step: int,
        epoch: int,
        run_dir: Path,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
        max_keep: int = 0,
        checkpoint_format: str = "sharded",
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        del model, only_trainable
        if checkpoint_format != "sharded":
            raise ValueError("Megatron FSDP writes resumable DCP checkpoints only")
        self._finish_pending_checkpoint()
        checkpoint_dir = Path(run_dir) / "checkpoints"
        # Loss is accumulated locally and can differ slightly across ranks.
        # Rank zero must choose the physical directory once; otherwise the
        # eight DCP writers can scatter one logical checkpoint across eight
        # different four-decimal loss suffixes.
        name = None
        if self.is_rank_zero():
            loss = f"{train_loss:.4f}" if train_loss is not None else "inf"
            name = f"step-{global_step:06d}-epoch-{epoch:02d}-loss={loss}"
        name = self.broadcast_from_rank0(name)
        final = checkpoint_dir / name
        incomplete = checkpoint_dir / f".{name}.incomplete"
        prepare_error = None
        already_complete = False
        if self.is_rank_zero():
            try:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                already_complete = (final / ".complete").is_file()
                if not already_complete:
                    if final.exists():
                        raise FileExistsError(
                            f"unpublished final checkpoint path exists: {final}")
                    stale_pattern = (
                        f".step-{global_step:06d}-epoch-{epoch:02d}-"
                        "loss=*.incomplete")
                    for stale in checkpoint_dir.glob(stale_pattern):
                        if stale.is_dir() and not stale.is_symlink():
                            shutil.rmtree(stale)
                        else:
                            stale.unlink(missing_ok=True)
                    incomplete.mkdir()
            except OSError as exc:
                prepare_error = f"{type(exc).__name__}: {exc}"
        prepare_error = self.broadcast_from_rank0(prepare_error)
        already_complete = self.broadcast_from_rank0(already_complete)
        if prepare_error is not None:
            raise RuntimeError(f"cannot prepare checkpoint: {prepare_error}")
        if already_complete:
            return
        dist.barrier(group=self._status_group)

        state = None
        future = None
        save_start_error = None
        if self._is_writer():
            try:
                import torch.distributed.checkpoint as dcp
                from torch.distributed.checkpoint import FileSystemWriter

                state = self._checkpoint_state(optimizer)
                planner = _MCoreUnevenSavePlanner()
                future = dcp.async_save(
                    state,
                    # The stager has already copied every payload to CPU.
                    # PyTorch 2.9's default copy-ahead loader still opens a
                    # CUDA stream from the background thread, whose current
                    # device defaults to GPU 0. Eight writers can therefore
                    # create extra contexts on rank 0's GPU and OOM at save
                    # time, especially after a high-water-mark resume.
                    # Disable only that redundant device-copy path; CPU/storage
                    # writing remains asynchronous and DCP math/ABI is intact.
                    storage_writer=FileSystemWriter(
                        str(incomplete), per_thread_copy_ahead=0),
                    planner=planner,
                    process_group=self._writer_control_group,
                    async_stager=_MCoreUnevenStager(planner),
                )
            except Exception as exc:  # propagated collectively below
                save_start_error = (
                    f"rank {self.rank()}: {type(exc).__name__}: {exc}")
        start_errors = self._collect_status_errors(save_start_error)
        if start_errors:
            # A writer may have returned a live Future before a peer reported
            # its startup failure.  Removing the directory at that point races
            # the surviving DCP worker and can leave files recreated after the
            # cleanup.  Drain every Future that actually started first, then
            # publish the combined failure and remove the unpublished tree.
            finish_start_error = None
            if self._is_writer() and future is not None:
                try:
                    future.result()
                except BaseException as exc:  # collective propagation below
                    finish_start_error = (
                        f"rank {self.rank()}: async cleanup: "
                        f"{type(exc).__name__}: {exc}")
            finish_start_errors = self._collect_status_errors(
                finish_start_error)
            cleanup_error = None
            if self.is_rank_zero():
                try:
                    if incomplete.exists():
                        shutil.rmtree(incomplete)
                except OSError as exc:
                    cleanup_error = f"{type(exc).__name__}: {exc}"
            cleanup_error = self.broadcast_from_rank0(cleanup_error)
            detail = "; ".join((*start_errors, *finish_start_errors))
            if cleanup_error is not None:
                detail += f"; cleanup failed: {cleanup_error}"
            raise RuntimeError(
                f"could not start asynchronous Megatron DCP save: {detail}")
        metadata: Dict[str, Any] = {
            "global_step": int(global_step),
            "epoch": int(epoch),
            "checkpoint_format": "sharded",
            "checkpoint_format_version": 3,
            "distributed_backend": "megatron_fsdp",
            "mcore_version": _EXPECTED_MCORE,
            "mcore_fsdp_api_version": _EXPECTED_MCORE_FSDP_API,
            "torch_version": torch.__version__,
            "world_size": self.world_size(),
            "saved_hsdp_replica_groups": 1,
            "device_mesh_shape": list(self.device_mesh.shape),
            "device_mesh_dim_names": list(self.device_mesh.mesh_dim_names),
            "prefetch_elements": self.prefetch_elements,
            "nccl_ubr": True,
            "nccl_ubr_registration_mode": self.ubr_registration_mode,
            "double_buffer": True,
        }
        if scheduler is not None:
            metadata["scheduler_state_dict"] = scheduler.state_dict()
        if extra_state:
            metadata.update(extra_state)
        self._pending_future = future
        self._pending_state = state
        self._pending_checkpoint = {
            "incomplete": str(incomplete),
            "final": str(final),
            "metadata": metadata,
            "max_keep": int(max_keep),
        }
        # async_save has completed GPU-to-CPU staging before it returns.  Keep
        # ranks aligned once, then overlap only the filesystem write with the
        # following training steps.
        dist.barrier(group=self._status_group)

    def _validate_checkpoint_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        error = None
        metadata = None
        if self.is_rank_zero():
            try:
                path = Path(checkpoint_path)
                if not (path / ".complete").is_file():
                    raise RuntimeError("checkpoint has no atomic .complete marker")
                metadata = torch_load_checkpoint(
                    path / "metadata.pt", map_location="cpu")
                if metadata.get("distributed_backend") != "megatron_fsdp":
                    raise ValueError(
                        "optimizer migration from FSDP2/DDP is intentionally "
                        "unsupported")
                if int(metadata.get("world_size", -1)) != self.world_size():
                    raise ValueError("resume world size differs from checkpoint")
                if metadata.get("device_mesh_shape") != list(
                        self.device_mesh.shape):
                    raise ValueError("resume device mesh differs from checkpoint")
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = self.broadcast_from_rank0(error)
        if error is not None:
            raise RuntimeError(f"checkpoint preflight failed: {error}")
        return self.broadcast_from_rank0(metadata)

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        del model
        self._finish_pending_checkpoint()
        metadata = self._validate_checkpoint_metadata(checkpoint_path)
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader
        from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
            preprocess_state_dict_for_uneven_dtensor,
        )

        model_state = self._root_model.checkpoint_state_dict()
        preprocess_state_dict_for_uneven_dtensor(model_state)
        state: Dict[str, Any] = {"model": model_state}
        if optimizer is not None:
            self._initialize_nonempty_adam_state(optimizer)
            optimizer_state = optimizer.state_dict()
            self._preprocess_optimizer_state(optimizer, optimizer_state)
            state["optimizer"] = optimizer_state
        load_error = None
        try:
            dcp.load(
                state,
                storage_reader=FileSystemReader(checkpoint_path),
                planner=_MCoreUnevenLoadPlanner(),
            )
        except Exception as exc:  # propagated collectively below
            load_error = f"rank {self.rank()}: {type(exc).__name__}: {exc}"
        load_errors = self._collect_status_errors(load_error)
        if load_errors:
            raise RuntimeError(
                "Megatron DCP load failed: " + "; ".join(load_errors))
        self._root_model.load_checkpoint_state_dict(state["model"])
        if optimizer is not None:
            if "optimizer" not in state:
                raise KeyError("checkpoint has no optimizer state")
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            if "scheduler_state_dict" not in metadata:
                raise KeyError("checkpoint has no scheduler state")
            assert_scheduler_resume_compatible(metadata, scheduler)
            scheduler.load_state_dict(metadata["scheduler_state_dict"])
        self._restore_runtime_markers(metadata)
        self._wrapped_model.install_optimized_model_weights()
        dist.barrier(group=self._status_group)
        return {
            "global_step": metadata.get("global_step", 0),
            "epoch": metadata.get("epoch", 0),
            "trainer_state": metadata.get("trainer_state", {}),
            "sampler_state": metadata.get("sampler_state"),
            "rng_state": metadata.get("rng_state"),
            "teacher_provenance": metadata.get("teacher_provenance"),
        }

    def _restore_runtime_markers(self, metadata: Mapping[str, Any]) -> None:
        for name in ("teacher_provenance", "initialization"):
            target = "initialization_report" if name == "initialization" else name
            if hasattr(self._root_model, target) and name in metadata:
                setattr(self._root_model, target, metadata[name])
        for module in self._root_model.modules():
            marker = getattr(module, "mark_official_weights_loaded", None)
            if callable(marker):
                marker()
            if hasattr(module, "allow_uninitialized"):
                module.allow_uninitialized = False
        if hasattr(self._root_model, "allow_uninitialized_teacher"):
            self._root_model.allow_uninitialized_teacher = False
        if hasattr(self._root_model, "_initialization_complete"):
            self._root_model._initialization_complete = True

    def load_model_weights(self, model: nn.Module, checkpoint_path: str) -> None:
        self.load_checkpoint(model, None, None, checkpoint_path)

    def fork_from_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
        *,
        restore_optimizer: bool,
    ) -> Dict[str, Any]:
        return self.load_checkpoint(
            model,
            optimizer if restore_optimizer else None,
            None,
            checkpoint_path,
        )

    def save_model_artifact(
        self,
        model: nn.Module,
        global_step: int,
        epoch: int,
        run_dir: Path,
        *,
        max_keep: int = 1,
    ) -> str:
        del model
        self._finish_pending_checkpoint()
        artifact_modules = self._root_model.model_artifact_modules()
        checkpoint_modules = self._root_model.checkpoint_modules()
        if not artifact_modules:
            raise RuntimeError("model_artifact_modules() cannot be empty")
        incompatible = [
            name for name, module in artifact_modules.items()
            if checkpoint_modules.get(name) is not module
        ]
        if incompatible:
            raise RuntimeError(
                "Megatron DCP artifact modules must use checkpoint names and "
                f"objects; incompatible={incompatible}")
        path = Path(run_dir) / "checkpoints" / (
            f"model-step-{global_step:06d}-epoch-{epoch:02d}.safetensors")
        error = None
        if self.is_rank_zero():
            temporary = Path(str(path) + ".incomplete")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary.unlink(missing_ok=True)
                candidates = [
                    candidate for candidate in path.parent.glob(
                        f"step-{global_step:06d}-epoch-{epoch:02d}-loss=*")
                    if candidate.is_dir()
                    and (candidate / ".complete").is_file()
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        "artifact export requires exactly one complete source "
                        f"checkpoint, found {candidates}")
                source = candidates[0].resolve()
                source_metadata = torch_load_checkpoint(
                    source / "metadata.pt", map_location="cpu")
                if int(source_metadata.get("global_step", -1)) != int(
                        global_step):
                    raise RuntimeError("artifact source global step differs")
                timeout = int(os.environ.get(
                    "ZVLA_ARTIFACT_EXPORT_TIMEOUT_SEC", "7200"))
                if timeout <= 0:
                    raise ValueError(
                        "ZVLA_ARTIFACT_EXPORT_TIMEOUT_SEC must be positive")
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = ""
                environment["PYTHONNOUSERSITE"] = "1"
                command = [
                    sys.executable,
                    "-m",
                    "zvla.utils.megatron_dcp_export",
                    "--checkpoint",
                    str(source),
                    "--output",
                    str(temporary),
                    "--global-step",
                    str(int(global_step)),
                    "--epoch",
                    str(int(epoch)),
                ]
                for prefix in artifact_modules:
                    command.extend(("--module-prefix", prefix))
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    env=environment,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError(
                        "CPU DCP exporter did not produce a model artifact; "
                        f"stdout={completed.stdout[-2000:]!r}, "
                        f"stderr={completed.stderr[-2000:]!r}")
                os.replace(temporary, path)
                link = path.parent / "latest-checkpoint.safetensors"
                temporary_link = (
                    path.parent / ".latest-checkpoint.safetensors.incomplete")
                temporary_link.unlink(missing_ok=True)
                temporary_link.symlink_to(path.name)
                os.replace(temporary_link, link)
                self._cleanup_old_artifacts(path.parent, int(max_keep))
            except (
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                temporary.unlink(missing_ok=True)
                error = f"{type(exc).__name__}: {exc}"
        error = self.broadcast_from_rank0(error)
        if error is not None:
            raise RuntimeError(f"model artifact export failed: {error}")
        dist.barrier(group=self._status_group)
        return str(path)

    def prune_checkpoints(
        self,
        run_dir: Path,
        *,
        checkpoint_format: str,
        max_keep: int,
    ) -> None:
        if checkpoint_format != "sharded":
            raise ValueError("Megatron checkpoint format must be sharded")
        self._finish_pending_checkpoint()
        if self.is_rank_zero():
            self._cleanup_old_sharded_checkpoints(
                Path(run_dir) / "checkpoints", int(max_keep))
        dist.barrier(group=self._status_group)

    @staticmethod
    def _update_latest_link(directory: Path, target: Path) -> None:
        temporary = directory / ".latest-checkpoint.incomplete"
        link = directory / "latest-checkpoint"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target.name)
        os.replace(temporary, link)

    @staticmethod
    def _cleanup_old_sharded_checkpoints(directory: Path, max_keep: int) -> None:
        if max_keep <= 0 or not directory.exists():
            return
        checkpoints = sorted(
            path for path in directory.glob("step-*-epoch-*-loss=*")
            if path.is_dir() and (path / ".complete").is_file())
        for path in checkpoints[:-max_keep]:
            shutil.rmtree(path)

    @staticmethod
    def _cleanup_old_artifacts(directory: Path, max_keep: int) -> None:
        if max_keep <= 0:
            return
        artifacts = sorted(directory.glob("model-step-*-epoch-*.safetensors"))
        for path in artifacts[:-max_keep]:
            path.unlink()

    def finalize(self) -> None:
        self._finish_pending_checkpoint()

    def release_runtime(self) -> None:
        """Drop every CUDA tensor owner before communicator destruction."""
        self._optimizer = None
        self._pending_state = None
        self._pending_future = None
        self._pending_checkpoint = None
        super().release_runtime()

    @staticmethod
    def _is_process_group(group) -> bool:
        return (
            group is not None
            and group != dist.GroupMember.NON_GROUP_MEMBER
            and group is not dist.group.WORLD
        )

    def destroy_process_groups(self) -> None:
        """Deregister the fixed pool, then destroy subgroups deterministically."""
        if not dist.is_available() or not dist.is_initialized():
            return
        if self._is_process_group(self._status_group):
            dist.barrier(group=self._status_group)

        if self._registered_pool is not None:
            torch.cuda.synchronize(self.training_device)
            if self._registered_pool.snapshot():
                for backend in self._unique_group_backends(self._registered_groups):
                    backend.deregister_mem_pool(self._registered_pool)
            torch.cuda.synchronize(self.training_device)
            if self._is_process_group(self._status_group):
                dist.barrier(group=self._status_group)

        groups = [self._ag_group]
        if self.device_mesh is not None:
            groups.extend((
                self.device_mesh.get_group("shard"),
                self.device_mesh.get_group("replicate"),
            ))
        groups.extend((
            self._writer_control_group,
            self._training_group,
            self._status_group,
        ))
        destroyed = set()
        for group in groups:
            if not self._is_process_group(group) or id(group) in destroyed:
                continue
            dist.destroy_process_group(group)
            destroyed.add(id(group))

        self._registered_groups = ()
        self._registered_pool = None
        self._registered_allocation_signature = None
        self._registered_pool_segments = None
        self._pending_registration_step = None
        self._registration_global_step = None
        self._buffers_registered = False
        self._ag_group = None
        self._writer_control_group = None
        self._training_group = None
        self._status_group = None
        self.device_mesh = None
