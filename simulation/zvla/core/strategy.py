# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


class BaseStrategy(ABC):
    """Abstract base for distributed training strategies.

    Encapsulates *all* distributed-related operations: model wrapping,
    checkpoint save/load, gradient clipping, and synchronisation.
    The trainer delegates every distributed operation to the strategy
    so that it never calls FSDP/DDP APIs directly.
    """

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id
        self._wrapped_model: Optional[nn.Module] = None
        self._root_model: Optional[nn.Module] = None

    # --------------------------------------------------------------------- #
    #  Model setup
    # --------------------------------------------------------------------- #
    @abstractmethod
    def setup(self, model: nn.Module) -> nn.Module:
        """Prepare ``model`` for distributed training.

        Args:
            model: Unwrapped ``VLAModel`` instance.

        Returns:
            The (possibly wrapped) model ready for training.
        """
        ...

    @abstractmethod
    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Return the original application model through strategy/compile wrappers."""
        ...

    # --------------------------------------------------------------------- #
    #  Checkpointing
    # --------------------------------------------------------------------- #
    @abstractmethod
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
        checkpoint_format: str = 'full',
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a training checkpoint."""
        ...

    @abstractmethod
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Load a checkpoint and return restored training state.

        Returns:
            Dict with at least ``global_step`` and ``epoch`` keys.
        """
        ...

    def load_model_weights(
        self,
        model: nn.Module,
        checkpoint_path: str,
    ) -> None:
        """Load model weights without restoring any training state.

        Strategies may override this to use a more memory-efficient loading
        path.  The default delegates to ``load_checkpoint`` with no optimizer
        or scheduler and deliberately discards the returned progress state.
        """
        self.load_checkpoint(
            model=model,
            optimizer=None,
            scheduler=None,
            checkpoint_path=checkpoint_path,
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
        """Publish one model-only safetensors artifact on shared storage.

        Distributed strategies must override this method because gathering a
        sharded model is collective.  It is intentionally separate from a
        resumable checkpoint: inference/stage hand-off files do not contain
        optimizer, scheduler, sampler, or RNG state.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement model artifact export")

    def prune_checkpoints(
        self,
        run_dir: Path,
        *,
        checkpoint_format: str,
        max_keep: int,
    ) -> None:
        """Apply a retention limit without writing another checkpoint."""
        del run_dir, checkpoint_format, max_keep

    def fork_from_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
        *,
        restore_optimizer: bool,
    ) -> Dict[str, Any]:
        """Initialize a new run from model/progress and optional Adam state.

        Unlike :meth:`load_checkpoint`, the new run owns its scheduler, RNG,
        sampler and W&B identity. Strategies with named distributed optimizer
        state may override this method to retain moments for only the
        parameters shared by the source and destination experiments.
        """
        return self.load_checkpoint(
            model=model,
            optimizer=optimizer if restore_optimizer else None,
            scheduler=None,
            checkpoint_path=checkpoint_path,
        )

    # --------------------------------------------------------------------- #
    #  Gradient & optimisation helpers
    # --------------------------------------------------------------------- #
    def setup_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        """Attach strategy-specific optimizer state after parameter grouping.

        The default implementation leaves the optimizer untouched.  Sharded
        backends may use this boundary to replace optimizer parameters with
        their distributed FP32 main shards and to install checkpoint hooks.
        """
        del model
        return optimizer

    def finish_gradient_sync(self, model: nn.Module) -> None:
        """Make every optimizer-visible gradient ready for clipping/update."""
        del model

    def optimizer_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Apply one optimizer update without running model-owned EMA hooks."""
        del model
        optimizer.step()

    def after_optimizer_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        root_model: nn.Module,
        global_step: int,
    ) -> None:
        """Update model-owned post-step state after the FP32 update is visible."""
        del model, optimizer
        root_model.after_optimizer_step(global_step)

    def zero_grad(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Clear optimizer and strategy-owned gradient storage."""
        del model
        optimizer.zero_grad(set_to_none=True)

    def finalize(self) -> None:
        """Wait for strategy-owned asynchronous work before process teardown."""

    def release_runtime(self) -> None:
        """Release model references before distributed communicators are torn down."""
        self._wrapped_model = None
        self._root_model = None

    def destroy_process_groups(self) -> None:
        """Destroy strategy-owned subgroups in their required collective order."""

    @abstractmethod
    def clip_grad_norm(
        self,
        model: nn.Module,
        max_norm: float,
        *,
        precomputed_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Clip gradients and return their pre-clipping global norm.

        ``precomputed_norm`` lets a logging step reuse an exact norm that was
        already collected for telemetry. Implementations may ignore it when
        their native clipping primitive cannot safely consume an external
        norm.

        Implementations must not raise on a non-finite norm before every rank
        has participated in the same collectives.  The trainer calls
        :meth:`assert_finite` on the returned scalar before ``optimizer.step``
        so one bad rank cannot silently corrupt Adam state or strand peers in
        a different control-flow branch.
        """
        ...

    def clip_grad_norm_and_assert_finite(
        self,
        model: nn.Module,
        max_norm: float,
        finite_values: torch.Tensor,
        *,
        precomputed_norm: Optional[torch.Tensor] = None,
        label: str,
    ) -> torch.Tensor:
        """Clip gradients and propagate a finite-value failure collectively.

        The default keeps the existing two-operation behavior.  Backends whose
        gradient norm is already a collective may override this boundary to
        encode the finite guard in that same operation.
        """
        norm = self.clip_grad_norm(
            model,
            max_norm,
            precomputed_norm=precomputed_norm,
        )
        self.assert_finite(
            torch.cat((
                finite_values.detach().float().reshape(-1),
                norm.detach().float().reshape(1),
            )),
            label=label,
        )
        return norm

    # --------------------------------------------------------------------- #
    #  Distributed helpers
    # --------------------------------------------------------------------- #
    @abstractmethod
    def is_rank_zero(self) -> bool:
        ...

    @abstractmethod
    def rank(self) -> int:
        ...

    @abstractmethod
    def world_size(self) -> int:
        ...

    @abstractmethod
    def barrier(self) -> None:
        ...

    def global_all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        op: dist.ReduceOp,
    ) -> None:
        """Reduce an application tensor across every training rank.

        The default process group is correct for the legacy strategies.
        Backends that reserve WORLD for framework-internal collectives may
        override this method with a dedicated full-world communicator.
        """
        if (dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            dist.all_reduce(tensor, op=op)

    def global_all_gather(
        self,
        output_tensors: list[torch.Tensor],
        tensor: torch.Tensor,
    ) -> None:
        """Gather equal-shaped application tensors from every training rank."""
        if (dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            dist.all_gather(output_tensors, tensor)

    def global_all_gather_into_tensor(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> None:
        """Gather one application tensor per rank into a flat output tensor."""
        if (dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            dist.all_gather_into_tensor(output_tensor, input_tensor)

    def global_all_gather_object(self, output_objects: list, obj: Any) -> None:
        """Gather Python checkpoint/control state from every training rank."""
        if (dist.is_available() and dist.is_initialized()
                and dist.get_world_size() > 1):
            dist.all_gather_object(output_objects, obj)

    def assert_finite(self, value: torch.Tensor, *, label: str) -> None:
        """Raise on every rank when ``value`` is non-finite anywhere.

        A local-only loss check is unsafe in distributed training: one rank
        could raise while its peers enter backward collectives.  Reduce a
        one-element failure flag first, then gather the offending ranks only
        on the exceptional path.  This method deliberately accepts tensors
        rather than Python floats so NCCL checks remain on-device.
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"finite guard {label!r} requires a tensor, got {type(value)}")
        local_failed = (~torch.isfinite(value.detach())).any().to(
            dtype=torch.int32)
        global_failed = local_failed.clone()
        distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if distributed:
            self.global_all_reduce(global_failed, op=dist.ReduceOp.MAX)
        if not bool(global_failed.item()):
            return

        failed_ranks = [self.rank()]
        if distributed:
            gathered = [torch.zeros_like(local_failed)
                        for _ in range(dist.get_world_size())]
            self.global_all_gather(gathered, local_failed)
            failed_ranks = [
                rank for rank, failed in enumerate(gathered)
                if bool(failed.item())
            ]
        raise FloatingPointError(
            f"non-finite {label} detected on distributed rank(s) "
            f"{failed_ranks}; optimizer state was not updated")

    def reduce_sharded_gradient_tensor(self, tensor: torch.Tensor) -> None:
        """Sum a local-shard statistic across the parameter shard group."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def gradient_shard_world_size(self) -> int:
        """Number of ranks that jointly own one sharded parameter copy."""
        return dist.get_world_size() if dist.is_initialized() else 1

    # --------------------------------------------------------------------- #
    #  Gradient accumulation helper
    # --------------------------------------------------------------------- #
    @contextmanager
    def no_sync_context(self, model: nn.Module):
        """Context manager to disable gradient synchronization.

        Used during gradient accumulation steps (non-final steps) to avoid
        unnecessary distributed communication overhead.
        """
        yield

    def broadcast_from_rank0(self, obj: Any) -> Any:
        """Broadcast a Python object from rank 0 to all ranks."""
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return obj
        obj_list = [obj if self.is_rank_zero() else None]
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]

    # --------------------------------------------------------------------- #
    #  Mixed-precision helpers (may be overridden)
    # --------------------------------------------------------------------- #
    @property
    def mixed_precision_enabled(self) -> bool:
        return False

    @property
    def mixed_precision_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def training_device(self) -> torch.device:
        return (
            torch.device('cuda', self.device_id)
            if torch.cuda.is_available()
            else torch.device('cpu')
        )
