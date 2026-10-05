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

import json
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from zvla.core import BaseStrategy, VLAForwardOutput, VLAModel
from zvla.engines.utils import initialize_overwatch

overwatch = initialize_overwatch(__name__)

_FP32_SEMANTIC_BATCH_FIELDS = frozenset({
    # Public robot/action ABIs are converted into relative model coordinates
    # inside the model.  Rounding absolute values to BF16 first can erase the
    # small difference that is the supervision target.
    'current_state',
    'target_action',
    'current_state_abs',
    'target_state_abs',
    'current_eef_pose_abs',
    'target_eef_pose_abs',
    # Physical time drives RoPE and conditioner inputs.  Keep the source
    # frequency in FP32; autocast will cast only the expensive projections.
    'primitive_dt',
    'video_dt',
})


class VLATrainer:
    """VLA training orchestrator.

    Compared to the legacy ``BaseTrainRunner``:
    - No FSDP/DDP APIs are called directly; everything is delegated to
      the ``strategy``.
    - The training loop interacts with models *only* through
      ``VLAForwardOutput``.
    - Optimiser setup is extracted to ``build_optimizer_and_scheduler``.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        model: VLAModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        train_dataloader: DataLoader,
        metric: Any,
        training_cfg: dict,
    ) -> None:
        self.strategy = strategy
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.metric = metric
        self.cfg = training_cfg

        self.current_epoch = 0
        self.global_step = 0
        self._loss_sum = None
        self._loss_count = 0
        self._resume_sampler_state_loaded = False
        self._resume_epoch_step_count = 0
        self._epoch_step_count = 0
        self._fork_report = None
        self._last_checkpoint_progress = None
        self._steps_per_epoch = self._resolve_steps_per_epoch(training_cfg)

        # Gradient accumulation
        self.grad_accum_steps = max(
            1, training_cfg.get('grad_accumulation_steps', 1))
        self._accum_count = 0
        self.log_every_n_steps = max(
            1, int(training_cfg.get('log_every_n_steps', 1)))
        self.source_metrics_every_n_steps = max(
            self.log_every_n_steps,
            int(training_cfg.get('source_metrics_every_n_steps',
                                 self.log_every_n_steps)),
        )
        if self.source_metrics_every_n_steps % self.log_every_n_steps:
            raise ValueError(
                'source_metrics_every_n_steps must be a multiple of '
                'log_every_n_steps')
        self.diagnostics_every_n_steps = max(
            self.log_every_n_steps,
            int(training_cfg.get('diagnostics_every_n_steps', 200)),
        )
        self._perf_interval_start = time.perf_counter()
        self._perf_data_wait_seconds = 0.0
        self._perf_batches = 0
        self._perf_local_samples = 0
        nvtx_value = os.environ.get('ZVLA_NVTX_TRAINING', '0')
        if nvtx_value not in {'0', '1'}:
            raise ValueError('ZVLA_NVTX_TRAINING must be 0 or 1')
        self._nvtx_training = nvtx_value == '1'

        checkpoint_cfg = training_cfg.get('checkpoint', {})
        resume_probe = checkpoint_cfg.get(
            'diagnostic_fail_after_checkpoint_step_once')
        if resume_probe is not None:
            if os.environ.get('ZVLA_ENABLE_DESTRUCTIVE_RESUME_PROBE') != '1':
                raise RuntimeError(
                    'diagnostic checkpoint failure injection requires '
                    'ZVLA_ENABLE_DESTRUCTIVE_RESUME_PROBE=1')
            if isinstance(resume_probe, bool) or int(resume_probe) <= 0:
                raise ValueError(
                    'diagnostic_fail_after_checkpoint_step_once must be a '
                    'positive optimizer step')
            self._diagnostic_resume_probe_step = int(resume_probe)
        else:
            self._diagnostic_resume_probe_step = None

        # Training mode
        self.max_steps = training_cfg.get('max_steps')
        self.max_epochs = training_cfg.get('max_epochs')
        if self.max_steps is not None and self.max_epochs is not None:
            raise ValueError('Only one of max_steps or max_epochs can be set')
        if self.max_steps is None and self.max_epochs is None:
            raise ValueError('One of max_steps or max_epochs must be set')

        self.training_mode = 'step_based' if self.max_steps is not None else 'epoch_based'
        self._initialize_gradient_metric_contract()

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def fit(self) -> str:
        """Run the full training loop."""
        self.model.train()
        self.optimizer.zero_grad()
        self._reset_performance_interval()

        sampler = self._get_sampler()

        if self.training_mode == 'step_based':
            return self._run_step_based(sampler)
        else:
            return self._run_epoch_based(sampler)

    def resume_from(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        state = self.strategy.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            checkpoint_path=checkpoint_path,
        )
        root_model = self.strategy.unwrap_model(self.model)
        verify_teacher = getattr(root_model, 'verify_teacher_provenance', None)
        if verify_teacher is not None:
            verify_teacher(state.get('teacher_provenance'))
        self.global_step = state['global_step']
        self.current_epoch = state['epoch']
        trainer_state = state.get('trainer_state')
        sampler_state = self._rank_sampler_state(state.get('sampler_state'))
        rng_state = self._rank_rng_state(state.get('rng_state'))
        missing_sections = [
            name
            for name, value in (
                ('trainer_state', trainer_state),
                ('sampler_state', sampler_state),
                ('rng_state', rng_state),
            )
            if not isinstance(value, dict) or not value
        ]
        if missing_sections:
            raise KeyError(
                'training checkpoint is missing deterministic lifecycle '
                f'sections: {missing_sections}'
            )
        missing_trainer = {'_accum_count', 'epoch_step_count'} - set(
            trainer_state
        )
        required_rng = {'python', 'numpy', 'torch'}
        if torch.cuda.is_available():
            required_rng.update({'torch_cuda', 'torch_cuda_device'})
        missing_rng = required_rng - set(rng_state)
        if missing_trainer or missing_rng:
            raise KeyError(
                'incomplete deterministic resume state: '
                f'missing_trainer={sorted(missing_trainer)}, '
                f'missing_rng={sorted(missing_rng)}'
            )
        self._accum_count = int(
            trainer_state['_accum_count'])
        self._resume_epoch_step_count = int(
            trainer_state['epoch_step_count'])
        self._epoch_step_count = self._resume_epoch_step_count
        # The loaded checkpoint already durably represents this exact
        # logical position.  Remember it so a completed-epoch recovery can
        # repair a missing model-only hand-off artifact without writing the
        # (potentially 70+ GiB) distributed resume checkpoint a second time.
        self._last_checkpoint_progress = (
            int(self.global_step),
            int(self.current_epoch),
            int(self._resume_epoch_step_count),
        )
        sampler = self._get_sampler()
        if sampler is None or not hasattr(sampler, 'load_state_dict'):
            raise TypeError(
                'Zerith VA resume requires a stateful batch sampler'
            )
        sampler.load_state_dict(sampler_state)
        self._resume_sampler_state_loaded = True
        if self.strategy.is_rank_zero():
            overwatch.info(f'Resumed training from step {self.global_step}, '
                           f'epoch {self.current_epoch}')
        # RNG restoration must be the final resume action.  Logging backends,
        # sampler implementations and future lifecycle hooks may consume
        # Python/NumPy/Torch randomness; restoring earlier would make the next
        # training sample diverge despite a byte-exact checkpoint.
        self._restore_rng_state(rng_state)

    def fork_from(self, checkpoint_path: str, *, restore_optimizer: bool = True) -> None:
        """Start a new experiment schedule from a completed training state."""
        state = self.strategy.fork_from_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=checkpoint_path,
            restore_optimizer=restore_optimizer,
        )
        root_model = self.strategy.unwrap_model(self.model)
        verify_teacher = getattr(root_model, 'verify_teacher_provenance', None)
        if verify_teacher is not None:
            verify_teacher(state.get('teacher_provenance'))
        self.global_step = int(state['global_step'])
        self.current_epoch = int(state['epoch'])
        self._fork_report = {
            'checkpoint': os.path.realpath(checkpoint_path),
            'source_global_step': self.global_step,
            'source_epoch': self.current_epoch,
            'optimizer_state_restored': bool(restore_optimizer),
            'restored_parameter_states': int(
                state.get('restored_parameter_states', 0)),
            'new_parameter_states': int(state.get('new_parameter_states', 0)),
        }
        if self.strategy.is_rank_zero():
            overwatch.info(
                'Forked experiment from '
                f'step {self.global_step}, epoch {self.current_epoch}; '
                'scheduler, sampler, RNG and run identity start fresh')

    # --------------------------------------------------------------------- #
    #  Training loops
    # --------------------------------------------------------------------- #
    def _run_step_based(self, sampler) -> str:
        max_steps = self.max_steps
        # A complete final DCP may precede the model-only export. Restarting
        # repairs that export without another optimizer/data step.
        if self.global_step >= max_steps:
            self._repair_completed_model_artifact(
                enabled=self._should_export_model_step())
            return self._get_checkpoint_path()
        dataloader_iter = None
        epoch_step_count = 0
        steps_per_epoch = self._steps_per_epoch

        with tqdm(
                total=max_steps,
                desc=self.metric.get_status() if hasattr(
                    self.metric, 'get_status') else 'Training',
                leave=False,
                disable=not self.strategy.is_rank_zero(),
                initial=self.global_step,
        ) as pbar:

            while self.global_step < max_steps:
                if dataloader_iter is None:
                    if (sampler is not None and hasattr(sampler, 'set_epoch')
                            and not self._resume_sampler_state_loaded):
                        sampler.set_epoch(self.current_epoch)
                    self._set_dataset_epoch(self.current_epoch)
                    dataloader_iter = iter(self.train_dataloader)
                    epoch_step_count = self._resume_epoch_step_count
                    self._resume_epoch_step_count = 0
                    self._epoch_step_count = epoch_step_count
                    self._resume_sampler_state_loaded = False

                try:
                    batch = self._next_batch(dataloader_iter)
                except StopIteration:
                    self.current_epoch += 1
                    self._epoch_step_count = 0
                    dataloader_iter = None
                    continue

                prev_step = self.global_step
                loss = self._profiled_training_step(batch)
                self._accumulate_loss(loss)
                epoch_step_count += 1
                self._epoch_step_count = epoch_step_count

                # Only log / checkpoint / update pbar on actual optimizer steps
                if self.global_step > prev_step:
                    self.metric.commit(
                        global_step=self.global_step,
                        epoch=self.current_epoch,
                        lr=self.lr_scheduler.get_last_lr()[0],
                        loss=loss,
                    )
                    if (self._should_log() and hasattr(self.metric, 'push')):
                        pbar.set_description(self.metric.push())
                    elif hasattr(self.metric, 'get_status'):
                        pbar.set_description(
                            self.metric.get_status(), refresh=False)
                    pbar.update(1)

                    # Checkpoint
                    if self._should_save_step():
                        self._save_checkpoint(
                            export_model_artifact=self._should_export_model_step())

                # Epoch boundary for samplers whose iterator cycles forever.
                if steps_per_epoch and epoch_step_count >= steps_per_epoch:
                    self.current_epoch += 1
                    epoch_step_count = 0
                    self._epoch_step_count = 0
                    dataloader_iter = None

        return self._get_checkpoint_path()

    def _run_epoch_based(self, sampler) -> str:
        max_epochs = self.max_epochs

        # A node can fail after the final resumable DCP checkpoint becomes
        # durable but before the gathered model-only safetensors hand-off is
        # published.  On restart ``current_epoch == max_epochs`` and the
        # training loop is correctly skipped; repair only that missing or
        # stale artifact so the next stage is not permanently blocked.
        if self.current_epoch >= max_epochs:
            self._repair_completed_epoch_artifact()
            return self._get_checkpoint_path()

        with tqdm(
                total=max_epochs,
                desc='Epochs',
                leave=False,
                disable=not self.strategy.is_rank_zero(),
                initial=self.current_epoch,
        ) as epoch_pbar:

            while self.current_epoch < max_epochs:
                if (sampler is not None and hasattr(sampler, 'set_epoch')
                        and not self._resume_sampler_state_loaded):
                    sampler.set_epoch(self.current_epoch)
                self._set_dataset_epoch(self.current_epoch)

                dataloader_iter = iter(self.train_dataloader)
                epoch_step_count = self._resume_epoch_step_count
                self._resume_epoch_step_count = 0
                self._epoch_step_count = epoch_step_count
                iter_total = self._steps_per_epoch or len(
                    self.train_dataloader)

                with tqdm(
                        total=iter_total,
                        desc=f'Epoch {self.current_epoch}',
                        leave=False,
                        disable=not self.strategy.is_rank_zero(),
                ) as iter_pbar:

                    while True:
                        try:
                            batch = self._next_batch(dataloader_iter)
                        except StopIteration:
                            break

                        prev_step = self.global_step
                        loss = self._profiled_training_step(batch)

                        self._accumulate_loss(loss)
                        epoch_step_count += 1
                        self._epoch_step_count = epoch_step_count

                        # Metrics: commit on every forward, but global_step only advances on optimizer steps
                        self.metric.commit(
                            global_step=self.global_step,
                            epoch=self.current_epoch,
                            lr=self.lr_scheduler.get_last_lr()[0],
                            loss=loss,
                        )
                        if (self.global_step > prev_step
                                and self._should_log()
                                and hasattr(self.metric, 'push')):
                            iter_pbar.set_description(self.metric.push())
                        elif hasattr(self.metric, 'get_status'):
                            iter_pbar.set_description(
                                self.metric.get_status(), refresh=False)
                        iter_pbar.update(1)

                        # Step-based checkpoints should also work in
                        # epoch-based training. Only save after an optimizer
                        # step so grad accumulation cannot create duplicate
                        # checkpoints for the same global_step.
                        if self.global_step > prev_step and self._should_save_step(
                        ):
                            self._save_checkpoint(
                                export_model_artifact=self._should_export_model_step())

                        # Epoch boundary: VariableChunkBatchSampler-style
                        # samplers cycle forever, so stop explicitly at the
                        # logical per-epoch length.
                        if epoch_step_count >= iter_total:
                            break

                self._resume_sampler_state_loaded = False
                self.current_epoch += 1
                self._epoch_step_count = 0
                epoch_pbar.update(1)

                if self._should_save_epoch():
                    self._save_checkpoint(
                        export_model_artifact=bool(
                            self.cfg.get('checkpoint', {}).get(
                                'export_model_at_epoch_end', False)))

        return self._get_checkpoint_path()

    def _repair_completed_epoch_artifact(self) -> None:
        self._repair_completed_model_artifact(enabled=bool(
            self.cfg.get('checkpoint', {}).get('export_model_at_epoch_end', False)))

    def _repair_completed_model_artifact(self, *, enabled: bool) -> None:
        """Re-export a missing/corrupt final model hand-off after resume.

        Artifact validation happens on rank zero and the decision is
        broadcast before entering the collective FSDP2 state-dict gather.
        The expected file and its public symlink must both identify the exact
        restored step/epoch and carry matching safetensors metadata.
        """
        if not enabled:
            return

        needs_export = None
        if self.strategy.is_rank_zero():
            save_epoch, _ = self._checkpoint_progress()
            checkpoint_dir = os.path.join(
                str(self.metric.run_dir), 'checkpoints')
            expected = os.path.join(
                checkpoint_dir,
                f'model-step-{self.global_step:06d}-epoch-'
                f'{save_epoch:02d}.safetensors',
            )
            latest = os.path.join(
                checkpoint_dir, 'latest-checkpoint.safetensors')
            valid = False
            try:
                from safetensors import safe_open

                if (os.path.isfile(expected) and os.path.isfile(latest)
                        and os.path.realpath(latest)
                        == os.path.realpath(expected)):
                    with safe_open(
                            expected, framework='pt', device='cpu') as handle:
                        metadata = handle.metadata() or {}
                        valid = (
                            bool(handle.keys())
                            and metadata.get('artifact_kind')
                            == 'model_handoff'
                            and metadata.get('global_step')
                            == str(int(self.global_step))
                            and metadata.get('epoch')
                            == str(int(save_epoch))
                        )
            except Exception:
                # Interrupted files, stale symlinks, invalid headers, and
                # metadata mismatches all take the same safe recovery path.
                valid = False
            needs_export = not valid
            if needs_export:
                overwatch.warning(
                    'Completed training has no valid current model hand-off; '
                    'repairing the artifact from the resumable checkpoint')

        needs_export = bool(
            self.strategy.broadcast_from_rank0(needs_export))
        if needs_export:
            self._save_checkpoint(export_model_artifact=True)

    # --------------------------------------------------------------------- #
    #  Single training step
    # --------------------------------------------------------------------- #
    def _profiled_training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        if not self._nvtx_training:
            return self._training_step(batch)
        torch.cuda.nvtx.range_push('zvla.train.step')
        try:
            return self._training_step(batch)
        finally:
            torch.cuda.nvtx.range_pop()

    def _training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        is_last_accum = (self._accum_count + 1) % self.grad_accum_steps == 0
        next_step = self.global_step + 1
        total_training_steps = int(self.lr_scheduler.vla_num_training_steps)
        should_collect = is_last_accum and (
            self._should_log(next_step) or next_step == total_training_steps)
        should_collect_expensive = (
            should_collect
            and next_step % self.diagnostics_every_n_steps == 0
        )
        should_collect_source = should_collect and (
            next_step % self.source_metrics_every_n_steps == 0
            or next_step == total_training_steps)
        root_model = self.strategy.unwrap_model(self.model)
        root_model.set_train_step_context(
            next_step,
            collect_metrics=should_collect,
            collect_expensive_metrics=should_collect_expensive,
            collect_source_metrics=should_collect_source,
            total_steps=total_training_steps,
        )

        use_cuda_timing = (
            should_collect
            and self.strategy.training_device.type == 'cuda'
            and torch.cuda.is_available()
        )
        transfer_event = start_event = forward_event = backward_event = None
        gradient_sync_event = None
        clip_guard_event = adam_scheduler_event = end_event = None
        cpu_transfer_start = time.perf_counter()
        if use_cuda_timing:
            transfer_event = torch.cuda.Event(enable_timing=True)
            start_event = torch.cuda.Event(enable_timing=True)
            forward_event = torch.cuda.Event(enable_timing=True)
            backward_event = torch.cuda.Event(enable_timing=True)
            gradient_sync_event = torch.cuda.Event(enable_timing=True)
            clip_guard_event = torch.cuda.Event(enable_timing=True)
            adam_scheduler_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            transfer_event.record()

        batch = self._move_batch_to_device(batch,
                                           self.strategy.training_device)
        if self.strategy.mixed_precision_enabled:
            batch = self._convert_batch_dtype(
                batch, self.strategy.mixed_precision_dtype)
        self._perf_local_samples += self._infer_batch_size(batch)
        cpu_compute_start = time.perf_counter()
        if use_cuda_timing:
            start_event.record()

        with torch.autocast(
                self.strategy.training_device.type,
                dtype=self.strategy.mixed_precision_dtype,
                enabled=self.strategy.mixed_precision_enabled,
        ):
            output = self.model(**batch)
        cpu_forward_end = time.perf_counter()
        if use_cuda_timing:
            forward_event.record()

        # Strict interface validation
        if not isinstance(output, VLAForwardOutput):
            raise TypeError(
                f'Model output must be VLAForwardOutput, got {type(output)}. '
                f'Please ensure your model\'s forward() returns VLAForwardOutput.'
            )

        # Scale loss for gradient accumulation
        loss = output.loss / self.grad_accum_steps

        # A non-final accumulation backward has no synchronization, so guard
        # it before ranks can diverge. On the normal (and production) final
        # accumulation, let every rank enter the same backward collectives and
        # combine the loss/gradient guard below. This removes one blocking
        # global scalar collective from every optimizer step.
        if not is_last_accum:
            try:
                self.strategy.assert_finite(
                    loss,
                    label=f"training loss before backward at step {next_step}",
                )
            except FloatingPointError:
                self.strategy.zero_grad(self.model, self.optimizer)
                raise

        # Backward with no_sync optimization for non-final accumulation steps
        sync_ctx = self.strategy.no_sync_context(
            self.model) if not is_last_accum else nullcontext()
        with sync_ctx:
            loss.backward()

        cpu_backward_end = time.perf_counter()
        if use_cuda_timing:
            backward_event.record()

        return_loss = loss * self.grad_accum_steps

        if is_last_accum:
            # Measure the complete post-backward update path (gradient norm,
            # clipping and Adam), not Adam alone.
            cpu_update_start = time.perf_counter()

            # Some backends reduce gradients asynchronously and expose FP32
            # optimizer shards only after the final reduce-scatter completes.
            # This boundary must precede delayed-group suppression, norm
            # calculation, clipping, and every optimizer read of ``.grad``.
            self.strategy.finish_gradient_sync(self.model)
            cpu_gradient_sync_end = time.perf_counter()
            if use_cuda_timing:
                gradient_sync_event.record()

            # ``grad=None`` keeps delayed groups outside Adam state creation
            # and decoupled weight decay, including across resume boundaries.
            self._suppress_delayed_optimizer_groups(next_step)

            # Telemetry reads the exact gradients that the optimizer is
            # allowed to consume.  It is deliberately not an input to the
            # clipping path: a logging regression must never change training
            # mathematics or weaken the backend's finite-gradient guard.
            grad_norms = self._compute_grad_norms() if should_collect else {}

            # Gradient clipping via strategy
            grad_clip = self.cfg.get('grad_clip')
            try:
                gradient_norm = self.strategy.clip_grad_norm_and_assert_finite(
                    self.model,
                    float('inf') if grad_clip is None else float(grad_clip),
                    return_loss,
                    label=(
                        f"loss or gradient norm before optimizer step "
                        f"{next_step}"
                    ),
                )
            except FloatingPointError:
                self.strategy.zero_grad(self.model, self.optimizer)
                raise
            cpu_clip_guard_end = time.perf_counter()
            if use_cuda_timing:
                clip_guard_event.record()

            # Optimiser state/shape errors are fatal. Replacing an optimizer
            # here would make resume behavior silently non-reproducible.
            self.strategy.optimizer_step(self.model, self.optimizer)
            self.lr_scheduler.step()
            cpu_adam_scheduler_end = time.perf_counter()
            if use_cuda_timing:
                adam_scheduler_event.record()
            self.global_step += 1
            self.strategy.after_optimizer_step(
                self.model,
                self.optimizer,
                root_model,
                self.global_step,
            )
            self.strategy.zero_grad(self.model, self.optimizer)
            cpu_ema_end = time.perf_counter()
            if use_cuda_timing:
                end_event.record()

            if should_collect:
                if use_cuda_timing:
                    end_event.synchronize()
                    step_timings = {
                        'perf_batch_to_device_ms':
                        transfer_event.elapsed_time(start_event),
                        'perf_forward_backward_ms':
                        start_event.elapsed_time(backward_event),
                        'perf_post_backward_update_ms':
                        backward_event.elapsed_time(end_event),
                        # This is the exposed, non-overlapped tail after
                        # backward. AG/RS work successfully hidden under
                        # compute is intentionally not double-counted.
                        'perf_nccl_gradient_sync_tail_ms':
                        backward_event.elapsed_time(gradient_sync_event),
                        'perf_forward_ms':
                        start_event.elapsed_time(forward_event),
                        'perf_backward_ms':
                        forward_event.elapsed_time(backward_event),
                        'perf_gradient_clip_guard_ms':
                        backward_event.elapsed_time(clip_guard_event),
                        'perf_adam_scheduler_ms':
                        clip_guard_event.elapsed_time(adam_scheduler_event),
                        'perf_ema_ms':
                        adam_scheduler_event.elapsed_time(end_event),
                    }
                else:
                    step_timings = {
                        'perf_batch_to_device_ms':
                        (cpu_compute_start - cpu_transfer_start) * 1000.0,
                        'perf_forward_backward_ms':
                        (cpu_backward_end - cpu_compute_start) * 1000.0,
                        'perf_post_backward_update_ms':
                        (cpu_ema_end - cpu_update_start) * 1000.0,
                        'perf_nccl_gradient_sync_tail_ms':
                        (cpu_gradient_sync_end - cpu_update_start) * 1000.0,
                        'perf_forward_ms':
                        (cpu_forward_end - cpu_compute_start) * 1000.0,
                        'perf_backward_ms':
                        (cpu_backward_end - cpu_forward_end) * 1000.0,
                        'perf_gradient_clip_guard_ms':
                        (cpu_clip_guard_end - cpu_update_start) * 1000.0,
                        'perf_adam_scheduler_ms':
                        (cpu_adam_scheduler_end - cpu_clip_guard_end) * 1000.0,
                        'perf_ema_ms':
                        (cpu_ema_end - cpu_adam_scheduler_end) * 1000.0,
                    }
                performance = self._collect_performance_metrics(step_timings)
                learning_rates = self._optimizer_group_learning_rates()
                model_metrics = dict(output.metrics)
                model_metrics['loss_total'] = return_loss.detach()
                if (self.global_step % self.source_metrics_every_n_steps
                        and self.global_step != total_training_steps):
                    model_metrics = {
                        key: value for key, value in model_metrics.items()
                        if '_source_' not in key
                        and not key.startswith('source_')
                    }
                reduced_metrics = self._reduce_scalar_metrics(model_metrics)
                return_loss = reduced_metrics['loss_total']
                if hasattr(self.metric, 'commit'):
                    self.metric.commit(
                        **grad_norms,
                        **learning_rates,
                        **performance,
                        **reduced_metrics,
                    )

        self._accum_count += 1

        # Return unscaled loss for logging
        return return_loss

    # --------------------------------------------------------------------- #
    #  Checkpoint helpers
    # --------------------------------------------------------------------- #
    def _should_save_step(self) -> bool:
        checkpoint_cfg = self.cfg.get('checkpoint', {})
        if self.global_step <= 0:
            return False

        explicit_steps = checkpoint_cfg.get('save_steps')
        if explicit_steps is not None and self.global_step in set(
                int(value) for value in explicit_steps):
            return True

        if self._should_export_model_step():
            return True

        interval = checkpoint_cfg.get('save_step_interval', 10000)
        if interval is None or interval <= 0:
            return False
        return (self.global_step % interval) == 0

    def _should_export_model_step(self) -> bool:
        checkpoint_cfg = self.cfg.get('checkpoint', {})
        explicit = checkpoint_cfg.get('export_model_steps')
        if explicit is not None:
            steps = tuple(int(value) for value in explicit)
            if any(value <= 0 for value in steps) or len(set(steps)) != len(steps):
                raise ValueError(
                    'checkpoint.export_model_steps must contain unique positive steps')
            return self.global_step > 0 and self.global_step in set(steps)
        interval = checkpoint_cfg.get('export_model_step_interval')
        if interval is None or int(interval) <= 0:
            return False
        return self.global_step > 0 and self.global_step % int(interval) == 0

    def _should_save_epoch(self) -> bool:
        interval = self.cfg.get('checkpoint', {}).get('save_epoch_interval', 1)
        if interval is None or interval <= 0:
            return False
        return (self.current_epoch % interval) == 0

    def _should_log(self, step: Optional[int] = None) -> bool:
        if step is None:
            step = self.global_step
        return step > 0 and (step % self.log_every_n_steps) == 0

    def _save_checkpoint(self, *, export_model_artifact: bool = False) -> None:
        checkpoint_started = time.perf_counter()
        save_epoch, save_epoch_step_count = self._checkpoint_progress()
        progress = (
            int(self.global_step),
            int(save_epoch),
            int(save_epoch_step_count),
        )
        checkpoint_cfg = self.cfg.get('checkpoint', {})
        max_keep = int(checkpoint_cfg.get('max_keep', 0))
        if export_model_artifact:
            max_keep = int(
                checkpoint_cfg.get('max_keep_at_epoch_end', max_keep))

        # A final optimizer step can satisfy both the periodic interval and
        # the epoch boundary.  Its logical progress is identical in both
        # cases; do not write the same 70+ GiB distributed checkpoint twice.
        save_resume_checkpoint = progress != self._last_checkpoint_progress
        if save_resume_checkpoint:
            avg_loss = None
            if self._loss_count:
                avg_loss = (self._loss_sum / self._loss_count).item()
                self._loss_sum = None
                self._loss_count = 0
            self.strategy.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                global_step=self.global_step,
                epoch=save_epoch,
                run_dir=self.metric.run_dir,
                train_loss=avg_loss,
                only_trainable=not checkpoint_cfg.get(
                    'save_full_model', True),
                max_keep=max_keep,
                checkpoint_format=checkpoint_cfg.get('format', 'full'),
                extra_state=self._build_extra_state(
                    save_epoch=save_epoch,
                    save_epoch_step_count=save_epoch_step_count,
                ),
            )
            self._last_checkpoint_progress = progress
            self._publish_geometry_consumer_ack(
                save_epoch=save_epoch,
                save_epoch_step_count=save_epoch_step_count,
            )
        elif export_model_artifact:
            self.strategy.prune_checkpoints(
                self.metric.run_dir,
                checkpoint_format=checkpoint_cfg.get('format', 'full'),
                max_keep=max_keep,
            )

        if export_model_artifact:
            self.strategy.save_model_artifact(
                model=self.model,
                global_step=self.global_step,
                epoch=save_epoch,
                run_dir=self.metric.run_dir,
                max_keep=int(checkpoint_cfg.get('model_artifact_max_keep', 1)),
            )
        self.strategy.barrier()
        if self.strategy.is_rank_zero():
            overwatch.info(
                'Checkpoint pause: '
                f'{time.perf_counter() - checkpoint_started:.3f}s '
                f'(resume={save_resume_checkpoint}, '
                f'artifact={export_model_artifact}, '
                f'global_step={self.global_step})')
        if self.global_step == self._diagnostic_resume_probe_step:
            # The opt-in diagnostic failure occurs only after every rank has
            # observed the atomically complete checkpoint. A coordinated
            # relaunch must resume at this exact step, so this branch is not
            # reached twice. Broken checkpoint discovery therefore fails the
            # probe instead of yielding a false success.
            raise RuntimeError(
                'intentional Zerith VA kill-and-resume probe after durable '
                f'checkpoint step {self.global_step}')

    def _publish_geometry_consumer_ack(
        self,
        *,
        save_epoch: int,
        save_epoch_step_count: int,
    ) -> None:
        """Publish a rolling-VGGT watermark only after a durable checkpoint.

        The producer/janitor may reclaim a geometry shard only from these
        acknowledgements and only while the referenced checkpoint still has a
        ``.complete`` marker.  A training batch finishing in memory is never a
        deletion boundary.
        """
        ack_dir = self.cfg.get('checkpoint', {}).get(
            'geometry_consumer_ack_dir')
        if not ack_dir:
            return
        sampler_state = self._build_logical_sampler_state(
            save_epoch=save_epoch,
            save_epoch_step_count=save_epoch_step_count,
        )
        error = None
        if self.strategy.is_rank_zero():
            try:
                checkpoint = os.path.realpath(self._get_checkpoint_path())
                complete = os.path.join(checkpoint, '.complete')
                if not os.path.isfile(complete):
                    raise FileNotFoundError(
                        'rolling geometry acknowledgement requires a durable '
                        f'checkpoint marker: {complete}')
                if not isinstance(sampler_state, dict):
                    raise TypeError(
                        'rolling geometry acknowledgement requires sampler state')
                expected_partition = (
                    'global_anchor_stream_exact_once_padded_v2')
                if sampler_state.get('partition_format') != expected_partition:
                    raise ValueError(
                        'rolling geometry acknowledgement requires the '
                        f'exact-once sampler, got '
                        f'{sampler_state.get("partition_format")!r}')
                payload = {
                    'format': 'fam-wam-geometry-consumer-ack-v2',
                    'checkpoint_path': checkpoint,
                    'checkpoint_complete_marker': complete,
                    'global_step': int(self.global_step),
                    'epoch': int(save_epoch),
                    'next_batch_idx': int(save_epoch_step_count),
                    'sampler_partition_format': expected_partition,
                    'source_anchors': int(sampler_state['source_anchors']),
                    'global_batch_size': int(
                        sampler_state['global_batch_size']),
                    'steps_per_rank': int(sampler_state['steps_per_rank']),
                    'schedule_seed': int(sampler_state['seed']),
                    'schedule_epoch': int(
                        sampler_state['shard_order_epoch']),
                    'shard_order_sha256': str(
                        sampler_state['shard_order_sha256']),
                    'world_size': int(self.strategy.world_size()),
                    'durability_rule': (
                        'published_only_after_atomic_checkpoint_complete'),
                }
                directory = os.path.realpath(str(ack_dir))
                os.makedirs(directory, exist_ok=True)
                name = (
                    f'checkpoint-step-{self.global_step:08d}-'
                    f'epoch-{save_epoch:04d}-cursor-'
                    f'{save_epoch_step_count:08d}.json')

                def publish(path: str) -> None:
                    temporary = path + f'.tmp.{os.getpid()}'
                    with open(temporary, 'w', encoding='utf-8') as handle:
                        json.dump(payload, handle, indent=2, sort_keys=True)
                        handle.write('\n')
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, path)

                publish(os.path.join(directory, name))
                publish(os.path.join(directory, 'latest.json'))
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        error = self.strategy.broadcast_from_rank0(error)
        if error is not None:
            raise RuntimeError(
                'failed to publish rolling geometry checkpoint '
                f'acknowledgement: {error}')

    def _get_checkpoint_path(self) -> str:
        from pathlib import Path
        sharded_link = Path(
            self.metric.run_dir) / 'checkpoints' / 'latest-checkpoint'
        if sharded_link.exists():
            return str(sharded_link)
        return str(
            Path(self.metric.run_dir) / 'checkpoints' / 'latest-checkpoint.pt')

    # --------------------------------------------------------------------- #
    #  Utilities
    # --------------------------------------------------------------------- #
    def _accumulate_loss(self, loss: torch.Tensor) -> None:
        loss = loss.detach().float()
        if self._loss_sum is None:
            self._loss_sum = loss.clone()
        else:
            self._loss_sum.add_(loss)
        self._loss_count += 1

    def _initialize_gradient_metric_contract(self) -> None:
        """Seal optimizer-main-gradient ownership before any checkpoint load.

        Megatron FSDP replaces application BF16 parameters in AdamW with FP32
        main-shard parameters.  Parameter identity therefore cannot be used to
        recover architectural ownership after strategy setup.  The optimizer
        builder stamps every group while identity is still available; cache
        that immutable contract here so loading an older optimizer state cannot
        remove the metadata used by training-time telemetry.
        """
        root_model = self.strategy.unwrap_model(self.model)
        groups = root_model.training_metric_parameter_groups()
        if not isinstance(groups, Mapping) or not groups:
            raise RuntimeError(
                'training_metric_parameter_groups() must return a non-empty mapping'
            )
        self._gradient_metric_names = tuple(groups)
        roles = root_model.training_metric_group_roles()
        if not isinstance(roles, Mapping):
            raise TypeError('training_metric_group_roles() must return a mapping')
        if set(roles) != set(self._gradient_metric_names):
            raise ValueError(
                'training metric role keys must exactly match parameter group '
                f'keys: groups={sorted(self._gradient_metric_names)}, '
                f'roles={sorted(roles)}')
        allowed_roles = {'world', 'vision', 'action'}
        normalized_roles = {}
        for name in self._gradient_metric_names:
            values = tuple(roles[name])
            if not values or len(values) != len(set(values)):
                raise ValueError(
                    f'training metric group {name!r} must declare unique roles')
            unknown = set(values) - allowed_roles
            if unknown:
                raise ValueError(
                    f'training metric group {name!r} has unknown roles '
                    f'{sorted(unknown)}')
            normalized_roles[name] = frozenset(values)
        self._gradient_metric_roles = normalized_roles

        optimizer_groups = {}
        for index, group in enumerate(self.optimizer.param_groups):
            full_name = str(group.get('name', f'group_{index}'))
            optimizer_name = full_name.split('/', 1)[0]
            metric_name = group.get('metric_group')
            if metric_name not in groups:
                raise ValueError(
                    f'optimizer group {full_name!r} has invalid semantic metric '
                    f'ownership {metric_name!r}')
            previous = optimizer_groups.setdefault(optimizer_name, metric_name)
            if previous != metric_name:
                raise ValueError(
                    f'optimizer subgroups for {optimizer_name!r} disagree on '
                    'semantic metric ownership')
        self._optimizer_metric_group_by_name = optimizer_groups

    def _compute_grad_norms(self) -> Dict[str, torch.Tensor]:
        """Compute exact norms from the gradients AdamW will consume.

        Reading ``root_model.parameters()`` is incorrect for mixed-precision
        Megatron FSDP: most compute parameters have ``grad=None`` after MCore
        moves the synchronized gradient into an FP32 optimizer main shard.
        Optimizer groups are the single source of truth for clipping, finite
        checks, delayed updates and telemetry.
        """
        names = self._gradient_metric_names
        name_to_index = {name: index for index, name in enumerate(names)}
        device = self.strategy.training_device
        norm_square = torch.zeros(len(names), dtype=torch.float32, device=device)
        zero_shards = torch.zeros_like(norm_square)
        missing_shards = torch.zeros_like(norm_square)
        nonfinite_shards = torch.zeros_like(norm_square)
        eligible_shards = torch.zeros_like(norm_square)
        empty_shards = torch.zeros_like(norm_square)
        gradients_by_group = [[] for _ in names]
        sharded = False

        next_step = self.global_step + 1
        for group_index, group in enumerate(self.optimizer.param_groups):
            if next_step <= int(group.get('update_after_step', 0)):
                continue
            full_name = str(group.get('name', f'group_{group_index}'))
            optimizer_name = full_name.split('/', 1)[0]
            try:
                metric_name = self._optimizer_metric_group_by_name[optimizer_name]
            except KeyError as exc:
                raise RuntimeError(
                    f'optimizer group {full_name!r} was not present in the sealed '
                    'gradient metric contract') from exc
            index = name_to_index[metric_name]
            parameters = tuple(group['params'])
            gradients = gradients_by_group[index]
            for parameter in parameters:
                if hasattr(parameter, 'to_local'):
                    sharded = True
                    local_parameter = parameter.to_local()
                else:
                    local_parameter = parameter
                # Uneven DTensor sharding deliberately leaves some ranks with
                # a zero-sized local optimizer shard.  MCore represents those
                # shards with ``grad=None`` and skips their Adam state; they
                # are not missing gradients and must not dilute this metric.
                if local_parameter.numel() == 0:
                    empty_shards[index] += 1
                    continue
                eligible_shards[index] += 1
                gradient = parameter.grad
                if gradient is None:
                    missing_shards[index] += 1
                    continue
                gradient = gradient.detach()
                if hasattr(gradient, 'to_local'):
                    sharded = True
                    gradient = gradient.to_local()
                gradients.append(gradient)
        for index, gradients in enumerate(gradients_by_group):
            if not gradients:
                continue
            norms = torch.stack(
                torch._foreach_norm(gradients, 2, dtype=torch.float32))
            norm_square[index] = norms.square().sum()
            zero_shards[index] = (norms == 0).sum()
            nonfinite_shards[index] = (~torch.isfinite(norms)).sum()

        world_size = self._world_size()
        if world_size > 1:
            # Norm squares and diagnostics share one small collective.  The
            # previous two scalar collectives serialized the communication
            # stream on every logging step without adding information.
            statistics = torch.stack((
                norm_square,
                zero_shards,
                missing_shards,
                nonfinite_shards,
                eligible_shards,
                empty_shards,
            ))
            # Process-group selection must be identical on every rank.  In
            # particular, never choose WORLD versus the shard communicator
            # from a rank-local DTensor observation: uneven optimizer shards
            # are allowed to have different local representations.  Every
            # strategy defines one deterministic gradient-shard group.
            self.strategy.reduce_sharded_gradient_tensor(statistics)
            if not sharded:
                statistics[0].div_(
                    self.strategy.gradient_shard_world_size())
            (norm_square, zero_shards, missing_shards, nonfinite_shards,
             eligible_shards, empty_shards) = statistics

        norms = norm_square.sqrt()
        result = {
            f'grad_norm_{name}': norms[index]
            for index, name in enumerate(names)
        }

        def _role_norm(indices: Tuple[int, ...]) -> torch.Tensor:
            if not indices:
                return norm_square.new_zeros(())
            return norm_square[list(indices)].sum().sqrt()

        result.update({
            'grad_norm_total': norm_square.sum().sqrt(),
            **{
                f'grad_norm_{role}': _role_norm(tuple(
                    index for index, name in enumerate(names)
                    if role in self._gradient_metric_roles[name]
                ))
                for role in ('world', 'vision', 'action')
            },
        })
        eligible_denominator = eligible_shards.sum().clamp_min(1.0)
        all_shards_denominator = (
            eligible_shards.sum() + empty_shards.sum()).clamp_min(1.0)
        result.update({
            'grad_zero_shard_fraction': (
                zero_shards.sum() / eligible_denominator),
            'grad_missing_shard_fraction': (
                missing_shards.sum() / eligible_denominator),
            'grad_nonfinite_shard_fraction': (
                nonfinite_shards.sum() / eligible_denominator),
            'grad_nonfinite_shard_count': nonfinite_shards.sum(),
            'grad_empty_shard_fraction': (
                empty_shards.sum() / all_shards_denominator),
        })
        return result

    def _compute_grad_norm(self) -> torch.Tensor:
        """Compute global gradient norm across all trainable parameters."""
        return self._compute_grad_norms()['grad_norm_total']

    def _optimizer_group_learning_rates(self) -> Dict[str, float]:
        grouped: Dict[str, float] = {}
        for index, group in enumerate(self.optimizer.param_groups):
            full_name = str(group.get('name', f'group_{index}'))
            name = full_name.split('/', 1)[0]
            update_after = int(group.get('update_after_step', 0))
            current_step = int(getattr(self, 'global_step', 0))
            learning_rate = (
                0.0 if current_step <= update_after else float(group['lr']))
            if name in grouped and not math.isclose(
                    grouped[name], learning_rate, rel_tol=1e-9, abs_tol=0.0):
                raise RuntimeError(
                    f'optimizer subgroups for {name!r} have different learning rates'
                )
            grouped[name] = learning_rate
        return {f'lr_{name}': value for name, value in grouped.items()}

    def _suppress_delayed_optimizer_groups(self, step: int) -> None:
        """Keep delayed groups completely outside early optimizer updates."""
        for group in self.optimizer.param_groups:
            if step <= int(group.get('update_after_step', 0)):
                for parameter in group['params']:
                    parameter.grad = None

    def _reduce_scalar_metrics(
        self,
        metrics: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        if not metrics:
            return {}
        keys = tuple(sorted(metrics))
        values = []
        for key in keys:
            value = metrics[key]
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if hasattr(value, 'to_local'):
                    value = value.to_local()
                if value.numel() != 1:
                    raise ValueError(
                        f'training metric {key!r} must be scalar, got {tuple(value.shape)}'
                    )
                value = value.float().reshape(())
            elif isinstance(value, (int, float)):
                value = torch.tensor(
                    float(value),
                    dtype=torch.float32,
                    device=self.strategy.training_device,
                )
            else:
                raise TypeError(
                    f'training metric {key!r} has unsupported type {type(value)}'
                )
            values.append(value.to(device=self.strategy.training_device))
        packed = torch.stack(values)
        world_size = self._world_size()
        if self._world_size() > 1:
            self.strategy.global_all_reduce(packed, op=dist.ReduceOp.SUM)
        summed = {key: packed[index] for index, key in enumerate(keys)}
        reduced = {
            key: value / world_size
            for key, value in summed.items()
            if not key.endswith("__numerator")
            and not key.endswith("__denominator")
            and not key.endswith("__sum")
        }
        for key, numerator in summed.items():
            if not key.endswith("__numerator"):
                continue
            prefix = key[:-len("__numerator")]
            denominator_key = prefix + "__denominator"
            if denominator_key not in summed:
                raise KeyError(
                    f"packed ratio metric {key!r} has no denominator")
            denominator = summed[denominator_key]
            # A source may be absent from a particular 500-step telemetry
            # sample.  Omitting it is truthful; logging a fabricated zero loss
            # makes sparse natural mixtures look artificially good.
            if float(denominator.item()) > 0.0:
                reduced[prefix] = numerator / denominator
        for key, value in summed.items():
            if key.endswith("__sum"):
                reduced[key[:-len("__sum")]] = value
        return reduced

    def _next_batch(self, dataloader_iter):
        started = time.perf_counter()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            raise
        self._perf_data_wait_seconds += time.perf_counter() - started
        self._perf_batches += 1
        return batch

    def _reset_performance_interval(self) -> None:
        self._perf_interval_start = time.perf_counter()
        self._perf_data_wait_seconds = 0.0
        self._perf_batches = 0
        self._perf_local_samples = 0
        if (torch.cuda.is_available()
                and self.strategy.training_device.type == 'cuda'):
            torch.cuda.reset_peak_memory_stats(self.strategy.training_device)

    def _collect_performance_metrics(
        self,
        step_timings: Mapping[str, float],
    ) -> Dict[str, torch.Tensor]:
        elapsed = time.perf_counter() - self._perf_interval_start
        data_wait_total = self._perf_data_wait_seconds
        data_wait = data_wait_total / max(self._perf_batches, 1)
        device = self.strategy.training_device
        if torch.cuda.is_available() and device.type == 'cuda':
            allocated_gib = torch.cuda.max_memory_allocated(device) / 2**30
            reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
        else:
            allocated_gib = reserved_gib = 0.0
        timing_keys = (
            'perf_batch_to_device_ms',
            'perf_forward_backward_ms',
            'perf_post_backward_update_ms',
            'perf_nccl_gradient_sync_tail_ms',
            'perf_forward_ms',
            'perf_backward_ms',
            'perf_gradient_clip_guard_ms',
            'perf_adam_scheduler_ms',
            'perf_ema_ms',
        )
        local_timings = tuple(float(step_timings[key]) for key in timing_keys)
        local_values = torch.tensor(
            (elapsed, data_wait, data_wait_total,
             float(self._perf_local_samples), *local_timings,
             allocated_gib, reserved_gib),
            dtype=torch.float64,
            device=device,
        )
        world_size = self._world_size()
        if world_size > 1:
            gathered = torch.empty(
                world_size * local_values.numel(),
                dtype=local_values.dtype,
                device=device,
            )
            # One tiny all-gather replaces the former max + sum collectives and
            # retains rank identity. This makes a recurring slow node visible
            # without adding another synchronization to the training path.
            self.strategy.global_all_gather_into_tensor(
                gathered, local_values)
            rank_values = gathered.reshape(world_size, local_values.numel())
        else:
            rank_values = local_values.unsqueeze(0)
        maxima, maximum_ranks = rank_values.max(dim=0)
        sums = rank_values.sum(dim=0)
        p95_index = max(0, math.ceil(0.95 * world_size) - 1)
        timing_max = {
            key: maxima[4 + index]
            for index, key in enumerate(timing_keys)
        }
        timing_mean = {
            key: sums[4 + index] / world_size
            for index, key in enumerate(timing_keys)
        }
        timing_p95 = {
            key: rank_values[:, 4 + index].sort().values[p95_index]
            for index, key in enumerate(timing_keys)
        }
        timing_slowest_ranks = {
            key: maximum_ranks[4 + index]
            for index, key in enumerate(timing_keys)
        }
        local_world_size = max(1, int(os.environ.get('LOCAL_WORLD_SIZE', '1')))
        result = {
            'perf_global_samples_per_second':
            sums[3] / maxima[0].clamp_min(1e-9),
            'perf_interval_seconds': maxima[0],
            'perf_data_wait_ms': maxima[1] * 1000.0,
            'perf_data_wait_fraction': maxima[2] / maxima[0].clamp_min(1e-9),
            'perf_interval_slowest_global_rank': maximum_ranks[0],
            'perf_interval_slowest_node_rank':
            torch.div(
                maximum_ranks[0], local_world_size, rounding_mode='floor'),
            'perf_interval_slowest_local_rank':
            maximum_ranks[0].remainder(local_world_size),
            'perf_data_wait_slowest_global_rank': maximum_ranks[2],
            'perf_data_wait_slowest_node_rank':
            torch.div(
                maximum_ranks[2], local_world_size, rounding_mode='floor'),
            'perf_data_wait_slowest_local_rank':
            maximum_ranks[2].remainder(local_world_size),
            'perf_forward_backward_ms':
            timing_max['perf_forward_backward_ms'],
            # Keep the historical optimizer key for dashboards, while the
            # explicit name records its corrected, wider timing boundary.
            'perf_post_backward_update_ms':
            timing_max['perf_post_backward_update_ms'],
            'perf_optimizer_ms':
            timing_max['perf_post_backward_update_ms'],
            'perf_peak_allocated_gib': maxima[-2],
            'perf_peak_reserved_gib': maxima[-1],
        }
        for key in timing_keys:
            result[key] = timing_max[key]
            result[f'{key}_rank_mean'] = timing_mean[key]
            result[f'{key}_rank_p95'] = timing_p95[key]
            result[f'{key}_max_over_mean'] = (
                timing_max[key] / timing_mean[key].clamp_min(1e-9))
            slowest_rank = timing_slowest_ranks[key]
            result[f'{key}_slowest_global_rank'] = slowest_rank
            result[f'{key}_slowest_node_rank'] = torch.div(
                slowest_rank, local_world_size, rounding_mode='floor')
            result[f'{key}_slowest_local_rank'] = slowest_rank.remainder(
                local_world_size)
        self._reset_performance_interval()
        return result

    @staticmethod
    def _infer_batch_size(batch: Mapping[str, Any]) -> int:
        robot = batch.get('robot')
        if isinstance(robot, Mapping):
            batch = robot
        preferred = batch.get('observation_latents')
        if isinstance(preferred, torch.Tensor) and preferred.ndim:
            return int(preferred.shape[0])
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim:
                return int(value.shape[0])
            if isinstance(value, Mapping):
                try:
                    return VLATrainer._infer_batch_size(value)
                except ValueError:
                    continue
        raise ValueError('cannot infer batch size for performance telemetry')

    @staticmethod
    def _world_size() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def _get_sampler(self):
        """Return the sampler or batch_sampler attached to the dataloader."""
        bs = getattr(self.train_dataloader, 'batch_sampler', None)
        if bs is not None and hasattr(bs, 'set_epoch'):
            return bs
        s = getattr(self.train_dataloader, 'sampler', None)
        if s is not None and hasattr(s, 'set_epoch'):
            return s
        return None

    def _set_dataset_epoch(self, epoch: int) -> None:
        dataset = getattr(self.train_dataloader, 'dataset', None)
        if dataset is not None and hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(int(epoch))

    def _build_extra_state(
        self,
        save_epoch: Optional[int] = None,
        save_epoch_step_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        if save_epoch is None or save_epoch_step_count is None:
            save_epoch, save_epoch_step_count = self._checkpoint_progress()
        sampler_state = self._build_logical_sampler_state(
            save_epoch=save_epoch,
            save_epoch_step_count=save_epoch_step_count,
        )
        rng_state = self._capture_rng_state()
        if dist.is_available() and dist.is_initialized():
            per_rank_sampler = [None] * dist.get_world_size()
            self.strategy.global_all_gather_object(
                per_rank_sampler, sampler_state)
            sampler_state = {'per_rank': per_rank_sampler}
            per_rank = [None] * dist.get_world_size()
            self.strategy.global_all_gather_object(per_rank, rng_state)
            rng_state = {'per_rank': per_rank}
        return {
            'run_id':
            getattr(self.metric, 'run_id', None),
            'wandb_run_id':
            getattr(self.metric, 'wandb_run_id', None),
            'optimizer_implementation':
            getattr(self.optimizer, 'implementation_version',
                    type(self.optimizer).__name__),
            'run_seed':
            self._run_seed(),
            'action_schema_fingerprint':
            self.cfg.get('action_schema_fingerprint'),
            'camera_schema_fingerprint':
            self.cfg.get('camera_schema_fingerprint'),
            'training_contract_fingerprint':
            self.cfg.get('training_contract_fingerprint'),
            'initialization':
            self.cfg.get('initialization'),
            'fork_from':
            self._fork_report,
            'trainer_state': {
                '_accum_count': self._accum_count,
                'epoch_step_count': int(save_epoch_step_count),
            },
            'sampler_state':
            sampler_state,
            'rng_state':
            rng_state,
            'scheduler_meta':
            self._scheduler_meta(),
            'teacher_provenance':
            getattr(
                self.strategy.unwrap_model(self.model),
                'teacher_provenance',
                None,
            ),
        }

    def _current_epoch_step_count(self) -> int:
        return int(self._epoch_step_count)

    def _checkpoint_progress(self) -> Tuple[int, int]:
        """Return epoch and in-epoch batch index to store in checkpoints."""
        save_epoch = int(self.current_epoch)
        save_epoch_step_count = int(self._epoch_step_count)
        steps_per_epoch = self._steps_per_epoch
        if steps_per_epoch and save_epoch_step_count >= steps_per_epoch:
            save_epoch += save_epoch_step_count // steps_per_epoch
            save_epoch_step_count = save_epoch_step_count % steps_per_epoch
        return save_epoch, save_epoch_step_count

    def _build_logical_sampler_state(
        self,
        save_epoch: int,
        save_epoch_step_count: int,
    ) -> Optional[Dict[str, Any]]:
        sampler = self._get_sampler()
        if sampler is not None and hasattr(sampler, 'state_dict'):
            if hasattr(sampler, 'logical_state_dict'):
                state = sampler.logical_state_dict(
                    epoch=save_epoch,
                    next_batch_idx=save_epoch_step_count,
                )
            else:
                state = sampler.state_dict()
            # DataLoader workers prefetch from the batch sampler, so the
            # sampler's cycle/next_batch_idx can run ahead of batches that
            # have actually completed training. Resume from trainer-owned
            # logical progress instead.
            state['epoch'] = int(save_epoch)
            if self._steps_per_epoch:
                state['cycle'] = 0
            else:
                state['cycle'] = int(state.get('cycle', 0))
            state['next_batch_idx'] = int(save_epoch_step_count)
            return state
        return None

    def _resolve_steps_per_epoch(self, training_cfg: dict) -> Optional[int]:
        configured = training_cfg.get('steps_per_epoch')
        if configured is not None:
            return int(configured)
        try:
            steps = len(self.train_dataloader)
        except (TypeError, AttributeError):
            return None
        if steps <= 0:
            return None
        if dist.is_available() and dist.is_initialized(
        ) and dist.get_world_size() > 1:
            device = (
                torch.device('cuda', torch.cuda.current_device())
                if torch.cuda.is_available() else torch.device('cpu'))
            step_tensor = torch.tensor(steps, dtype=torch.long, device=device)
            self.strategy.global_all_reduce(
                step_tensor, op=dist.ReduceOp.MIN)
            steps = int(step_tensor.item())
        return int(steps)

    def _scheduler_meta(self) -> Dict[str, Any]:
        if self.lr_scheduler is None:
            return {}
        keys = (
            'vla_scheduler_type',
            'vla_num_training_steps',
            'vla_num_warmup_steps',
            'vla_steps_per_epoch',
            'vla_grad_accum_steps',
        )
        return {
            key.removeprefix('vla_'): getattr(self.lr_scheduler, key)
            for key in keys if hasattr(self.lr_scheduler, key)
        }

    def _run_seed(self) -> int:
        seed = self.cfg.get('run_seed',
                            os.environ.get('EXPERIMENT_GLOBAL_SEED', 0))
        return int(seed)

    @staticmethod
    def _capture_rng_state() -> Dict[str, Any]:
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            # One torchrun process owns exactly one current CUDA device; only
            # its generator is serialized.
            device = torch.cuda.current_device()
            state['torch_cuda'] = torch.cuda.get_rng_state(device)
            state['torch_cuda_device'] = int(device)
        return state

    def _rank_rng_state(
        self,
        state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if state and 'per_rank' in state:
            per_rank = state['per_rank']
            world_size = self.strategy.world_size()
            if (not isinstance(per_rank, list)
                    or len(per_rank) != world_size):
                raise KeyError(
                    'distributed checkpoint RNG state does not contain '
                    f'exactly {world_size} rank entries')
            return per_rank[self.strategy.rank()]
        if state and self.strategy.world_size() > 1:
            raise KeyError(
                'distributed checkpoint is missing per-rank RNG state')
        return state

    def _rank_sampler_state(
        self,
        state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if state and 'per_rank' in state:
            per_rank = state['per_rank']
            world_size = self.strategy.world_size()
            if (not isinstance(per_rank, list)
                    or len(per_rank) != world_size):
                raise KeyError(
                    'distributed checkpoint sampler state does not contain '
                    f'exactly {world_size} rank entries')
            return per_rank[self.strategy.rank()]
        if state and self.strategy.world_size() > 1:
            raise KeyError(
                'distributed checkpoint is missing per-rank sampler state')
        return state

    @staticmethod
    def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        if 'python' in state:
            random.setstate(state['python'])
        if 'numpy' in state:
            np.random.set_state(state['numpy'])
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
        if 'torch_cuda' in state and torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            saved_device = state.get('torch_cuda_device')
            if (saved_device is not None
                    and int(saved_device) != current_device):
                raise RuntimeError(
                    'checkpoint CUDA RNG local-device mapping changed: '
                    f'{saved_device} != {current_device}')
            torch.cuda.set_rng_state(state['torch_cuda'], current_device)

    @staticmethod
    def _convert_batch_dtype(batch: Dict, dtype: torch.dtype) -> Dict:
        """Cast bulk model inputs while preserving robot/time semantics.

        Raw state, action and physical-time values remain FP32 so model-owned
        coordinate conversion is identical to inference.  Latents, text and
        frozen geometry features use the configured compute dtype.
        """
        converted = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    target_dtype = (
                        torch.float32
                        if key in _FP32_SEMANTIC_BATCH_FIELDS else dtype)
                    converted[key] = value.to(dtype=target_dtype)
                else:
                    converted[key] = value
            elif isinstance(value, dict):
                converted[key] = VLATrainer._convert_batch_dtype(value, dtype)
            else:
                converted[key] = value
        return converted

    @staticmethod
    def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device=device, non_blocking=True)
            elif isinstance(value, dict):
                moved[key] = VLATrainer._move_batch_to_device(value, device)
            else:
                moved[key] = value
        return moved
