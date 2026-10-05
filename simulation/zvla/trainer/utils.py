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

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from zvla.core import BaseStrategy, OptimizerGroupSpec, VLAModel

from zvla.optimizers.fsdp2_local_adamw import FSDP2LocalAdamW
from zvla.optimizers.schedulers import (get_constant_schedule,
                                           get_cosine_schedule_with_warmup,
                                           get_cosine_with_min_lr_schedule_with_warmup,
                                           get_step_based_schedule)


def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: dict,
    num_training_steps: int,
    strategy: Optional[BaseStrategy] = None,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Build AdamW optimizer and LR scheduler.

    Eliminates the ~120 lines of duplicated parameter-grouping logic
    from the legacy BaseTrainRunner.

    Args:
        model: The model (already wrapped by Strategy if distributed).
        cfg: Training sub-config dict with keys ``learning_rate``,
            ``weight_decay``, ``scheduler``.
        num_training_steps: Total number of training steps.

    Returns:
        (optimizer, scheduler)
    """
    lr = cfg['learning_rate']
    weight_decay = float(cfg.get('weight_decay') or 0.0)
    if 'backbone_lr_scale' in cfg:
        raise ValueError(
            'backbone_lr_scale was removed; define exact optimizer_group_specs() '
            'on the model instead'
        )
    root = strategy.unwrap_model(model) if strategy is not None else model
    if not isinstance(root, VLAModel):
        raise TypeError(
            f'optimizer construction requires VLAModel, got {type(root).__name__}'
        )
    specs = root.optimizer_group_specs()
    if not isinstance(specs, tuple) or not specs:
        raise TypeError('optimizer_group_specs() must return a non-empty tuple')
    if any(not isinstance(spec, OptimizerGroupSpec) for spec in specs):
        raise TypeError('optimizer_group_specs() returned an invalid specification')

    trainable = {id(parameter): parameter for parameter in root.parameters()
                 if parameter.requires_grad}
    metric_groups = root.training_metric_parameter_groups()
    if not isinstance(metric_groups, dict) and not hasattr(
            metric_groups, 'items'):
        raise TypeError(
            'training_metric_parameter_groups() must return a mapping')
    if not metric_groups:
        raise ValueError(
            'training_metric_parameter_groups() must return a non-empty mapping')
    metric_owner: Dict[int, str] = {}
    for metric_name, parameters in metric_groups.items():
        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError('training metric group names must be non-empty strings')
        for parameter in tuple(parameters):
            identity = id(parameter)
            if identity not in trainable:
                raise ValueError(
                    f'training metric group {metric_name!r} contains a frozen '
                    'or foreign parameter')
            if identity in metric_owner:
                raise ValueError(
                    f'parameter appears in training metric groups '
                    f'{metric_owner[identity]!r} '
                    f'and {metric_name!r}')
            metric_owner[identity] = metric_name
    missing_metric = set(trainable) - set(metric_owner)
    if missing_metric:
        raise ValueError(
            f'{len(missing_metric)} trainable parameters are absent from '
            'training_metric_parameter_groups()')

    assigned: Dict[int, str] = {}
    param_groups = []
    for spec in specs:
        spec_metric_groups = {
            metric_owner[id(parameter)] for parameter in spec.parameters
        }
        if len(spec_metric_groups) != 1:
            raise ValueError(
                f'optimizer group {spec.name!r} crosses semantic metric groups '
                f'{sorted(spec_metric_groups)}; split the optimizer specification '
                'so main-shard telemetry remains exact')
        metric_group = next(iter(spec_metric_groups))
        decay_parameters = []
        no_decay_parameters = []
        for parameter in spec.parameters:
            identity = id(parameter)
            if identity not in trainable:
                raise ValueError(
                    f'optimizer group {spec.name!r} contains a frozen or foreign parameter'
                )
            if identity in assigned:
                raise ValueError(
                    f'parameter appears in optimizer groups {assigned[identity]!r} '
                    f'and {spec.name!r}'
                )
            assigned[identity] = spec.name
            target = no_decay_parameters if parameter.ndim <= 1 else decay_parameters
            target.append(parameter)
        for suffix, parameters, group_weight_decay in (
            ('decay', decay_parameters, weight_decay),
            ('no_decay', no_decay_parameters, 0.0),
        ):
            if parameters:
                param_groups.append({
                    'name': f'{spec.name}/{suffix}',
                    'params': parameters,
                    'lr': lr * spec.lr_scale,
                    'weight_decay': group_weight_decay,
                    'update_after_step': spec.update_after_step,
                    # This survives MCore's BF16-compute -> FP32-main-parameter
                    # replacement, where Python parameter identity no longer
                    # links optimizer gradients back to the application model.
                    'metric_group': metric_group,
                })
    missing = [parameter for identity, parameter in trainable.items()
               if identity not in assigned]
    if missing:
        raise ValueError(
            f'{len(missing)} trainable parameters are absent from optimizer_group_specs()'
        )
    optimizer_cfg = cfg.get('optimizer', {}) or {}
    if optimizer_cfg.get('type', 'adamw').lower() != 'adamw':
        raise ValueError('Zerith VA unified trainer supports AdamW only')
    betas = tuple(float(value) for value in optimizer_cfg.get('betas', (0.9, 0.999)))
    if len(betas) != 2 or not all(0.0 <= value < 1.0 for value in betas):
        raise ValueError('AdamW betas must contain two values in [0,1)')
    fused = bool(trainable) and all(
        parameter.is_cuda for parameter in trainable.values())
    shard_local_fused = bool(optimizer_cfg.get('fsdp2_local_fused', False))
    if shard_local_fused:
        if strategy is None or strategy.__class__.__name__ != 'FSDP2Strategy':
            raise ValueError(
                'optimizer.fsdp2_local_fused requires the FSDP2 strategy')
        if getattr(strategy, 'sharding_strategy', 'no_shard') == 'no_shard':
            raise ValueError(
                'optimizer.fsdp2_local_fused requires sharded parameters')
        if not fused:
            raise ValueError(
                'optimizer.fsdp2_local_fused requires CUDA parameters')
    optimizer_type = FSDP2LocalAdamW if shard_local_fused else torch.optim.AdamW
    optimizer = optimizer_type(
        param_groups, lr=lr, betas=betas, fused=fused
    )
    if strategy is not None:
        optimizer = strategy.setup_optimizer(model, optimizer)

    # --- Scheduler ---
    sched_cfg = cfg.get('scheduler', {}) or {}
    sched_type = sched_cfg.get('type', 'constant')

    if sched_type == 'constant':
        scheduler = get_constant_schedule(optimizer)
        num_warmup_steps = 0
    elif sched_type == 'cosine':
        warmup_ratio = sched_cfg.get('warmup_ratio', 0.0)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        minimum = sched_cfg.get('min_lr_ratio')
        if minimum is None:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps)
        else:
            minimum = float(minimum)
            if not 0.0 <= minimum <= 1.0:
                raise ValueError('scheduler.min_lr_ratio must be in [0,1]')
            scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer,
                num_warmup_steps,
                num_training_steps,
                min_lr_rate=minimum,
            )
        for pg in optimizer.param_groups:
            pg['lr'] = 0.0
    elif sched_type == 'step-based':
        lr_schedule = sched_cfg.get('lr_schedule')
        if lr_schedule is None:
            raise ValueError("lr_schedule must be provided for step-based scheduler")
        scheduler = get_step_based_schedule(optimizer, num_training_steps, lr_schedule)
        num_warmup_steps = 0
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

    scheduler.vla_scheduler_type = sched_type
    scheduler.vla_num_training_steps = int(num_training_steps)
    scheduler.vla_num_warmup_steps = int(num_warmup_steps)
    scheduler.vla_steps_per_epoch = cfg.get('steps_per_epoch')
    scheduler.vla_grad_accum_steps = max(1, int(cfg.get('grad_accumulation_steps', 1)))

    return optimizer, scheduler


def estimate_training_steps(cfg: dict, dataset_len: int, world_size: int = 1) -> int:
    """Estimate total training steps from config and dataset length.

    Args:
        cfg: Training sub-config dict.  Must contain ``per_device_batch_size``
            (injected by the caller from ``data.train.per_device_batch_size``)
            so that the step count is not inflated by a factor of batch_size.
        dataset_len: Number of training examples.
        world_size: Distributed world size.

    Returns:
        Total number of training steps.
    """
    per_device_batch = cfg.get('per_device_batch_size', None)
    if per_device_batch is None:
        import warnings
        warnings.warn(
            'estimate_training_steps: per_device_batch_size not found in cfg; '
            'defaulting to 1, which will over-estimate total_steps by a factor '
            'of the actual batch size and make warmup far too long. '
            'Pass per_device_batch_size explicitly.',
            stacklevel=2,
        )
        per_device_batch = 1
    global_batch = per_device_batch * world_size

    if cfg.get('max_steps') is not None:
        return cfg['max_steps']

    max_epochs = cfg.get('max_epochs')
    if max_epochs is None:
        raise ValueError("Either max_steps or max_epochs must be set")
    scheduler_cfg = cfg.get('scheduler', {}) or {}
    schedule_epochs = scheduler_cfg.get('duration_epochs', max_epochs)
    if int(schedule_epochs) <= 0:
        raise ValueError("scheduler.duration_epochs must be positive")

    grad_accum_steps = max(1, int(cfg.get('grad_accumulation_steps', 1)))
    steps_per_epoch = cfg.get('steps_per_epoch')
    if steps_per_epoch is not None:
        total_forward_steps = int(steps_per_epoch) * int(schedule_epochs)
        return max(1, total_forward_steps // grad_accum_steps)

    # Round dataset length up to global batch size
    n_examples = math.ceil(dataset_len / global_batch) * global_batch
    total_forward_steps = (n_examples * schedule_epochs) // global_batch
    return max(1, total_forward_steps // grad_accum_steps)
