"""Checkpoint I/O helpers."""

from pathlib import Path

import torch


def torch_load_checkpoint(path, map_location="cpu", *, mmap=False):
    """Load training checkpoints that may contain Python/NumPy RNG states."""
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file

        device = str(map_location) if map_location is not None else "cpu"
        return {"model": load_file(str(path), device=device)}
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
            mmap=mmap,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def scheduler_resume_mismatch(checkpoint, scheduler):
    """Return why an exact resume would change the LR schedule, if any."""
    if scheduler is None or 'scheduler_state_dict' not in checkpoint:
        return None
    current_total = getattr(scheduler, 'vla_num_training_steps', None)
    if current_total is None:
        return None

    meta = checkpoint.get('scheduler_meta')
    if not meta:
        return (
            'Checkpoint has no scheduler_meta. If it was created before the '
            'grad_accumulation-aware total-step fix, loading its scheduler '
            'state may cause an LR discontinuity.'
        )
    saved_total = meta.get('num_training_steps')
    saved_accum = meta.get('grad_accum_steps')
    current_accum = getattr(scheduler, 'vla_grad_accum_steps', None)
    if saved_total is not None and int(saved_total) != int(current_total):
        return (
            'Checkpoint scheduler total steps differ from this run '
            f'({saved_total} != {current_total}); LR may jump after resume.'
        )
    if (saved_accum is not None and current_accum is not None
            and int(saved_accum) != int(current_accum)):
        return (
            'Checkpoint grad_accum_steps differs from this run '
            f'({saved_accum} != {current_accum}); LR schedule may not '
            'match the original run.'
        )
    return None


def assert_scheduler_resume_compatible(checkpoint, scheduler):
    """Fail closed when an exact resume would use a different schedule."""
    message = scheduler_resume_mismatch(checkpoint, scheduler)
    if message:
        raise ValueError(
            f'{message} Use the explicit fork workflow for a new schedule.')


def warn_scheduler_resume_mismatch(checkpoint, scheduler, overwatch=None):
    """Warn for legacy backends that still permit schedule-changing resume."""
    message = scheduler_resume_mismatch(checkpoint, scheduler)

    if message:
        if overwatch is not None:
            overwatch.warning(message)
        else:
            import warnings
            warnings.warn(message, stacklevel=2)
