"""Fused AdamW over FSDP2 DTensor local shards.

PyTorch 2.9's AdamW accepts DTensor parameters, but dispatching one
multi-tensor optimizer operation through DTensor is disproportionately
expensive for the 5B-parameter FSDP2 model. FSDP2 already exposes the exact
local parameter, gradient, and optimizer-state shards. Applying the same
``torch._fused_adamw_`` operation to those aligned local tensors is
mathematically identical and preserves the ordinary AdamW state-dict ABI.
"""

from __future__ import annotations

from typing import Optional

import torch

if not hasattr(torch, "_fused_adamw_"):
    raise RuntimeError(
        "FSDP2LocalAdamW requires the fused AdamW kernel exposed by PyTorch")


def _local_tensor(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor, got {type(value)}")
    local = value.to_local() if hasattr(value, "to_local") else value
    if local.device.type != "cuda":
        raise RuntimeError(
            f"FSDP2LocalAdamW requires CUDA-local {label}, got {local.device}")
    return local


class FSDP2LocalAdamW(torch.optim.AdamW):
    """AdamW with the standard state schema and shard-local fused execution."""

    implementation_version = "fsdp2-local-fused-adamw-v1"

    def __init__(self, params, *args, **kwargs) -> None:
        if kwargs.get("differentiable", False):
            raise ValueError(
                "FSDP2LocalAdamW does not support differentiable=True")
        # ``_init_group`` uses this flag to put scalar steps on CUDA and to
        # create the same state layout as stock fused AdamW.
        kwargs["fused"] = True
        kwargs.pop("foreach", None)
        super().__init__(params, *args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_scale = getattr(self, "grad_scale", None)
        found_inf = getattr(self, "found_inf", None)
        if grad_scale is not None or found_inf is not None:
            raise RuntimeError(
                "FSDP2LocalAdamW is for BF16 training without GradScaler")

        for group_index, group in enumerate(self.param_groups):
            if group.get("differentiable", False):
                raise RuntimeError(
                    "FSDP2LocalAdamW does not support differentiable groups")
            params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            has_complex = self._init_group(
                group,
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            if has_complex:
                raise RuntimeError(
                    "FSDP2LocalAdamW does not support complex parameters")
            if not params:
                continue

            local_params = [
                _local_tensor(value, label="parameter") for value in params
            ]
            local_grads = [
                _local_tensor(value, label="gradient") for value in grads
            ]
            local_exp_avgs = [
                _local_tensor(value, label="first moment")
                for value in exp_avgs
            ]
            local_exp_avg_sqs = [
                _local_tensor(value, label="second moment")
                for value in exp_avg_sqs
            ]
            local_max_exp_avg_sqs = [
                _local_tensor(value, label="maximum second moment")
                for value in max_exp_avg_sqs
            ]
            local_steps = [
                _local_tensor(value, label="optimizer step")
                for value in state_steps
            ]

            device_dtypes = {(value.device, value.dtype)
                             for value in local_params}
            if len(device_dtypes) != 1:
                raise RuntimeError(
                    "each FSDP2LocalAdamW parameter group must have one "
                    f"CUDA device/dtype, group {group_index} has "
                    f"{sorted(map(str, device_dtypes))}")
            aligned = zip(
                local_params,
                local_grads,
                local_exp_avgs,
                local_exp_avg_sqs,
            )
            for tensor_index, tensors in enumerate(aligned):
                shapes = {tuple(value.shape) for value in tensors}
                if len(shapes) != 1:
                    raise RuntimeError(
                        "FSDP2LocalAdamW local shard shapes differ in group "
                        f"{group_index}, tensor {tensor_index}: {shapes}")

            torch._foreach_add_(local_steps, 1)
            torch._fused_adamw_(
                local_params,
                local_grads,
                local_exp_avgs,
                local_exp_avg_sqs,
                local_max_exp_avg_sqs,
                local_steps,
                amsgrad=bool(group["amsgrad"]),
                lr=group["lr"],
                beta1=group["betas"][0],
                beta2=group["betas"][1],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=bool(group["maximize"]),
                grad_scale=None,
                found_inf=None,
            )
        return loss
