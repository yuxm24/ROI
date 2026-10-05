"""Geometry-token interfaces for the VGGT + Wan multi-expert MoT."""

from __future__ import annotations

import math
from typing import MutableMapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ZerithVAConfig
from .kv import DenseKV, LayerKV
from .primitives import (
    gather_packed_tokens,
    scatter_packed_tokens,
    select_packed_token_indices,
)


class GeometryTokenBridge(nn.Module):
    """Map frozen VGGT patch features into Reader-width geometry tokens."""

    def __init__(self, config: ZerithVAConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.geometry_feature_dim)
        self.projection = nn.Linear(config.geometry_feature_dim,
                                    config.reader_dim)
        self.output_norm = nn.LayerNorm(config.reader_dim)
        self.camera_embedding = nn.Parameter(
            torch.zeros(config.camera_slots, config.reader_dim))

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            features.shape[0],
            self.config.spatial_tokens,
            self.config.geometry_feature_dim,
        )
        if tuple(features.shape) != expected:
            raise ValueError(f"geometry features must have shape {expected}")
        if tuple(valid.shape) != expected[:2] or valid.dtype != torch.bool:
            raise ValueError("geometry_valid must be boolean (B,147)")
        value = self.input_norm(
            features.to(dtype=self.input_norm.weight.dtype))
        value = self.projection(value)
        camera_ids = torch.arange(
            self.config.camera_slots,
            device=value.device,
        ).repeat_interleave(self.config.geometry_tokens_per_camera)
        value = value + self.camera_embedding[camera_ids][None]
        value = self.output_norm(value)
        return value * valid.unsqueeze(-1).to(value.dtype)


class _GeometryTransitionBindingLayer(nn.Module):
    """One shared low-rank geometry/transition factorization."""

    def __init__(self, width: int, rank: int, fusion: str) -> None:
        super().__init__()
        self.geometry = nn.Linear(width, rank, bias=False)
        self.transition = nn.Linear(width, rank, bias=False)
        self.output = nn.Linear(rank, width, bias=False)
        self.fusion = fusion
        # One no-decay scalar for h1, h2 and h4.  Keeping the parameter
        # one-dimensional also makes the optimizer's no-decay rule explicit.
        self.gates = nn.Parameter(torch.zeros(3))
        nn.init.xavier_uniform_(self.geometry.weight)
        nn.init.xavier_uniform_(self.transition.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(
        self,
        geometry: torch.Tensor,
        transitions: torch.Tensor,
    ) -> torch.Tensor:
        geometry_factor = self.geometry(geometry)[:, None]
        if self.fusion == "geometry_gated":
            geometry_factor = F.silu(geometry_factor)
        elif self.fusion != "linear_product":
            raise RuntimeError(f"unsupported G4D fusion: {self.fusion}")
        transition_factor = self.transition(transitions)
        return self.output(geometry_factor * transition_factor)


class GeometryAnchored4DBinder(nn.Module):
    """Bind current geometry to adjacent future Wan Value transitions.

    Keys and the current Value are immutable.  Each future Value receives a
    zero-gated, low-rank residual derived from the geometry token at the same
    canonical patch and the preceding temporal Value difference.
    """

    def __init__(self, config: ZerithVAConfig) -> None:
        super().__init__()
        self.config = config
        self.residual_scale = 1.0
        self.layers = nn.ModuleList(
            _GeometryTransitionBindingLayer(
                config.reader_dim,
                config.g4d_binding_rank,
                config.g4d_binding_fusion,
            )
            for _ in range(config.kv_layers))

    def set_residual_scale(self, scale: float) -> None:
        scale = float(scale)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("G4D residual scale must be finite and non-negative")
        self.residual_scale = scale

    @property
    def gate_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(layer.gates for layer in self.layers)

    @property
    def projection_parameters(self) -> tuple[nn.Parameter, ...]:
        gate_ids = {id(parameter) for parameter in self.gate_parameters}
        return tuple(parameter for parameter in self.parameters()
                     if id(parameter) not in gate_ids)

    @staticmethod
    def _normalize(value: torch.Tensor, eps: float) -> torch.Tensor:
        # Normalize in fp32, then let the local projection dtype determine the
        # matmul precision under FSDP/autocast.
        return F.layer_norm(
            value.float(),
            (value.shape[-1], ),
            weight=None,
            bias=None,
            eps=eps,
        )

    @staticmethod
    def _masked_rms(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weight = valid[..., None].float()
        count = weight.sum() * value.shape[-1]
        return torch.sqrt((value.detach().float().square() * weight).sum() /
                          count.clamp_min(1.0))

    def forward(
        self,
        world: DenseKV,
        geometry_tokens: torch.Tensor,
        geometry_valid: torch.Tensor,
        *,
        metrics_out: Optional[MutableMapping[str, torch.Tensor]] = None,
    ) -> DenseKV:
        if len(world.layers) != len(self.layers):
            raise ValueError(
                "G4D-Bind layer count must match the dense K/V interface")
        batch, times, spatial = world.valid.shape
        if times != 4:
            raise ValueError("G4D-Bind requires current+h1+h2+h4")
        expected_geometry = (batch, spatial, self.config.reader_dim)
        if tuple(geometry_tokens.shape) != expected_geometry:
            raise ValueError(
                f"geometry tokens must have shape {expected_geometry}")
        if (tuple(geometry_valid.shape) != (batch, spatial)
                or geometry_valid.dtype != torch.bool):
            raise ValueError("geometry_valid must be boolean (B,S)")

        geometry_valid = geometry_valid & world.valid[:, 0]
        packed_indices, _ = select_packed_token_indices(
            geometry_valid, self.config.active_spatial_tokens)
        packed_geometry = self._normalize(
            gather_packed_tokens(geometry_tokens, packed_indices),
            self.config.eps,
        )
        corrected_layers = []
        all_residual_rms = []
        all_value_rms = []
        all_gate_values = []

        for source, binder in zip(world.layers, self.layers, strict=True):
            value = source.value.permute(0, 1, 3, 2,
                                         4).reshape(batch, times, spatial,
                                                    self.config.reader_dim)
            packed_value = gather_packed_tokens(
                value.transpose(1, 2), packed_indices).transpose(1, 2)
            raw_transitions = packed_value[:, 1:] - packed_value[:, :-1]
            dense_valid = (
                geometry_valid[:, None] & world.valid[:, :-1]
                & world.valid[:, 1:])
            packed_valid = gather_packed_tokens(
                dense_valid.transpose(1, 2), packed_indices).transpose(1, 2)
            projection_dtype = binder.geometry.weight.dtype
            bound = binder(
                packed_geometry.to(dtype=projection_dtype),
                self._normalize(raw_transitions,
                                self.config.eps).to(dtype=projection_dtype),
            )
            bound = bound * packed_valid[..., None].to(bound.dtype)
            # Training keeps FP32 optimizer masters while deployment stores
            # the Binder in BF16.  Round the gate before tanh so both paths
            # apply the same deployed gate value.
            gates = torch.tanh(binder.gates.to(bound.dtype))
            packed_residual = (
                gates[None, :, None, None] * bound)
            if self.residual_scale != 1.0:
                packed_residual = packed_residual * self.residual_scale
            dense_residual = scatter_packed_tokens(
                packed_residual.transpose(1, 2),
                packed_indices,
                spatial,
            ).transpose(1, 2)
            corrected_futures = (value[:, 1:].float() +
                                 dense_residual.float()).to(value.dtype)
            dense = torch.cat((value[:, :1], corrected_futures), dim=1)

            if metrics_out is not None:
                for horizon_index in range(len(self.config.horizons)):
                    valid = packed_valid[:, horizon_index]
                    residual = packed_residual[:, horizon_index]
                    residual_rms = self._masked_rms(residual, valid)
                    value_rms = self._masked_rms(
                        packed_value[:, horizon_index + 1], valid)
                    gate = gates[horizon_index]
                    all_residual_rms.append(residual_rms)
                    all_value_rms.append(value_rms)
                    all_gate_values.append(gate.detach())

            corrected_value = dense.reshape(
                batch,
                times,
                spatial,
                self.config.reader_heads,
                self.config.head_dim,
            ).permute(0, 1, 3, 2, 4)
            corrected_layers.append(LayerKV(source.key, corrected_value))

        if metrics_out is not None:
            residual_rms = torch.stack(all_residual_rms)
            value_rms = torch.stack(all_value_rms)
            gates = torch.stack(all_gate_values)
            metrics_out["g4d_bound_residual_rms"] = residual_rms.mean()
            metrics_out["g4d_original_future_value_rms"] = value_rms.mean()
            metrics_out["g4d_residual_value_ratio"] = (
                residual_rms / value_rms.clamp_min(self.config.eps)).mean()
            metrics_out["g4d_residual_value_ratio_max"] = (
                residual_rms / value_rms.clamp_min(self.config.eps)).max()
            metrics_out["g4d_gate_abs_mean"] = gates.abs().mean()
            metrics_out["g4d_gate_saturation_fraction"] = (
                gates.abs() > 0.95).float().mean()

        return DenseKV(tuple(corrected_layers), world.valid)
