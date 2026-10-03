# SPDX-License-Identifier: Apache-2.0
# Adapted from the ZWAM/ZVLA model source. Inference-only distribution.
# See NOTICE and the repository LICENSE_POLICY.md.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils._pytree import register_dataclass

from .config import FaMWAMConfig
from .kv import DenseKV, LayerKV
from .primitives import (
    ProjectedKV,
    SinusoidalConditioner,
    WanAttention,
    WanFeedForward,
    async_tensor_assert,
    build_rotary_frequencies,
    factorized_grid_positions,
    gather_packed_tokens,
    layer_norm_fp32,
    padding_multiplicity_bias,
    prepare_zero_padded_text,
    select_packed_token_indices,
)


@dataclass(frozen=True)
class ReaderLayerCondition:
    world: LayerKV
    world_valid: torch.Tensor
    text: ProjectedKV
    dynamic_world_tokens: int = 0
    geometry_world_tokens: int = 0
    world_attention_bias: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ReaderConditionCache:
    layers: Tuple[ReaderLayerCondition, ...]
    batch_size: int


@dataclass(frozen=True)
class ReaderLayerConditionSource:
    """Parameter-free inputs materialized by one Reader layer call."""

    world: LayerKV
    world_valid: torch.Tensor
    geometry: Optional[torch.Tensor] = None
    geometry_valid: Optional[torch.Tensor] = None
    geometry_rope: Any = None
    dynamic_world_tokens: int = 0
    world_attention_bias: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ReaderConditionSource:
    """Raw condition interface used to build an inference cache."""

    layers: Tuple[ReaderLayerConditionSource, ...]
    text_emb: torch.Tensor
    batch_size: int


@dataclass(frozen=True)
class ReaderLayerRequest:
    """One operation whose managed parameters belong to a Reader layer."""

    operation: str
    args: Tuple[Any, ...]
    kwargs: Mapping[str, Any]


register_dataclass(ReaderLayerRequest)


class MixedReaderLayer(nn.Module):
    """One Wan-shaped block with world K/V and action K/V in one attention."""

    def __init__(self, dim: int, heads: int, ffn_dim: int, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = WanAttention(dim, heads)
        self.attn2 = WanAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.ffn = WanFeedForward(dim, ffn_dim)
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, request: ReaderLayerRequest) -> Any:
        if not isinstance(request, ReaderLayerRequest):
            raise TypeError("MixedReaderLayer expects ReaderLayerRequest")
        operations = {
            "project_text": self._project_text_impl,
            "project_geometry": self._project_geometry_impl,
            "materialize_condition": self._materialize_condition_impl,
            "action": self._action_impl,
        }
        try:
            operation = operations[request.operation]
        except KeyError as exc:
            raise ValueError(f"unknown Reader layer operation {request.operation!r}") from exc
        return operation(*request.args, **dict(request.kwargs))

    def _request(self, operation: str, *args, **kwargs) -> Any:
        return self(ReaderLayerRequest(operation, tuple(args), dict(kwargs)))

    def project_text(self, *args, **kwargs):
        return self._request("project_text", *args, **kwargs)

    def project_geometry(self, *args, **kwargs):
        return self._request("project_geometry", *args, **kwargs)

    def materialize_condition(self, *args, **kwargs):
        return self._request("materialize_condition", *args, **kwargs)

    def _project_text_impl(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn2.project_kv(text)

    def _project_geometry_impl(
        self, geometry: torch.Tensor, positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn1.project_kv(geometry, positions=positions)

    def _materialize_condition_impl(
        self,
        source: ReaderLayerConditionSource,
        text: torch.Tensor,
        *,
        text_prefix_length: int,
        text_logical_length: int,
    ) -> ReaderLayerCondition:
        key = source.world.key[:, 0]
        value = source.world.value[:, 0]
        condition_valid = source.world_valid
        geometry_count = 0
        if source.geometry is not None:
            (geometry_key, geometry_value) = self.attn1.project_kv(
                source.geometry.to(dtype=key.dtype), positions=source.geometry_rope
            )
            geometry_count = geometry_key.shape[2]
            key = torch.cat((key, geometry_key), dim=2)
            value = torch.cat((value, geometry_value), dim=2)
            condition_valid = torch.cat((condition_valid, source.geometry_valid), dim=1)
        (text_key, text_value) = self.attn2.project_kv(text)
        text_bias = padding_multiplicity_bias(
            text_key, prefix_length=text_prefix_length, logical_length=text_logical_length
        )
        return ReaderLayerCondition(
            LayerKV(key[:, None], value[:, None]),
            condition_valid,
            ProjectedKV(text_key, text_value, None, text_bias),
            source.dynamic_world_tokens,
            geometry_count,
            source.world_attention_bias,
        )

    def _action_impl(
        self,
        hidden: torch.Tensor,
        action_valid: torch.Tensor,
        condition: ReaderLayerCondition,
        action_rope,
        time_projection: torch.Tensor,
    ) -> torch.Tensor:
        modulation = (
            self.scale_shift_table.unsqueeze(0)
            + time_projection.unflatten(-1, (6, hidden.shape[-1]))[:, None]
        )
        (shift, scale, gate, ff_shift, ff_scale, ff_gate) = modulation.unbind(dim=-2)
        value = layer_norm_fp32(self.norm1, hidden.float())
        value = (value * (1.0 + scale) + shift).to(hidden.dtype)
        action_query = self.attn1.project_q(value, positions=action_rope)
        (action_key, action_value) = self.attn1.project_kv(value, positions=action_rope)
        world_key = condition.world.key[:, 0]
        world_value = condition.world.value[:, 0]
        key = torch.cat((world_key, action_key), dim=2)
        projected_value = torch.cat((world_value, action_value), dim=2)
        key_valid = torch.cat((condition.world_valid, action_valid), dim=1)
        attention_bias = None
        if condition.world_attention_bias is not None:
            action_bias = condition.world_attention_bias.new_zeros(
                *condition.world_attention_bias.shape[:-1], action_key.shape[2]
            )
            attention_bias = torch.cat((condition.world_attention_bias, action_bias), dim=-1)
        attended = self.attn1.attend_preprojected(
            action_query, key, projected_value, key_valid=key_valid, attention_bias=attention_bias
        )
        hidden = (hidden.float() + gate * attended.float()).to(hidden.dtype)
        cross = layer_norm_fp32(self.norm2, hidden)
        hidden = hidden + self.attn2.attend_projected(
            cross,
            condition.text.key,
            condition.text.value,
            key_valid=condition.text.valid,
            attention_bias=condition.text.attention_bias,
        )
        value = layer_norm_fp32(self.norm3, hidden.float())
        value = (value * (1.0 + ff_scale) + ff_shift).to(hidden.dtype)
        feed_forward = self.ffn(value)
        hidden = (hidden.float() + ff_gate * feed_forward.float()).to(hidden.dtype)
        hidden = hidden * action_valid.unsqueeze(-1).to(hidden.dtype)
        return hidden


class PolicyReader(nn.Module):
    """Recursive Reader blocks over a fixed-size world K/V interface."""

    def __init__(self, config: FaMWAMConfig) -> None:
        super().__init__()
        self.config = config
        self.action_embed = nn.Linear(config.action_dim, config.reader_dim)
        self.feature_mask_embed = nn.Linear(config.action_dim, config.reader_dim, bias=False)
        self.primitive_dt_embedding = nn.Sequential(
            nn.Linear(2, config.reader_dim),
            nn.SiLU(),
            nn.Linear(config.reader_dim, config.reader_dim),
        )
        self.state_embedding = (
            nn.Sequential(
                nn.Linear(config.state_dim * 2, config.reader_dim),
                nn.SiLU(),
                nn.Linear(config.reader_dim, config.reader_dim),
            )
            if config.use_proprioception
            else None
        )
        self.condition_embedder = SinusoidalConditioner(config.reader_dim, config.text_dim)
        self.layers = nn.ModuleList(
            (
                MixedReaderLayer(
                    config.reader_dim, config.reader_heads, config.reader_ffn_dim, config.eps
                )
                for _ in range(config.reader_layers)
            )
        )
        self.output_norm = nn.LayerNorm(config.reader_dim)
        self.output_projection = nn.Linear(config.reader_dim, config.action_dim)
        if config.native7_minmax:
            self.native7_action_embed = nn.Linear(7, config.reader_dim)
            self.native7_mask_embed = nn.Linear(7, config.reader_dim, bias=False)
            self.native7_output_projection = nn.Linear(config.reader_dim, 7)

    def prepare_condition_source(
        self,
        world_kv: DenseKV,
        text_emb: torch.Tensor,
        text_valid: Optional[torch.Tensor] = None,
        *,
        external_camera_weight: int = 1,
        geometry_tokens: Optional[torch.Tensor] = None,
        geometry_valid: Optional[torch.Tensor] = None,
    ) -> ReaderConditionSource:
        """Pack raw world/text/geometry inputs without reading parameters."""
        if len(world_kv.layers) != self.config.kv_layers:
            raise ValueError("Reader world K/V count must match the configured interface")
        (batch, times, spatial) = world_kv.valid.shape
        if times != 4 or spatial != self.config.spatial_tokens:
            raise ValueError("Reader world K/V must be current+h1+h2+h4 on canonical grid")
        if text_emb.shape[0] != batch:
            raise ValueError("text and world batch sizes differ")
        if text_emb.shape[1] not in (self.config.text_prefix_length, self.config.text_length):
            raise ValueError("Reader text must use compact or logical Wan context")
        if text_valid is not None:
            raise ValueError("Reader preserves official unmasked Wan text semantics")
        if external_camera_weight not in (1, 2):
            raise ValueError("external_camera_weight must be 1 or 2")
        world_valid = world_kv.valid.reshape(batch, times * spatial)
        (world_indices, packed_world_valid) = select_packed_token_indices(
            world_valid, times * self.config.active_spatial_tokens
        )
        packed_geometry = packed_geometry_valid = geometry_rope = None
        if geometry_tokens is not None:
            expected_geometry = (batch, spatial, self.config.reader_dim)
            if tuple(geometry_tokens.shape) != expected_geometry:
                raise ValueError(f"geometry tokens must have shape {expected_geometry}")
            if (
                geometry_valid is None
                or tuple(geometry_valid.shape) != (batch, spatial)
                or geometry_valid.dtype != torch.bool
            ):
                raise ValueError("geometry_valid must be boolean (B,147)")
            (geometry_indices, packed_geometry_valid) = select_packed_token_indices(
                geometry_valid, self.config.active_spatial_tokens
            )
            packed_geometry = gather_packed_tokens(geometry_tokens, geometry_indices)
            rows = self.config.canonical_latent_height // self.config.patch_size
            cols = self.config.canonical_latent_width // self.config.patch_size
            geometry_positions = factorized_grid_positions(
                batch,
                1,
                rows,
                cols,
                geometry_tokens.device,
                row_bases=self.config.camera_rope_bases,
            )
            geometry_positions = gather_packed_tokens(geometry_positions, geometry_indices)
            geometry_rope = build_rotary_frequencies(geometry_positions, self.config.head_dim)
        elif geometry_valid is not None:
            raise ValueError("geometry tokens and validity must be provided together")
        external_indices = external_valid = None
        if external_camera_weight == 2:
            tokens_per_camera = spatial // self.config.camera_slots
            spatial_index = torch.arange(spatial, device=world_valid.device)
            camera_mask = (spatial_index < tokens_per_camera)[None, None, :]
            external = (world_kv.valid & camera_mask).reshape(batch, times * spatial)
            (external_indices, external_valid) = select_packed_token_indices(
                external, times * tokens_per_camera
            )

        def pack(source: LayerKV, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            dense_key = (
                source.key.transpose(1, 2)
                .reshape(batch, self.config.reader_heads, times * spatial, self.config.head_dim)
                .transpose(1, 2)
            )
            dense_value = (
                source.value.transpose(1, 2)
                .reshape(batch, self.config.reader_heads, times * spatial, self.config.head_dim)
                .transpose(1, 2)
            )
            return (
                gather_packed_tokens(dense_key, indices).transpose(1, 2),
                gather_packed_tokens(dense_value, indices).transpose(1, 2),
            )

        sources = []
        for index in range(len(self.layers)):
            source_index = self.config.reader_kv_indices[index]
            (world_key, world_value) = pack(world_kv.layers[source_index], world_indices)
            keys = [world_key]
            values = [world_value]
            validity = [packed_world_valid]
            if external_indices is not None:
                (external_key, external_value) = pack(
                    world_kv.layers[source_index], external_indices
                )
                keys.append(external_key)
                values.append(external_value)
                validity.append(external_valid)
            key = torch.cat(keys, dim=2)
            value = torch.cat(values, dim=2)
            condition_valid = torch.cat(validity, dim=1)
            dynamic_tokens = key.shape[2]
            sources.append(
                ReaderLayerConditionSource(
                    LayerKV(key[:, None], value[:, None]),
                    condition_valid,
                    packed_geometry,
                    packed_geometry_valid,
                    geometry_rope,
                    dynamic_tokens,
                    None,
                )
            )
        return ReaderConditionSource(tuple(sources), text_emb, batch)

    def build_condition_cache(
        self,
        world_kv: DenseKV,
        text_emb: torch.Tensor,
        text_valid: Optional[torch.Tensor] = None,
        *,
        external_camera_weight: int = 1,
        geometry_tokens: Optional[torch.Tensor] = None,
        geometry_valid: Optional[torch.Tensor] = None,
    ) -> ReaderConditionCache:
        """Materialize a reusable inference cache through layer calls."""
        source = self.prepare_condition_source(
            world_kv,
            text_emb,
            text_valid,
            external_camera_weight=external_camera_weight,
            geometry_tokens=geometry_tokens,
            geometry_valid=geometry_valid,
        )
        compact_text = prepare_zero_padded_text(
            source.text_emb,
            prefix_length=self.config.text_prefix_length,
            logical_length=self.config.text_length,
        )
        text = self.condition_embedder.text(compact_text, source.layers[0].world.key.dtype)
        conditions = tuple(
            (
                layer.materialize_condition(
                    layer_source,
                    text,
                    text_prefix_length=self.config.text_prefix_length,
                    text_logical_length=self.config.text_length,
                )
                for (layer, layer_source) in zip(self.layers, source.layers, strict=False)
            )
        )
        return ReaderConditionCache(conditions, source.batch_size)

    def forward(
        self,
        noisy_action: torch.Tensor,
        action_feature_valid: torch.Tensor,
        sigma: torch.Tensor,
        primitive_dt: torch.Tensor,
        condition_cache: ReaderConditionCache,
        current_state: Optional[torch.Tensor] = None,
        current_state_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(condition_cache, ReaderConditionCache):
            raise TypeError("Reader condition must be ReaderConditionCache")
        if tuple(action_feature_valid.shape) != tuple(noisy_action.shape):
            raise ValueError("action feature mask must exactly match noisy_action")
        expected = (
            condition_cache.batch_size,
            self.config.action_prediction_steps,
            self.config.action_dim,
        )
        if tuple(noisy_action.shape) != expected:
            raise ValueError(f"noisy_action must have shape {expected}")
        (batch, primitives, _) = noisy_action.shape
        if tuple(sigma.shape) != (batch,):
            raise ValueError("sigma must be (B,) in [0,1]")
        async_tensor_assert(((sigma >= 0) & (sigma <= 1)).all(), "sigma must be in [0,1]")
        if tuple(primitive_dt.shape) != (batch,):
            raise ValueError("primitive_dt must be positive (B,)")
        async_tensor_assert((primitive_dt > 0).all(), "primitive_dt must be positive")
        feature_valid = action_feature_valid.bool()
        action_valid = feature_valid.any(dim=-1)
        action = noisy_action * feature_valid.to(noisy_action.dtype)
        hidden = self.action_embed(action) + self.feature_mask_embed(feature_valid.to(action.dtype))
        if self.config.native7_minmax:
            from zwam_core.actions.mixed_native7 import native7_rows

            native = native7_rows(feature_valid)
            native_hidden = self.native7_action_embed(action[..., :7]) + self.native7_mask_embed(
                feature_valid[..., :7].to(action.dtype)
            )
            hidden = torch.where(native, native_hidden, hidden)
        ratio = primitive_dt.float() / self.config.primitive_dt_reference
        dt = torch.stack((ratio, ratio.log()), dim=-1)
        hidden = (
            hidden
            + self.primitive_dt_embedding(dt.to(self.primitive_dt_embedding[0].weight.dtype)).to(
                hidden.dtype
            )[:, None]
        )
        if self.state_embedding is not None:
            expected_state = (batch, self.config.state_dim)
            if current_state is None or current_state_valid is None:
                raise ValueError("proprioceptive Reader requires current state and mask")
            if (
                tuple(current_state.shape) != expected_state
                or tuple(current_state_valid.shape) != expected_state
            ):
                raise ValueError(f"current state and mask must have shape {expected_state}")
            state_valid = current_state_valid.bool()
            state = current_state * state_valid.to(current_state.dtype)
            state_input = torch.cat((state, state_valid.to(current_state.dtype)), dim=-1)
            state_hidden = self.state_embedding(
                state_input.to(self.state_embedding[0].weight.dtype)
            )
            hidden = hidden + state_hidden.to(hidden.dtype)[:, None]
        hidden = hidden * action_valid.unsqueeze(-1).to(hidden.dtype)
        (_, time_projection) = self.condition_embedder.time(
            sigma.float() * self.config.flow_timestep_scale, hidden.dtype
        )
        position = torch.zeros(batch, primitives, 3, device=hidden.device, dtype=torch.float32)
        position[..., 0] = (
            torch.arange(primitives, device=hidden.device).float()[None]
            * primitive_dt.float()[:, None]
            / self.config.video_dt_reference
        )
        action_rope = build_rotary_frequencies(position, self.config.head_dim)
        for layer, condition in zip(self.layers, condition_cache.layers, strict=True):
            hidden = layer(
                ReaderLayerRequest(
                    "action", (hidden, action_valid, condition, action_rope, time_projection), {}
                )
            )
        velocity = self.output_projection(self.output_norm(hidden))
        if self.config.native7_minmax:
            native_velocity = self.native7_output_projection(self.output_norm(hidden))
            native_velocity = torch.nn.functional.pad(
                native_velocity, (0, self.config.action_dim - 7)
            )
            velocity = torch.where(native, native_velocity, velocity)
        velocity = velocity * feature_valid.to(velocity.dtype)
        return velocity


@torch.no_grad()
def integrate_action_flow(
    reader: PolicyReader,
    action_feature_valid: torch.Tensor,
    primitive_dt: torch.Tensor,
    condition_cache: ReaderConditionCache,
    noise: torch.Tensor,
    steps: int,
    current_state: Optional[torch.Tensor] = None,
    current_state_valid: Optional[torch.Tensor] = None,
    *,
    fp32_accumulation: bool = False,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("action flow steps must be positive")
    if fp32_accumulation:
        noise = noise.float()
    action = noise * action_feature_valid.to(noise.dtype)
    schedule = torch.linspace(1.0, 0.0, steps + 1, device=action.device, dtype=torch.float32)
    for index in range(steps):
        sigma = schedule[index].expand(action.shape[0])
        velocity = reader(
            action,
            action_feature_valid,
            sigma,
            primitive_dt,
            condition_cache,
            current_state,
            current_state_valid,
        )
        if fp32_accumulation:
            velocity = velocity.float()
        action = action + (schedule[index + 1] - schedule[index]) * velocity
        action = action * action_feature_valid.to(action.dtype)
    return action
