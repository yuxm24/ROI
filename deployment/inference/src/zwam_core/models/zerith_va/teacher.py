# SPDX-License-Identifier: Apache-2.0
# Inference-only adaptation; see NOTICE.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ZerithVAConfig
from .kv import DenseKV, LayerKV
from .primitives import (
    SinusoidalConditioner,
    WanCompatibleBlock,
    build_rotary_frequencies,
    causal_temporal_frame_mask,
    factorized_grid_positions,
    gather_packed_tokens,
    prepare_zero_padded_text,
    scatter_packed_tokens,
    select_packed_token_indices,
)


@dataclass(frozen=True)
class WanWorldRequest:
    """One hook-visible operation issued to :class:`EvolvingWanWorld`."""

    operation: str
    args: Tuple[Any, ...]
    kwargs: Mapping[str, Any]


class EvolvingWanWorld(nn.Module):
    """Wan encoder with checkpoint-compatible projections and camera embeddings."""

    def __init__(self, config: ZerithVAConfig, *, allow_uninitialized: bool = False) -> None:
        super().__init__()
        self.config = config
        self.allow_uninitialized = bool(allow_uninitialized)
        patch_area = config.patch_size * config.patch_size
        self.patch_embedding_mlp = nn.Linear(
            config.latent_channels * patch_area, config.student_dim
        )
        self.condition_embedder = SinusoidalConditioner(config.student_dim, config.text_dim)
        self.blocks = nn.ModuleList(
            (
                WanCompatibleBlock(
                    config.student_dim, config.student_heads, config.student_ffn_dim, config.eps
                )
                for _ in range(config.teacher_layers)
            )
        )
        self.output_norm = nn.LayerNorm(
            config.student_dim, eps=config.eps, elementwise_affine=False
        )
        self.output_scale_shift = nn.Parameter(
            torch.randn(1, 2, config.student_dim) / config.student_dim**0.5
        )
        self.output_projection = nn.Linear(config.student_dim, config.latent_channels * patch_area)
        self.camera_embedding = nn.Parameter(torch.zeros(config.camera_slots, config.student_dim))
        self._official_weights_loaded = False

    def forward(self, request: WanWorldRequest) -> Any:
        """Execute exactly one managed Wan operation inside module hooks."""
        if not isinstance(request, WanWorldRequest):
            raise TypeError(
                "EvolvingWanWorld expects a WanWorldRequest; direct managed parameter-method calls are not supported"
            )
        operations = {"extract_teacher_carrier_kv": self._extract_teacher_carrier_kv_impl}
        try:
            operation = operations[request.operation]
        except KeyError as exc:
            raise ValueError(f"unknown Wan world operation {request.operation!r}") from exc
        return operation(*request.args, **dict(request.kwargs))

    def _request(self, operation: str, *args, **kwargs) -> Any:
        return self(WanWorldRequest(operation, tuple(args), dict(kwargs)))

    def extract_teacher_carrier_kv(self, *args, **kwargs):
        return self._request("extract_teacher_carrier_kv", *args, **kwargs)

    @property
    def official_weights_loaded(self) -> bool:
        return self._official_weights_loaded

    def mark_official_weights_loaded(self) -> None:
        self._official_weights_loaded = True

    def _require_weights(self) -> None:
        if not self._official_weights_loaded and (not self.allow_uninitialized):
            raise RuntimeError("EvolvingWanWorld has no verified Wan2.2-TI2V-5B base weights")

    def _camera_tokens(self, rows: int, cols: int) -> torch.Tensor:
        if rows % self.camera_embedding.shape[0]:
            raise ValueError("patch rows must divide evenly across camera slots")
        rows_per_camera = rows // self.camera_embedding.shape[0]
        camera_ids = torch.arange(
            self.camera_embedding.shape[0], device=self.camera_embedding.device
        ).repeat_interleave(rows_per_camera * cols)
        return F.embedding(camera_ids, self.camera_embedding)[None]

    def _patchify(
        self, latent: torch.Tensor, spatial_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        if latent.ndim != 5:
            raise ValueError("world latent must have shape (B,F,C,H,W)")
        (batch, frames, channels, height, width) = latent.shape
        if tuple(spatial_valid.shape) != (batch, frames, height, width):
            raise ValueError("world spatial_valid must have shape (B,F,H,W)")
        if channels != self.config.latent_channels:
            raise ValueError("world latent channel width differs from config")
        patch = self.config.patch_size
        if height % patch or width % patch:
            raise ValueError("world latent size must divide evenly by patch_size")
        flat = latent.reshape(batch * frames, channels, height, width)
        patches = F.unfold(flat, kernel_size=patch, stride=patch).transpose(1, 2)
        valid = (
            F.max_pool2d(
                spatial_valid.reshape(batch * frames, 1, height, width).float(),
                kernel_size=patch,
                stride=patch,
            )
            .flatten(1)
            .bool()
            .reshape(batch, -1)
        )
        (rows, cols) = (height // patch, width // patch)
        return (patches.reshape(batch, -1, patches.shape[-1]), valid, rows, cols)

    def _encode(
        self,
        latent: torch.Tensor,
        spatial_valid: torch.Tensor,
        text_emb: torch.Tensor,
        timestep: torch.Tensor,
        *,
        capture_kv: bool,
        time_positions: Optional[torch.Tensor] = None,
        text_repeats: int = 1,
        temporal_frame_mask: Optional[torch.Tensor] = None,
        stop_after_last_capture: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[LayerKV, ...], torch.Tensor, Tuple[int, int], torch.Tensor]:
        self._require_weights()
        (batch, frames) = latent.shape[:2]
        if text_repeats <= 0 or text_emb.shape[0] * text_repeats != batch:
            raise ValueError("world text batch times repeats must match latent batch")
        if text_emb.shape[1] not in (self.config.text_prefix_length, self.config.text_length):
            raise ValueError("world text must use compact or logical Wan context")
        if tuple(timestep.shape) != (batch,):
            raise ValueError("world timestep must have shape (B,)")
        (patches, token_valid, rows, cols) = self._patchify(latent, spatial_valid)
        dense_token_count = token_valid.shape[1]
        packed_capacity = self.config.active_spatial_tokens * frames
        (packed_indices, packed_valid) = select_packed_token_indices(token_valid, packed_capacity)
        self_pair_mask = None
        if temporal_frame_mask is not None:
            if temporal_frame_mask.ndim == 2:
                temporal_frame_mask = temporal_frame_mask[None].expand(batch, -1, -1)
            expected = (batch, frames, frames)
            if (
                temporal_frame_mask.dtype != torch.bool
                or tuple(temporal_frame_mask.shape) != expected
            ):
                raise ValueError(f"temporal_frame_mask must be boolean with shape {expected}")
            dense_frame = torch.arange(frames, device=latent.device).repeat_interleave(rows * cols)
            dense_frame = dense_frame[None].expand(batch, -1)
            packed_frame = dense_frame.gather(1, packed_indices)
            batch_index = torch.arange(batch, device=latent.device)[:, None, None]
            self_pair_mask = temporal_frame_mask.to(latent.device)[
                batch_index, packed_frame[:, :, None], packed_frame[:, None, :]
            ]
        hidden = self.patch_embedding_mlp(gather_packed_tokens(patches, packed_indices))
        camera = self._camera_tokens(rows, cols).to(hidden.dtype)
        camera = camera.repeat(1, frames, 1).expand(batch, -1, -1)
        hidden = hidden + gather_packed_tokens(camera, packed_indices)
        hidden = hidden * packed_valid.unsqueeze(-1).to(hidden.dtype)
        positions = factorized_grid_positions(
            batch,
            frames,
            rows,
            cols,
            hidden.device,
            row_bases=self.config.camera_rope_bases,
            time_positions=time_positions,
        )
        positions = gather_packed_tokens(positions, packed_indices)
        rope = build_rotary_frequencies(positions, self.config.head_dim)
        compact_text = prepare_zero_padded_text(
            text_emb,
            prefix_length=self.config.text_prefix_length,
            logical_length=self.config.text_length,
        )
        text = self.condition_embedder.text(compact_text, hidden.dtype)
        (time_embedding, time_projection) = self.condition_embedder.time(timestep, hidden.dtype)
        target_indices = set(self.config.teacher_target_layers)
        last_target_index = max(target_indices)
        captured = []
        for index, block in enumerate(self.blocks):
            result = block(
                hidden,
                time_projection,
                text,
                token_valid=packed_valid,
                self_pair_mask=self_pair_mask,
                text_valid=None,
                text_repeats=text_repeats,
                text_logical_length=self.config.text_length,
                text_prefix_length=self.config.text_prefix_length,
                self_positions=rope,
                return_self_kv=capture_kv and index in target_indices,
            )
            if capture_kv and index in target_indices:
                (hidden, key, value) = result
                dense_key = scatter_packed_tokens(
                    key.transpose(1, 2), packed_indices, dense_token_count, packed_valid
                ).transpose(1, 2)
                dense_value = scatter_packed_tokens(
                    value.transpose(1, 2), packed_indices, dense_token_count, packed_valid
                ).transpose(1, 2)
                spatial = rows * cols
                captured.append(
                    LayerKV(
                        dense_key.reshape(
                            batch, dense_key.shape[1], frames, spatial, dense_key.shape[-1]
                        ).transpose(1, 2),
                        dense_value.reshape(
                            batch, dense_value.shape[1], frames, spatial, dense_value.shape[-1]
                        ).transpose(1, 2),
                    )
                )
            else:
                hidden = result
            if stop_after_last_capture and index == last_target_index:
                break
        if capture_kv and len(captured) != len(self.config.teacher_target_layers):
            raise RuntimeError("incomplete teacher K/V capture")
        hidden = scatter_packed_tokens(hidden, packed_indices, dense_token_count, packed_valid)
        return (
            hidden,
            tuple(captured),
            token_valid.reshape(batch, frames, rows * cols),
            (rows, cols),
            time_embedding,
        )

    @torch.no_grad()
    def _extract_teacher_carrier_kv_impl(
        self,
        current_latent: torch.Tensor,
        future_noise: torch.Tensor,
        current_spatial_valid: torch.Tensor,
        text_emb: torch.Tensor,
        video_dt: torch.Tensor,
    ) -> DenseKV:
        """Run one causal prefill over current latents and future noise."""
        batch = current_latent.shape[0]
        expected_noise = (
            batch,
            self.config.flow_frames - 1,
            self.config.latent_channels,
            current_latent.shape[-2],
            current_latent.shape[-1],
        )
        if tuple(future_noise.shape) != expected_noise:
            raise ValueError(f"teacher-carrier future noise must have shape {expected_noise}")
        expected_valid = (batch, current_latent.shape[-2], current_latent.shape[-1])
        if tuple(current_spatial_valid.shape) != expected_valid:
            raise ValueError(f"teacher-carrier spatial validity must have shape {expected_valid}")
        if tuple(video_dt.shape) != (batch,):
            raise ValueError("video_dt must have shape (B,)")
        latent = torch.cat((current_latent[:, None], future_noise), dim=1)
        spatial_valid = current_spatial_valid[:, None].expand(-1, self.config.flow_frames, -1, -1)
        frame_index = torch.arange(
            self.config.flow_frames, device=latent.device, dtype=torch.float32
        )
        physical_time = (
            video_dt.float()[:, None] * frame_index[None] / self.config.video_dt_reference
        )
        timestep = torch.full(
            (batch,), self.config.flow_timestep_scale, device=latent.device, dtype=torch.float32
        )
        (_, layers, token_valid, _, _) = self._encode(
            latent,
            spatial_valid,
            text_emb,
            timestep,
            capture_kv=True,
            time_positions=physical_time,
            temporal_frame_mask=causal_temporal_frame_mask(self.config.flow_frames, latent.device),
            stop_after_last_capture=True,
        )
        selected = torch.tensor((0, *self.config.horizons), device=latent.device, dtype=torch.long)
        return DenseKV(
            tuple(
                (
                    LayerKV(
                        layer.key.index_select(1, selected), layer.value.index_select(1, selected)
                    )
                    for layer in layers
                )
            ),
            token_valid.index_select(1, selected),
        )


class LiveWanWorld(EvolvingWanWorld):
    pass


class JointPrefillWanCarrier(nn.Module):
    """One Wan prefill supplying the action reader with world K/V."""

    carrier_kind = "joint_max_noise_future_prefill"

    def __init__(self, config: ZerithVAConfig, *, allow_uninitialized: bool = False) -> None:
        super().__init__()
        self.config = config
        self.wan = LiveWanWorld(config, allow_uninitialized=allow_uninitialized)
        generator = torch.Generator(device="cpu").manual_seed(0)
        self.register_buffer(
            "fixed_future_noise",
            torch.randn(
                1,
                config.flow_frames - 1,
                config.latent_channels,
                config.canonical_latent_height,
                config.canonical_latent_width,
                generator=generator,
            ),
        )

    def future_noise(self, current_latent: torch.Tensor) -> torch.Tensor:
        return self.fixed_future_noise.to(
            device=current_latent.device, dtype=current_latent.dtype
        ).expand(current_latent.shape[0], -1, -1, -1, -1)

    @torch.no_grad()
    def construct_world_kv(
        self,
        current_latent: torch.Tensor,
        current_spatial_valid: torch.Tensor,
        text_emb: torch.Tensor,
        video_dt: torch.Tensor,
    ) -> DenseKV:
        return self.wan.extract_teacher_carrier_kv(
            current_latent,
            self.future_noise(current_latent),
            current_spatial_valid,
            text_emb,
            video_dt,
        )
