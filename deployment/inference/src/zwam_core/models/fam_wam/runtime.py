# SPDX-License-Identifier: Apache-2.0
# Adapted from the ZWAM/ZVLA model source. Inference-only distribution.
# See NOTICE and the repository LICENSE_POLICY.md.
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from zwam_core.actions.mixed_native7 import codec_class

from .config import (
    FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION,
    FaMWAMConfig,
)
from .geometry import GeometryAnchored4DBinder, GeometryTokenBridge
from .policy_reader import PolicyReader, integrate_action_flow
from .primitives import async_tensor_assert
from .teacher import JointPrefillWanCarrier


@dataclass
class FaMInferenceState:
    seed: int = 0
    batch_size: Optional[int] = None
    counters: Dict[str, int] = field(
        default_factory=lambda: {"world_builds": 0, "world_prefills": 0, "reader_calls": 0}
    )


class FaMWAMPolicy(nn.Module):
    """One world prefill followed by cached iterative action flow."""

    def deployment_modules(self) -> Mapping[str, nn.Module]:
        return {"world": self.world, "policy_reader": self.policy_reader}

    def reset_inference(self, seed: int = 0) -> None:
        self._runtime = FaMInferenceState(seed=int(seed))
        self._generator = None

    def _ensure_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None:
            generator_device = device if device.type == "cuda" else torch.device("cpu")
            self._generator = torch.Generator(device=generator_device)
            self._generator.manual_seed(self._runtime.seed)
        return self._generator

    def _inference_autocast(self, device: torch.device):
        dtype = next(self.parameters()).dtype
        if device.type in {"cpu", "cuda"} and dtype in {torch.bfloat16, torch.float16}:
            return torch.autocast(device_type=device.type, dtype=dtype)
        return nullcontext()

    def _validate_batch(self, batch: int) -> None:
        if self._runtime.batch_size is None:
            self._runtime.batch_size = batch
        elif self._runtime.batch_size != batch:
            raise RuntimeError("batch size changed within an episode")

    def _validate_timebase(
        self, batch: int, primitive_dt: torch.Tensor, video_dt: torch.Tensor
    ) -> None:
        if tuple(primitive_dt.shape) != (batch,) or tuple(video_dt.shape) != (batch,):
            raise ValueError("primitive_dt and video_dt must both have shape (B,)")
        async_tensor_assert(
            torch.isfinite(primitive_dt).all() & (primitive_dt > 0).all(),
            "primitive_dt must be finite and positive",
        )
        if self.config.state_trajectory_contract:
            async_tensor_assert(
                torch.isfinite(video_dt).all()
                & torch.isclose(
                    video_dt, primitive_dt * self.config.action_chunk_size, rtol=1e-05, atol=1e-07
                ).all(),
                "video_dt must equal action_chunk_size * primitive_dt",
            )

    @torch.no_grad()
    def act(
        self,
        observation_latent: torch.Tensor,
        observation_spatial_valid: torch.Tensor,
        text_emb: torch.Tensor,
        primitive_dt: torch.Tensor,
        video_dt: torch.Tensor,
        action_feature_valid: Optional[torch.Tensor] = None,
        *,
        current_state: Optional[torch.Tensor] = None,
        current_state_valid: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        external_camera_weight: int = 1,
        geometry_features: Optional[torch.Tensor] = None,
        geometry_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_timebase(int(observation_latent.shape[0]), primitive_dt, video_dt)
        with self._inference_autocast(observation_latent.device):
            return self._act_impl(
                observation_latent,
                observation_spatial_valid,
                text_emb,
                primitive_dt,
                video_dt,
                action_feature_valid,
                current_state=current_state,
                current_state_valid=current_state_valid,
                noise=noise,
                external_camera_weight=external_camera_weight,
                geometry_features=geometry_features,
                geometry_valid=geometry_valid,
            )

    def _act_impl(
        self,
        observation_latent: torch.Tensor,
        observation_spatial_valid: torch.Tensor,
        text_emb: torch.Tensor,
        primitive_dt: torch.Tensor,
        video_dt: torch.Tensor,
        action_feature_valid: Optional[torch.Tensor],
        *,
        current_state: Optional[torch.Tensor],
        current_state_valid: Optional[torch.Tensor],
        noise: Optional[torch.Tensor],
        external_camera_weight: int,
        geometry_features: Optional[torch.Tensor],
        geometry_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch = observation_latent.shape[0]
        self._validate_batch(batch)
        cache = self._construct_condition_cache(
            observation_latent,
            observation_spatial_valid,
            text_emb,
            video_dt,
            current_state,
            current_state_valid,
            external_camera_weight,
            geometry_features,
            geometry_valid,
        )
        self._runtime.counters["world_builds"] += 1
        self._runtime.counters["world_prefills"] += 1
        if action_feature_valid is None:
            action_feature_valid = torch.ones(
                batch,
                self.config.action_prediction_steps,
                self.config.action_dim,
                device=observation_latent.device,
                dtype=torch.bool,
            )
        if noise is None:
            noise = torch.randn(
                action_feature_valid.shape,
                generator=self._ensure_generator(observation_latent.device),
                device=observation_latent.device,
                dtype=torch.float32,
            )
        action = integrate_action_flow(
            self.policy_reader,
            action_feature_valid,
            primitive_dt,
            cache,
            noise,
            self.config.action_flow_steps,
            current_state,
            current_state_valid,
            fp32_accumulation=True,
        )
        self._runtime.counters["reader_calls"] += self.config.action_flow_steps
        return action


class FaMWAMV5TeacherPolicy(FaMWAMPolicy):
    """H1 command23 policy with Wan dynamics and VGGT geometry."""

    def __init__(self, config: FaMWAMConfig) -> None:
        super().__init__()
        if config.architecture_version != FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION:
            raise ValueError("Unsupported inference architecture")
        self.config = config
        self.world = JointPrefillWanCarrier(config, allow_uninitialized=True)
        self.policy_reader = PolicyReader(config)
        self._runtime = FaMInferenceState()
        self._generator = None
        self.geometry_bridge = GeometryTokenBridge(config)
        self.g4d_binder = GeometryAnchored4DBinder(config)
        # Materialize small codec buffers on CPU even for a meta network.
        with torch.device("cpu"):
            self.state_codec = codec_class(config)(embodiments=config.state_codec_embodiments)

    def deployment_modules(self) -> Mapping[str, nn.Module]:
        return {
            "world": self.world,
            "policy_reader": self.policy_reader,
            "geometry_bridge": self.geometry_bridge,
            "g4d_binder": self.g4d_binder,
            "state_codec": self.state_codec,
        }

    def _construct_condition_cache(
        self,
        observation_latent: torch.Tensor,
        observation_spatial_valid: torch.Tensor,
        text_emb: torch.Tensor,
        video_dt: torch.Tensor,
        current_state: Optional[torch.Tensor],
        current_state_valid: Optional[torch.Tensor],
        external_camera_weight: int,
        geometry_features: Optional[torch.Tensor],
        geometry_valid: Optional[torch.Tensor],
    ):
        del current_state, current_state_valid
        if geometry_features is None or geometry_valid is None:
            raise ValueError("The H1 policy requires online VGGT features")
        world = self.world.construct_world_kv(
            observation_latent, observation_spatial_valid, text_emb, video_dt
        )
        geometry_valid = geometry_valid.bool()
        geometry_tokens = self.geometry_bridge(geometry_features, geometry_valid)
        world = self.g4d_binder(world, geometry_tokens, geometry_valid)
        return self.policy_reader.build_condition_cache(
            world,
            text_emb,
            external_camera_weight=external_camera_weight,
            geometry_tokens=geometry_tokens,
            geometry_valid=geometry_valid,
        )

    @torch.no_grad()
    def act_state_trajectory(
        self,
        observation_latent: torch.Tensor,
        observation_spatial_valid: torch.Tensor,
        text_emb: torch.Tensor,
        primitive_dt: torch.Tensor,
        video_dt: torch.Tensor,
        current_state_abs: torch.Tensor,
        current_state_abs_valid: torch.Tensor,
        embodiment_id: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        external_camera_weight: int = 1,
        geometry_features: Optional[torch.Tensor] = None,
        geometry_valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict an absolute H1 command23 trajectory from the current state.

        The first three columns describe the waist; each arm contributes
        ``xyz + rotation6D + gripper``.
        Callers must not normalize, difference, or integrate the returned
        trajectory.  Controller-specific command conversion happens after
        this public model boundary. ``primitive_dt`` is the native state-row
        period and ``video_dt`` must be four times that value; both affect
        temporal RoPE and never request input resampling.
        """
        current_valid = current_state_abs_valid.bool()
        (current_state, current_state_valid) = self.state_codec.encode_inference_state(
            current_state_abs, current_valid, embodiment_id
        )
        action_valid = self.state_codec.inference_action_valid(
            current_state_abs, current_valid, embodiment_id, self.config.action_prediction_steps
        )
        internal_action = self.act(
            observation_latent,
            observation_spatial_valid,
            text_emb,
            primitive_dt,
            video_dt,
            action_valid,
            current_state=current_state,
            current_state_valid=current_state_valid,
            noise=noise,
            external_camera_weight=external_camera_weight,
            geometry_features=geometry_features,
            geometry_valid=geometry_valid,
        )
        return self.state_codec.decode_predictions(
            internal_action, current_state_abs, current_valid, embodiment_id, action_valid
        )
