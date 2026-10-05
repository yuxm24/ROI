"""Zerith VA post-training model, inference policy and weight loaders.

Wan2.2 world prefill -> VGGT geometry bridge -> G4D value binding -> Reader.
The benchmark owns the state/action codec; the released base checkpoint
carries only the four neural modules.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open

from zvla.actions.mixed_native7 import codec_class
from zvla.core import OptimizerGroupSpec, VLAForwardOutput, VLAModel
from zvla.engines import VLAS

from .config import ZerithVAConfig
from .geometry import GeometryAnchored4DBinder, GeometryTokenBridge
from .kv import DenseKV, LayerKV
from .policy_reader import (
    PolicyReader,
    action_flow_matching_loss,
    integrate_action_flow,
)
from .teacher import JointPrefillWanCarrier

NEURAL_MODULES = ("teacher_carrier", "policy_reader", "geometry_bridge",
                  "g4d_binder")
ARTIFACT_MODULES = NEURAL_MODULES + ("state_codec", )
# MetaWorld's applied command (dx, dy, dz, gripper) reuses these OSC7 columns.
_NATIVE4_FROM_NATIVE7 = (0, 1, 2, 6)


def _trainable(module: nn.Module) -> Tuple[nn.Parameter, ...]:
    return tuple(p for p in module.parameters() if p.requires_grad)


def _scale_dense_kv_gradient(value: DenseKV, beta: float) -> DenseKV:
    """Keep forward values; scale only the action-path gradient into Wan."""

    def scaled(tensor: torch.Tensor) -> torch.Tensor:
        detached = tensor.detach()
        return detached + beta * (tensor - detached)

    return DenseKV(
        tuple(
            LayerKV(scaled(layer.key), scaled(layer.value))
            for layer in value.layers), value.valid)


def _read_module_state(handle, prefix: str, module: nn.Module,
                       derive_native4: bool) -> Dict[str, torch.Tensor]:
    state = {}
    columns = torch.tensor(_NATIVE4_FROM_NATIVE7)
    for name in module.state_dict():
        key = f"{prefix}.{name}"
        if derive_native4 and ".native4_" in f".{name}":
            value = handle.get_tensor(key.replace("native4_", "native7_"))
            if "output_projection" in name:
                value = value.index_select(0, columns)
            elif name.endswith(".weight"):
                value = value.index_select(1, columns)
        else:
            value = handle.get_tensor(key)
        state[name] = value
    return state


@VLAS.register_module()
class ZerithVAPostTrainModel(VLAModel):
    """Jointly trains Wan, the Reader, the geometry bridge and G4D."""

    def __init__(
        self,
        config: Optional[dict] = None,
        *,
        pretrained_checkpoint: Optional[str] = None,
        state_codec_stats: Optional[str] = None,
        wan_lr_scale: float = 1.0,
        reader_lr_scale: float = 1.0,
        reader_head_lr_scale: float = 10.0,
        geometry_lr_scale: float = 10.0,
        g4d_lr_scale: float = 10.0,
        g4d_gate_lr_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.fam_config = ZerithVAConfig.from_dict(config or {})
        self.teacher_carrier = JointPrefillWanCarrier(self.fam_config)
        self.policy_reader = PolicyReader(self.fam_config)
        self.geometry_bridge = GeometryTokenBridge(self.fam_config)
        self.g4d_binder = GeometryAnchored4DBinder(self.fam_config)
        codec = codec_class(self.fam_config)
        self.state_codec = (
            codec.from_json(
                state_codec_stats,
                expected_embodiments=self.fam_config.state_codec_embodiments)
            if state_codec_stats is not None else
            codec(embodiments=self.fam_config.state_codec_embodiments))
        self.pretrained_checkpoint = pretrained_checkpoint
        self.state_codec_stats = state_codec_stats
        self.wan_lr_scale = float(wan_lr_scale)
        self.reader_lr_scale = float(reader_lr_scale)
        self.reader_head_lr_scale = float(reader_head_lr_scale)
        self.geometry_lr_scale = float(geometry_lr_scale)
        self.g4d_lr_scale = float(g4d_lr_scale)
        self.g4d_gate_lr_scale = float(g4d_gate_lr_scale)
        self.initialization_report: Dict[str, object] = {}
        self._initialization_complete = False

    def load_pretrained_weights(self, checkpoint_path=None) -> None:
        """Copy the four neural modules; the target codec stays as built."""
        if self._initialization_complete or not self.pretrained_checkpoint:
            return
        path = Path(self.pretrained_checkpoint).expanduser()
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for prefix in NEURAL_MODULES:
                module = getattr(self, prefix)
                module.load_state_dict(
                    _read_module_state(handle, prefix, module,
                                       self.fam_config.metaworld_native4),
                    strict=True)
        self.initialization_report = {
            "pretrained_checkpoint": str(path),
            "modules": list(NEURAL_MODULES),
        }
        self._initialization_complete = True

    def action_to_wan_beta(self) -> float:
        """Action-to-Wan gradient coefficient for the current step."""
        total = getattr(self, "_train_total_steps", None)
        if total is None:
            return 1.0
        step = int(getattr(self, "_train_metric_step", 0))
        progress = min(1.0, max(0.0, step / float(max(1, int(total)))))
        freeze = self.fam_config.action_to_wan_freeze_fraction
        ramp = self.fam_config.action_to_wan_ramp_fraction
        if progress <= freeze:
            return 0.0
        if ramp == 0.0 or progress >= freeze + ramp:
            return 1.0
        return (progress - freeze) / ramp

    def forward(self, **robot: torch.Tensor) -> VLAForwardOutput:
        target, target_valid, state, state_valid = (
            self.state_codec.encode_training_targets(
                robot["current_state_abs"],
                robot["current_state_abs_valid"].bool(),
                robot["target_state_abs"],
                robot["target_state_abs_valid"].bool(),
                robot["embodiment_id"],
            ))
        metrics = {} if self.collect_train_metrics else None
        per_source = self.collect_source_train_metrics
        world_values = self.teacher_carrier.construct_training_world(
            robot["observation_latents"][:, -1],
            robot["future_latents"],
            robot["observation_spatial_valid"][:, -1],
            robot["future_spatial_valid"],
            robot["text_emb"],
            robot["video_dt"],
            return_per_sample=per_source,
        )
        flow, video_sigma, world = world_values[:3]
        beta = self.action_to_wan_beta()
        geometry_valid = robot["geometry_valid"].bool()
        geometry_tokens = self.geometry_bridge(robot["geometry_features"],
                                               geometry_valid)
        bound_world = self.g4d_binder(
            _scale_dense_kv_gradient(world, beta),
            geometry_tokens,
            geometry_valid,
            metrics_out=metrics,
        )
        condition = self.policy_reader.prepare_condition_source(
            bound_world,
            robot["text_emb"],
            geometry_tokens=geometry_tokens,
            geometry_valid=geometry_valid,
        )
        source_names = self.fam_config.training_source_names
        action = action_flow_matching_loss(
            self.policy_reader,
            target,
            target_valid,
            robot["primitive_dt"],
            condition,
            state,
            state_valid,
            sample_source_id=robot["source_id"] if per_source else None,
            sample_source_names=source_names if per_source else None,
            metrics_out=metrics,
        )
        if metrics is not None:
            valid_state = state[state_valid.bool()].detach().float()
            metrics.update({
                "loss_action_flow": action.detach(),
                "loss_world_flow": flow.detach(),
                "video_prefill_sigma": video_sigma.detach().mean(),
                "action_gradient_to_wan": float(beta),
                "geometry_valid_fraction": geometry_valid.float().mean(),
                "data_observation_valid_fraction":
                robot["observation_valid"].float().mean(),
                "data_spatial_valid_fraction":
                robot["observation_spatial_valid"].float().mean(),
                "data_action_feature_valid_fraction":
                target_valid.float().mean(),
                "data_state_valid_fraction": state_valid.float().mean(),
                "data_state_normalized_abs_mean": valid_state.abs().mean(),
            })
            if per_source:
                per_sample = world_values[3].detach().float()
                for identifier, name in enumerate(source_names):
                    selected = (robot["source_id"] == identifier).float()
                    prefix = f"loss_world_source_{name}"
                    metrics[prefix + "__numerator"] = (per_sample *
                                                       selected).sum()
                    metrics[prefix + "__denominator"] = selected.sum()
        return VLAForwardOutput(
            action + self.fam_config.world_flow_weight * flow,
            metrics=metrics or {},
        )

    def _reader_parameter_split(self):
        backbone = _trainable(self.policy_reader.layers)
        ids = {id(p) for p in backbone}
        heads = tuple(p for p in _trainable(self.policy_reader)
                      if id(p) not in ids)
        return backbone, heads

    def optimizer_group_specs(self) -> Tuple[OptimizerGroupSpec, ...]:
        backbone, heads = self._reader_parameter_split()
        return (
            OptimizerGroupSpec("wan", _trainable(self.teacher_carrier.wan),
                               self.wan_lr_scale),
            OptimizerGroupSpec("reader_backbone", backbone,
                               self.reader_lr_scale),
            OptimizerGroupSpec("reader_heads", heads,
                               self.reader_head_lr_scale),
            OptimizerGroupSpec("geometry_bridge",
                               _trainable(self.geometry_bridge),
                               self.geometry_lr_scale),
            OptimizerGroupSpec("g4d_projection",
                               self.g4d_binder.projection_parameters,
                               self.g4d_lr_scale),
            OptimizerGroupSpec("g4d_gate", self.g4d_binder.gate_parameters,
                               self.g4d_gate_lr_scale),
        )

    def training_metric_parameter_groups(
            self) -> Mapping[str, Tuple[nn.Parameter, ...]]:
        backbone, heads = self._reader_parameter_split()
        return OrderedDict((
            ("wan", _trainable(self.teacher_carrier.wan)),
            ("reader_backbone", backbone),
            ("reader_heads", heads),
            ("geometry_bridge", _trainable(self.geometry_bridge)),
            ("g4d_projection", self.g4d_binder.projection_parameters),
            ("g4d_gate", self.g4d_binder.gate_parameters),
        ))

    def training_metric_group_roles(self) -> Mapping[str, Tuple[str, ...]]:
        return {
            "wan": ("world", "vision"),
            "reader_backbone": ("action", ),
            "reader_heads": ("action", ),
            "geometry_bridge": ("world", ),
            "g4d_projection": ("world", ),
            "g4d_gate": ("world", ),
        }

    def checkpoint_modules(self) -> Mapping[str, nn.Module]:
        return OrderedDict(
            (name, getattr(self, name)) for name in ARTIFACT_MODULES)


class ZerithVAPolicy(nn.Module):
    """Inference policy: absolute State20 in, absolute State20 targets out."""

    def __init__(self, config: ZerithVAConfig) -> None:
        super().__init__()
        self.config = config
        self.world = JointPrefillWanCarrier(config)
        self.policy_reader = PolicyReader(config)
        self.geometry_bridge = GeometryTokenBridge(config)
        self.g4d_binder = GeometryAnchored4DBinder(config)
        self.state_codec = codec_class(config)(
            embodiments=config.state_codec_embodiments)
        self._seed = 0
        self._generator: Optional[torch.Generator] = None

    def reset_inference(self, seed: int = 0) -> None:
        self._seed = int(seed)
        self._generator = None

    def _noise(self, shape, device: torch.device) -> torch.Tensor:
        if self._generator is None:
            self._generator = torch.Generator(
                device=device if device.type == "cuda" else "cpu")
            self._generator.manual_seed(self._seed)
        return torch.randn(shape, generator=self._generator, device=device,
                           dtype=torch.float32)

    def _autocast(self, device: torch.device):
        dtype = next(self.parameters()).dtype
        if dtype in (torch.bfloat16, torch.float16):
            return torch.autocast(device_type=device.type, dtype=dtype)
        return nullcontext()

    @torch.no_grad()
    def act(
        self,
        observation_latent: torch.Tensor,
        observation_spatial_valid: torch.Tensor,
        text_emb: torch.Tensor,
        primitive_dt: torch.Tensor,
        video_dt: torch.Tensor,
        action_feature_valid: torch.Tensor,
        *,
        current_state: torch.Tensor,
        current_state_valid: torch.Tensor,
        geometry_features: torch.Tensor,
        geometry_valid: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the model-private normalized Action20 plan."""
        device = observation_latent.device
        with self._autocast(device):
            world = self.world.construct_world_kv(
                observation_latent, observation_spatial_valid, text_emb,
                video_dt)
            geometry_valid = geometry_valid.bool()
            geometry_tokens = self.geometry_bridge(geometry_features,
                                                   geometry_valid)
            world = self.g4d_binder(world, geometry_tokens, geometry_valid)
            cache = self.policy_reader.build_condition_cache(
                world,
                text_emb,
                geometry_tokens=geometry_tokens,
                geometry_valid=geometry_valid,
            )
            if noise is None:
                noise = self._noise(action_feature_valid.shape, device)
            return integrate_action_flow(
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
        geometry_features: torch.Tensor,
        geometry_valid: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict absolute desired State20 rows from absolute State20.

        ``primitive_dt`` is the native control period and ``video_dt`` is
        four times that value; inputs are never resampled.
        """
        current_valid = current_state_abs_valid.bool()
        state, state_valid = self.state_codec.encode_inference_state(
            current_state_abs, current_valid, embodiment_id)
        action_valid = self.state_codec.inference_action_valid(
            current_state_abs, current_valid, embodiment_id,
            self.config.action_prediction_steps)
        internal = self.act(
            observation_latent,
            observation_spatial_valid,
            text_emb,
            primitive_dt,
            video_dt,
            action_valid,
            current_state=state,
            current_state_valid=state_valid,
            geometry_features=geometry_features,
            geometry_valid=geometry_valid,
            noise=noise,
        )
        return self.state_codec.decode_predictions(
            internal, current_state_abs, current_valid, embodiment_id,
            action_valid)


def load_policy_from_model_artifact(
    path: str | Path,
    config: ZerithVAConfig,
) -> ZerithVAPolicy:
    """Load a training hand-off (``model-step-*.safetensors``) for export."""
    policy = ZerithVAPolicy(config)
    names = {"world": "teacher_carrier"}
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in policy.state_dict():
            prefix, rest = name.split(".", 1)
            state[name] = handle.get_tensor(f"{names.get(prefix, prefix)}.{rest}")
    policy.load_state_dict(state, strict=True)
    return policy.eval()
