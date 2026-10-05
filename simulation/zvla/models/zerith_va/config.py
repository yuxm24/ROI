"""Architecture configuration for the Zerith VA policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple

ZERITH_VA_TEACHER_ARCHITECTURE_VERSION = (
    "zerith-va-action20-linear-g4d-teacher")
# Read artifacts exported before the model was renamed without changing weights.
LEGACY_TEACHER_ARCHITECTURE_VERSION = (
    "fam-wam-action20-linear-g4d-teacher-v5.2")


@dataclass(frozen=True)
class ZerithVAConfig:
    """Defaults are the released Zerith VA architecture; benches override only
    the action interface fields."""

    architecture_version: str = ZERITH_VA_TEACHER_ARCHITECTURE_VERSION

    latent_channels: int = 48
    # Twenty columns are a transport envelope: robots use relative
    # xyz + rotation6D + gripper per arm, LIBERO uses native OSC7 in 0:7 and
    # MetaWorld uses its applied 4-D command in 0:4.
    action_dim: int = 20
    state_dim: int = 20
    state_trajectory_contract: bool = True
    native7_minmax: bool = True
    metaworld_native4: bool = False
    # One codec row per embodiment; ``embodiment_id`` only selects the row.
    state_codec_embodiments: Tuple[str, ...] = ("libero_franka", )
    # Telemetry labels only; they never alter sampling or the loss.
    training_source_names: Tuple[str, ...] = ("libero", )
    use_proprioception: bool = True
    # Four native-rate control steps elapse between consecutive video latents.
    action_chunk_size: int = 4
    # The Reader predicts a 16-step plan; deployment executes the first chunk.
    action_prediction_steps: int = 16
    context_frames: int = 1
    text_dim: int = 4096
    text_length: int = 512
    # Only this prefix is stored; the remaining Wan context is exact zeros.
    text_prefix_length: int = 64
    patch_size: int = 2
    camera_slots: int = 3
    max_active_camera_slots: int = 3
    camera_rope_bases: Tuple[int, ...] = (0, 32, 64)
    canonical_latent_height: int = 42
    canonical_latent_width: int = 14

    # VGGT patch features pooled to the same 7x7 grid per camera slot.
    geometry_feature_dim: int = 2048
    geometry_tokens_per_camera: int = 49
    g4d_binding_rank: int = 256
    g4d_binding_fusion: str = "linear_product"

    # Wan2.2-TI2V-5B width.
    student_dim: int = 3072
    student_heads: int = 24
    student_ffn_dim: int = 14336
    kv_tap_layers: Tuple[int, ...] = tuple(range(10))

    reader_dim: int = 3072
    reader_heads: int = 24
    reader_ffn_dim: int = 14336
    reader_layers: int = 10
    reader_kv_indices: Tuple[int, ...] = tuple(range(10))
    teacher_layers: int = 30
    # Wan blocks whose self-attention K/V feed the Reader.
    teacher_target_layers: Tuple[int, ...] = (9, 10, 11, 12, 13, 14, 16, 18,
                                              23, 24)

    horizons: Tuple[int, ...] = (1, 2, 4)
    flow_frames: int = 5
    action_flow_steps: int = 10
    flow_timestep_scale: float = 1000.0
    # Dimensionless RoPE/conditioning references only. Samples carry their
    # native physical dt and are never resampled to these rates.
    primitive_dt_reference: float = 0.05
    video_dt_reference: float = 0.2
    world_flow_weight: float = 1.0
    # Action-to-Wan gradients are withheld for the first 5% of training and
    # then introduced linearly over the next 5%; world flow always trains Wan.
    action_to_wan_freeze_fraction: float = 0.05
    action_to_wan_ramp_fraction: float = 0.05
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.student_dim // self.student_heads

    @property
    def spatial_tokens(self) -> int:
        height = self.canonical_latent_height // self.patch_size
        width = self.canonical_latent_width // self.patch_size
        return height * width

    @property
    def active_spatial_tokens(self) -> int:
        return (self.spatial_tokens // self.camera_slots *
                self.max_active_camera_slots)

    @property
    def kv_layers(self) -> int:
        return len(self.kv_tap_layers)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ZerithVAConfig":
        normalized = dict(value)
        if normalized.get("architecture_version") == LEGACY_TEACHER_ARCHITECTURE_VERSION:
            normalized["architecture_version"] = ZERITH_VA_TEACHER_ARCHITECTURE_VERSION
        for key in ("camera_rope_bases", "state_codec_embodiments",
                    "training_source_names", "kv_tap_layers",
                    "reader_kv_indices", "teacher_target_layers", "horizons"):
            if key in normalized:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)

    @classmethod
    def tiny(cls, **overrides: Any) -> "ZerithVAConfig":
        """A few-thousand-parameter model for unit tests."""
        values: Dict[str, Any] = {
            "latent_channels": 6,
            "text_dim": 12,
            "text_length": 3,
            "text_prefix_length": 3,
            "canonical_latent_height": 12,
            "canonical_latent_width": 4,
            "camera_rope_bases": (0, 4, 8),
            "geometry_feature_dim": 24,
            "geometry_tokens_per_camera": 4,
            "g4d_binding_rank": 8,
            "student_dim": 32,
            "student_heads": 4,
            "student_ffn_dim": 64,
            "kv_tap_layers": (0, 1),
            "reader_dim": 32,
            "reader_heads": 4,
            "reader_ffn_dim": 64,
            "reader_layers": 2,
            "reader_kv_indices": (0, 1),
            "teacher_layers": 4,
            "teacher_target_layers": (1, 3),
            "action_flow_steps": 3,
        }
        values.update(overrides)
        return cls(**values)
