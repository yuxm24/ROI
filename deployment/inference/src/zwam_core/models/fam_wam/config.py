# SPDX-License-Identifier: Apache-2.0
# Inference structural configuration only; see NOTICE.
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple

FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION = "fam-wam-action20-linear-g4d-teacher-v5.2"


@dataclass(frozen=True)
class FaMWAMConfig:
    """Production defaults; tests must request :meth:`tiny` explicitly."""

    architecture_version: str = FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION
    latent_channels: int = 48
    action_dim: int = 23
    state_dim: int = 23
    absolute_pose_contract: bool = False
    absolute_pose_dim: int = 16
    state_trajectory_contract: bool = True
    native7_minmax: bool = True
    h1_command23: bool = True
    action_normalization: str = "legacy"
    state_trajectory_dim: int = 23
    state_codec_embodiments: Tuple[str, ...] = ("h1",)
    use_proprioception: bool = True
    action_chunk_size: int = 4
    action_prediction_steps: int = 16
    context_frames: int = 12
    text_dim: int = 4096
    text_length: int = 64
    text_prefix_length: int = 64
    patch_size: int = 2
    camera_slots: int = 3
    max_active_camera_slots: int = 3
    camera_rope_bases: Tuple[int, ...] = (0, 32, 64)
    canonical_latent_height: int = 42
    canonical_latent_width: int = 14
    geometry_feature_dim: int = 2048
    geometry_tokens_per_camera: int = 49
    g4d_binding_rank: int = 256
    g4d_binding_fusion: str = "linear_product"
    student_dim: int = 3072
    student_heads: int = 24
    student_ffn_dim: int = 14336
    student_layers: int = 10
    student_source_layers: Tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29)
    kv_tap_layers: Tuple[int, ...] = (1, 3, 5, 7, 9)
    reader_dim: int = 3072
    reader_heads: int = 24
    reader_ffn_dim: int = 14336
    reader_layers: int = 5
    reader_source_layers: Tuple[int, ...] = (5, 11, 17, 23, 29)
    reader_kv_indices: Tuple[int, ...] = (0, 1, 2, 3, 4)
    teacher_layers: int = 30
    teacher_target_layers: Tuple[int, ...] = (5, 11, 17, 23, 29)
    horizons: Tuple[int, ...] = (1, 2, 4)
    flow_frames: int = 5
    action_flow_steps: int = 10
    flow_timestep_scale: float = 1000.0
    primitive_dt_reference: float = 0.05
    video_dt_reference: float = 0.2
    eps: float = 1e-06

    def __post_init__(self) -> None:
        integer_positive = (
            "latent_channels",
            "action_dim",
            "state_dim",
            "absolute_pose_dim",
            "state_trajectory_dim",
            "action_chunk_size",
            "action_prediction_steps",
            "context_frames",
            "text_dim",
            "text_length",
            "text_prefix_length",
            "patch_size",
            "camera_slots",
            "max_active_camera_slots",
            "geometry_feature_dim",
            "geometry_tokens_per_camera",
            "g4d_binding_rank",
            "student_dim",
            "student_heads",
            "student_ffn_dim",
            "student_layers",
            "reader_dim",
            "reader_heads",
            "reader_ffn_dim",
            "reader_layers",
            "teacher_layers",
            "flow_frames",
            "action_flow_steps",
        )
        invalid = {
            name: getattr(self, name) for name in integer_positive if getattr(self, name) <= 0
        }
        if invalid:
            raise ValueError(f"FaM-WAM values must be positive: {invalid}")
        if (
            self.architecture_version != FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION
            or not self.h1_command23
            or not self.state_trajectory_contract
            or self.absolute_pose_contract
            or not self.native7_minmax
            or self.action_normalization != "legacy"
            or self.action_prediction_steps != 16
            or (self.action_dim, self.state_dim, self.state_trajectory_dim) != (23, 23, 23)
        ):
            raise ValueError("This runtime requires the v5.2 H1 command23 contract and horizon16")
        if (
            "h1" not in self.state_codec_embodiments
            or any(
                not isinstance(value, str) or not value for value in self.state_codec_embodiments
            )
            or len(set(self.state_codec_embodiments)) != len(self.state_codec_embodiments)
        ):
            raise ValueError("Codec embodiments must be unique non-empty names including h1")
        if self.student_dim != self.reader_dim:
            raise ValueError("Student and Reader must share the Wan K/V width")
        if self.action_prediction_steps < self.action_chunk_size:
            raise ValueError(
                "action_prediction_steps cannot be shorter than the executed action chunk"
            )
        if self.action_prediction_steps % self.action_chunk_size:
            raise ValueError(
                "action_prediction_steps must contain an integer number of action chunks"
            )
        if self.text_prefix_length > self.text_length:
            raise ValueError("text_prefix_length cannot exceed logical text_length")
        if self.student_heads != self.reader_heads:
            raise ValueError("Student and Reader must share the Wan head layout")
        if self.student_dim % self.student_heads:
            raise ValueError("hidden dimensions must divide evenly by head counts")
        if len(self.student_source_layers) != self.student_layers:
            raise ValueError("student_source_layers must map every Student layer")
        if tuple(sorted(set(self.kv_tap_layers))) != self.kv_tap_layers:
            raise ValueError("kv_tap_layers must be sorted and unique")
        if min(self.kv_tap_layers) < 0 or max(self.kv_tap_layers) >= self.student_layers:
            raise ValueError("kv_tap_layers contains a layer outside the Student")
        mapped_targets = tuple((self.student_source_layers[index] for index in self.kv_tap_layers))
        if tuple(self.teacher_target_layers) != mapped_targets:
            raise ValueError(
                "Teacher targets must match the Wan coordinates selected by kv_tap_layers"
            )
        if len(self.reader_source_layers) != self.reader_layers:
            raise ValueError("reader_source_layers must map every Reader layer")
        if len(self.reader_kv_indices) != self.reader_layers:
            raise ValueError("reader_kv_indices must map every Reader layer")
        if min(self.reader_kv_indices) < 0 or max(self.reader_kv_indices) >= len(
            self.teacher_target_layers
        ):
            raise ValueError("reader_kv_indices contains an index outside the world K/V interface")
        expected_reader_sources = tuple(
            (self.teacher_target_layers[index] for index in self.reader_kv_indices)
        )
        if tuple(self.reader_source_layers) != expected_reader_sources:
            raise ValueError("Reader source layers must match their world K/V coordinates")
        if (
            min(self.teacher_target_layers) < 0
            or max(self.teacher_target_layers) >= self.teacher_layers
        ):
            raise ValueError("teacher target layer lies outside the Wan backbone")
        if len(set(self.teacher_target_layers)) != len(self.teacher_target_layers):
            raise ValueError("teacher target layers must be unique")
        if len(self.camera_rope_bases) != self.camera_slots:
            raise ValueError("camera_rope_bases must contain one base per slot")
        if self.max_active_camera_slots > self.camera_slots:
            raise ValueError("max_active_camera_slots cannot exceed camera_slots")
        if tuple(sorted(set(self.camera_rope_bases))) != self.camera_rope_bases:
            raise ValueError("camera RoPE bases must be sorted and unique")
        if (
            self.canonical_latent_height % self.patch_size
            or self.canonical_latent_width % self.patch_size
            or self.canonical_latent_height // self.patch_size % self.camera_slots
        ):
            raise ValueError("canonical latent grid must patchify into equal camera row blocks")
        if self.geometry_tokens_per_camera != self.spatial_tokens // self.camera_slots:
            raise ValueError("geometry tokens must match one canonical Wan camera grid")
        if self.g4d_binding_rank > self.reader_dim:
            raise ValueError("g4d_binding_rank cannot exceed the Wan/Reader width")
        if self.g4d_binding_fusion != "linear_product":
            raise ValueError("H1 inference requires g4d_binding_fusion=linear_product")
        if tuple(self.horizons) != (1, 2, 4):
            raise ValueError("FaM-WAM fixes future horizons to (1,2,4)")
        if self.flow_frames != max(self.horizons) + 1:
            raise ValueError("world flow clip must contain o_t through o_t+4")

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
        if self.spatial_tokens % self.camera_slots:
            raise ValueError("spatial token grid must divide across camera slots")
        return self.spatial_tokens // self.camera_slots * self.max_active_camera_slots

    @property
    def kv_layers(self) -> int:
        return len(self.kv_tap_layers)

    @property
    def action_prediction_chunks(self) -> int:
        return self.action_prediction_steps // self.action_chunk_size

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "FaMWAMConfig":
        tuple_fields = {
            "camera_rope_bases",
            "state_codec_embodiments",
            "student_source_layers",
            "kv_tap_layers",
            "reader_source_layers",
            "reader_kv_indices",
            "teacher_target_layers",
            "horizons",
        }
        normalized = dict(value)
        for key in tuple_fields:
            if key in normalized:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "FaMWAMConfig":
        if not isinstance(value, Mapping):
            raise TypeError("FaM-WAM configuration document must be a mapping")
        if "model" in value:
            value = value["model"]["config"]
        elif "config" in value:
            value = value["config"]
        return cls.from_dict(dict(value))

    @classmethod
    def tiny(cls, **overrides: Any) -> "FaMWAMConfig":
        values: Dict[str, Any] = {
            "latent_channels": 6,
            "context_frames": 3,
            "text_dim": 12,
            "text_length": 3,
            "text_prefix_length": 3,
            "patch_size": 2,
            "camera_slots": 3,
            "max_active_camera_slots": 3,
            "camera_rope_bases": (0, 4, 8),
            "canonical_latent_height": 12,
            "canonical_latent_width": 4,
            "geometry_feature_dim": 24,
            "geometry_tokens_per_camera": 4,
            "g4d_binding_rank": 8,
            "student_dim": 32,
            "student_heads": 4,
            "student_ffn_dim": 64,
            "student_layers": 4,
            "student_source_layers": (0, 1, 2, 3),
            "kv_tap_layers": (1, 3),
            "reader_dim": 32,
            "reader_heads": 4,
            "reader_ffn_dim": 64,
            "reader_layers": 2,
            "reader_source_layers": (1, 3),
            "reader_kv_indices": (0, 1),
            "teacher_layers": 4,
            "teacher_target_layers": (1, 3),
            "action_flow_steps": 3,
            "use_proprioception": False,
        }
        values.update(overrides)
        return cls(**values)
