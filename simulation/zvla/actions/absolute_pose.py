"""Model-owned absolute-pose/action transform for robot training.

The storage and public inference boundary is deliberately homogeneous:

``[arm0 xyz(m), qw qx qy qz, gripper_opening(0..1), arm1 ...]``.

Each arm is expressed in its own named, right-handed canonical arm-base FLU
frame (x forward, y left, z up).  The two arm bases are not assumed to share
an origin.  A dataset adapter must apply a verified source-base -> canonical-
arm-base transform; it may not silently identify ``base_link0`` and
``base_link1`` or precompute model-relative deltas.  Missing arms are
zero-filled and feature-masked.  This module alone owns the absolute <->
anchor-relative transform and is checkpointed with the policy so training and
inference execute byte-identical math.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from zvla.utils.shared_metadata import read_shared_text, shared_metadata_exists


ABSOLUTE_EEF_POSE_CONTRACT = "zvla-absolute-eef-pose-wxyz-base-flu-v1"
INTERNAL_ACTION_CONTRACT = "zvla-action32-anchor-relative-world-v1"
POSE_CODEC_STATS_FORMAT = "zvla-absolute-eef-pose-codec-stats-v1"
MODEL_SIDE_NORMALIZATION_ID = "model_side_absolute_pose_codec_v1"
ABSOLUTE_EEF_POSE_DIM = 16
INTERNAL_ACTION_DIM = 32

ABSOLUTE_EEF_POSE_LAYOUT = (
    "arm0.position.x_m",
    "arm0.position.y_m",
    "arm0.position.z_m",
    "arm0.orientation.qw",
    "arm0.orientation.qx",
    "arm0.orientation.qy",
    "arm0.orientation.qz",
    "arm0.gripper.opening_fraction",
    "arm1.position.x_m",
    "arm1.position.y_m",
    "arm1.position.z_m",
    "arm1.orientation.qw",
    "arm1.orientation.qx",
    "arm1.orientation.qy",
    "arm1.orientation.qz",
    "arm1.gripper.opening_fraction",
)

# The private tensor keeps the repository-wide Action32 coordinates so the
# policy head remains architecture-compatible with prior checkpoints. This
# absolute-EFF corpus enables only the two seven-wide EEF blocks. Joint, base,
# and reserved coordinates remain exact zero with a false feature mask.
INTERNAL_ACTION_LAYOUT = (
    "arm0.anchor_relative.translation.x_m",
    "arm0.anchor_relative.translation.y_m",
    "arm0.anchor_relative.translation.z_m",
    "arm0.anchor_relative.rotation_vector.x_rad",
    "arm0.anchor_relative.rotation_vector.y_rad",
    "arm0.anchor_relative.rotation_vector.z_rad",
    "arm0.gripper.command_close_pos1_open_neg1",
    "arm0.joint_delta.0_unused",
    "arm0.joint_delta.1_unused",
    "arm0.joint_delta.2_unused",
    "arm0.joint_delta.3_unused",
    "arm0.joint_delta.4_unused",
    "arm0.joint_delta.5_unused",
    "arm0.joint_delta.6_unused",
    "arm1.anchor_relative.translation.x_m",
    "arm1.anchor_relative.translation.y_m",
    "arm1.anchor_relative.translation.z_m",
    "arm1.anchor_relative.rotation_vector.x_rad",
    "arm1.anchor_relative.rotation_vector.y_rad",
    "arm1.anchor_relative.rotation_vector.z_rad",
    "arm1.gripper.command_close_pos1_open_neg1",
    "arm1.joint_delta.0_unused",
    "arm1.joint_delta.1_unused",
    "arm1.joint_delta.2_unused",
    "arm1.joint_delta.3_unused",
    "arm1.joint_delta.4_unused",
    "arm1.joint_delta.5_unused",
    "arm1.joint_delta.6_unused",
    "base.delta.translation.x_unused",
    "base.delta.translation.y_unused",
    "base.delta.rotation.yaw_unused",
    "reserved.0_unused",
)

_ARM_POSE_OFFSETS = (0, 8)
_ARM_ACTION_OFFSETS = (0, 14)
_EEF_ACTION_INDICES = tuple(range(0, 7)) + tuple(range(14, 21))
INTERNAL_ACTION_VALID_INDICES = _EEF_ACTION_INDICES
_EPS = 1e-8

if len(ABSOLUTE_EEF_POSE_LAYOUT) != ABSOLUTE_EEF_POSE_DIM:
    raise RuntimeError("absolute EEF pose layout width drifted")
if len(INTERNAL_ACTION_LAYOUT) != INTERNAL_ACTION_DIM:
    raise RuntimeError("private Action32 layout width drifted")


def _normalize_quaternion_wxyz(value: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value.float(), dim=-1, keepdim=True)
    normalized = value.float() / norm.clamp_min(_EPS)
    identity = torch.zeros_like(normalized)
    identity[..., 0] = 1.0
    normalized = torch.where(norm > _EPS, normalized, identity)
    # q and -q encode the same rotation.  ``w >= 0`` alone is not unique for
    # exact 180-degree rotations (w == 0), including the authoritative Franka
    # source transform.  Use the first non-negligible w/x/y/z component as a
    # deterministic lexicographic hemisphere tie-break.
    pivot = torch.zeros_like(normalized[..., 0])
    for component in reversed(normalized.unbind(dim=-1)):
        pivot = torch.where(component.abs() > _EPS, component, pivot)
    return torch.where(
        pivot[..., None] < 0.0, -normalized, normalized)


def _quaternion_conjugate_wxyz(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., :1], -value[..., 1:]), dim=-1)


def _quaternion_multiply_wxyz(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    result = torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1)
    return _normalize_quaternion_wxyz(result).to(result.dtype)


def _quaternion_wxyz_to_rotation_vector(value: torch.Tensor) -> torch.Tensor:
    value = _normalize_quaternion_wxyz(value)
    xyz = value[..., 1:]
    xyz_norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(
        xyz_norm,
        value[..., :1].clamp(0.0, 1.0),
    )
    scale = torch.where(
        xyz_norm <= _EPS,
        2.0 + xyz_norm.square() / 3.0,
        angle / xyz_norm.clamp_min(_EPS),
    )
    return xyz * scale


def _rotation_vector_to_quaternion_wxyz(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    angle = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle <= _EPS,
        0.5 - angle.square() / 48.0,
        torch.sin(half) / angle.clamp_min(_EPS),
    )
    return _normalize_quaternion_wxyz(
        torch.cat((torch.cos(half), value * scale), dim=-1))


def _strict_vector(
    document: Mapping[str, Any],
    key: str,
    width: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if key not in document:
        raise KeyError(f"pose codec stats are missing {key!r}")
    value = torch.as_tensor(document[key], dtype=dtype)
    if tuple(value.shape) != (width,):
        raise ValueError(
            f"pose codec stats {key} must have shape ({width},), got "
            f"{tuple(value.shape)}")
    if dtype != torch.bool and not torch.isfinite(value).all():
        raise ValueError(f"pose codec stats {key} contains NaN or Inf")
    return value


class CanonicalAbsolutePoseCodec(nn.Module):
    """Encode raw absolute EEF targets and decode policy outputs.

    Internal motion targets are relative to the *anchor observation* for the
    whole predicted chunk: translation is ``p_target - p_anchor`` in the same
    canonical per-arm base and rotation is the base-frame left delta
    ``R_target @ R_anchor^-1``.  Inference uses the exact inverse with the same
    anchor; it never accumulates predicted deltas step by step.
    """

    def __init__(self, stats: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        if stats is None:
            action_valid = torch.zeros(INTERNAL_ACTION_DIM, dtype=torch.bool)
            action_valid[list(_EEF_ACTION_INDICES)] = True
            stats = {
                "format": POSE_CODEC_STATS_FORMAT,
                "input_contract": ABSOLUTE_EEF_POSE_CONTRACT,
                "internal_action_contract": INTERNAL_ACTION_CONTRACT,
                "state_q01": [-1.0] * ABSOLUTE_EEF_POSE_DIM,
                "state_q99": [1.0] * ABSOLUTE_EEF_POSE_DIM,
                "state_stat_valid": [True] * ABSOLUTE_EEF_POSE_DIM,
                "action_q01": [-1.0] * INTERNAL_ACTION_DIM,
                "action_q99": [1.0] * INTERNAL_ACTION_DIM,
                "action_stat_valid": action_valid.tolist(),
            }
        self._validate_document_header(stats)
        state_q01 = _strict_vector(
            stats, "state_q01", ABSOLUTE_EEF_POSE_DIM, dtype=torch.float32)
        state_q99 = _strict_vector(
            stats, "state_q99", ABSOLUTE_EEF_POSE_DIM, dtype=torch.float32)
        state_valid = _strict_vector(
            stats, "state_stat_valid", ABSOLUTE_EEF_POSE_DIM, dtype=torch.bool)
        action_q01 = _strict_vector(
            stats, "action_q01", INTERNAL_ACTION_DIM, dtype=torch.float32)
        action_q99 = _strict_vector(
            stats, "action_q99", INTERNAL_ACTION_DIM, dtype=torch.float32)
        action_valid = _strict_vector(
            stats, "action_stat_valid", INTERNAL_ACTION_DIM, dtype=torch.bool)
        if torch.any(state_valid & (state_q99 <= state_q01)):
            raise ValueError("valid state pose stats require q99 > q01")
        if torch.any(action_valid & (action_q99 <= action_q01)):
            raise ValueError("valid internal action stats require q99 > q01")
        unexpected_action = action_valid.clone()
        unexpected_action[list(_EEF_ACTION_INDICES)] = False
        if bool(unexpected_action.any()):
            raise ValueError(
                "absolute EEF training may not enable joint/base Action32 slots")

        canonical = json.dumps(
            dict(stats), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest = bytes.fromhex(hashlib.sha256(canonical).hexdigest())
        self.register_buffer("state_q01", state_q01)
        self.register_buffer("state_q99", state_q99)
        self.register_buffer("state_stat_valid", state_valid)
        self.register_buffer("action_q01", action_q01)
        self.register_buffer("action_q99", action_q99)
        self.register_buffer("action_stat_valid", action_valid)
        self.register_buffer(
            "contract_version", torch.tensor(1, dtype=torch.int64))
        self.register_buffer(
            "stats_sha256", torch.tensor(list(digest), dtype=torch.uint8))

    @staticmethod
    def _validate_document_header(stats: Mapping[str, Any]) -> None:
        expected = {
            "format": POSE_CODEC_STATS_FORMAT,
            "input_contract": ABSOLUTE_EEF_POSE_CONTRACT,
            "internal_action_contract": INTERNAL_ACTION_CONTRACT,
        }
        actual = {key: stats.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"pose codec stats contract mismatch: {actual} != {expected}")

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        expected_dataset_manifest_fingerprint: Optional[str] = None,
        expected_source_adapter_fingerprint: Optional[str] = None,
    ) -> "CanonicalAbsolutePoseCodec":
        path = Path(path)
        if not shared_metadata_exists(path):
            raise FileNotFoundError(f"pose codec stats are missing: {path}")
        value = json.loads(read_shared_text(path))
        if not isinstance(value, Mapping):
            raise TypeError("pose codec stats must be a JSON object")
        expected = {
            "dataset_manifest_fingerprint":
            expected_dataset_manifest_fingerprint,
            "source_adapter_fingerprint":
            expected_source_adapter_fingerprint,
        }
        mismatches = {
            key: (value.get(key), required)
            for key, required in expected.items()
            if required is not None and value.get(key) != required
        }
        if mismatches:
            raise ValueError(
                f"pose codec stats dataset contract mismatch: {mismatches}")
        return cls(value)

    @property
    def stats_fingerprint(self) -> str:
        return bytes(self.stats_sha256.tolist()).hex()

    @staticmethod
    def _validate_shapes(
        current: torch.Tensor,
        current_valid: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_valid: Optional[torch.Tensor] = None,
    ) -> None:
        if current.ndim != 2 or current.shape[-1] != ABSOLUTE_EEF_POSE_DIM:
            raise ValueError("current_eef_pose_abs must have shape (B,16)")
        if current_valid.shape != current.shape or current_valid.dtype != torch.bool:
            raise ValueError(
                "current_eef_pose_abs_valid must be boolean with shape (B,16)")
        if target is not None:
            if target.ndim != 3 or target.shape[0] != current.shape[0] or target.shape[-1] != ABSOLUTE_EEF_POSE_DIM:
                raise ValueError("target_eef_pose_abs must have shape (B,T,16)")
            if target_valid is None or target_valid.shape != target.shape or target_valid.dtype != torch.bool:
                raise ValueError(
                    "target_eef_pose_abs_valid must be boolean with shape (B,T,16)")

    @staticmethod
    def _canonicalize(
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        arms = []
        arm_masks = []
        for offset in _ARM_POSE_OFFSETS:
            arm = value[..., offset:offset + 8].float()
            mask = valid[..., offset:offset + 8].bool()
            pose_valid = mask[..., :7].all(dim=-1, keepdim=True)
            translation = torch.where(
                pose_valid, arm[..., :3], torch.zeros_like(arm[..., :3]))
            quaternion = _normalize_quaternion_wxyz(arm[..., 3:7])
            identity = torch.zeros_like(quaternion)
            identity[..., 0] = 1.0
            quaternion = torch.where(pose_valid, quaternion, identity)
            gripper_valid = mask[..., 7:8]
            gripper = torch.where(
                gripper_valid, arm[..., 7:8], torch.zeros_like(arm[..., 7:8]))
            arms.append(torch.cat((translation, quaternion, gripper), dim=-1))
            arm_masks.append(torch.cat((
                pose_valid.expand(*pose_valid.shape[:-1], 7),
                gripper_valid,
            ), dim=-1))
        return torch.cat(arms, dim=-1), torch.cat(arm_masks, dim=-1)

    @staticmethod
    def _normalize(
        value: torch.Tensor,
        valid: torch.Tensor,
        q01: torch.Tensor,
        q99: torch.Tensor,
        stats_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = valid & stats_valid
        scale = (q99 - q01).clamp_min(_EPS)
        normalized = ((value - q01) / scale * 2.0 - 1.0).clamp(-1.0, 1.0)
        return torch.where(mask, normalized, torch.zeros_like(normalized)), mask

    @staticmethod
    def _denormalize(
        value: torch.Tensor,
        q01: torch.Tensor,
        q99: torch.Tensor,
    ) -> torch.Tensor:
        return (value + 1.0) * 0.5 * (q99 - q01) + q01

    def encode_raw_training_targets(
        self,
        current_eef_pose_abs: torch.Tensor,
        current_eef_pose_abs_valid: torch.Tensor,
        target_eef_pose_abs: torch.Tensor,
        target_eef_pose_abs_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unnormalized relative Action32 targets and absolute state.

        This is the single statistics boundary used by preprocessing and by
        :meth:`encode_training_targets`. It intentionally performs no clipping
        or quantile normalization, so q01/q99 are measured in the exact model
        coordinates later consumed during training and inference.
        """
        self._validate_shapes(
            current_eef_pose_abs,
            current_eef_pose_abs_valid,
            target_eef_pose_abs,
            target_eef_pose_abs_valid,
        )
        current, current_valid = self._canonicalize(
            current_eef_pose_abs, current_eef_pose_abs_valid)
        target, target_valid = self._canonicalize(
            target_eef_pose_abs, target_eef_pose_abs_valid)
        batch, horizon = target.shape[:2]
        raw_action = target.new_zeros(batch, horizon, INTERNAL_ACTION_DIM)
        action_valid = torch.zeros(
            batch, horizon, INTERNAL_ACTION_DIM,
            dtype=torch.bool, device=target.device)

        for pose_offset, action_offset in zip(
                _ARM_POSE_OFFSETS, _ARM_ACTION_OFFSETS):
            current_arm = current[:, None, pose_offset:pose_offset + 8]
            current_mask = current_valid[:, None, pose_offset:pose_offset + 8]
            target_arm = target[..., pose_offset:pose_offset + 8]
            target_mask = target_valid[..., pose_offset:pose_offset + 8]
            pose_valid = current_mask[..., :7].all(
                dim=-1, keepdim=True) & target_mask[..., :7].all(
                    dim=-1, keepdim=True)
            translation = target_arm[..., :3] - current_arm[..., :3]
            rotation = _quaternion_wxyz_to_rotation_vector(
                _quaternion_multiply_wxyz(
                    target_arm[..., 3:7],
                    _quaternion_conjugate_wxyz(current_arm[..., 3:7]),
                ))
            # Deployment masks a gripper prediction when the anchor gripper
            # observation is unavailable.  Training must use the identical
            # condition, otherwise it would optimize targets that can never
            # be emitted through the public inference interface.
            gripper_valid = (
                current_mask[..., 7:8] & target_mask[..., 7:8])
            gripper_command = 1.0 - 2.0 * target_arm[..., 7:8]
            raw_action[..., action_offset:action_offset + 7] = torch.cat((
                translation,
                rotation,
                gripper_command,
            ), dim=-1)
            action_valid[..., action_offset:action_offset + 7] = torch.cat((
                pose_valid.expand(batch, horizon, 6),
                gripper_valid,
            ), dim=-1)

        return raw_action, action_valid, current, current_valid

    def encode_training_targets(
        self,
        current_eef_pose_abs: torch.Tensor,
        current_eef_pose_abs_valid: torch.Tensor,
        target_eef_pose_abs: torch.Tensor,
        target_eef_pose_abs_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return q01/q99-normalized Action32 targets and pose state."""
        raw_action, action_valid, current, current_valid = (
            self.encode_raw_training_targets(
                current_eef_pose_abs,
                current_eef_pose_abs_valid,
                target_eef_pose_abs,
                target_eef_pose_abs_valid,
            )
        )

        normalized_action, action_valid = self._normalize(
            raw_action,
            action_valid,
            self.action_q01,
            self.action_q99,
            self.action_stat_valid,
        )
        normalized_state, state_valid = self._normalize(
            current,
            current_valid,
            self.state_q01,
            self.state_q99,
            self.state_stat_valid,
        )
        return normalized_action, action_valid, normalized_state, state_valid

    def inference_action_valid(
        self,
        current_eef_pose_abs: torch.Tensor,
        current_eef_pose_abs_valid: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Build the exact Action32 mask expected by the flow solver."""
        self._validate_shapes(
            current_eef_pose_abs, current_eef_pose_abs_valid)
        if horizon <= 0:
            raise ValueError("action horizon must be positive")
        _, current_valid = self._canonicalize(
            current_eef_pose_abs, current_eef_pose_abs_valid)
        result = torch.zeros(
            current_valid.shape[0], horizon, INTERNAL_ACTION_DIM,
            dtype=torch.bool, device=current_valid.device)
        for pose_offset, action_offset in zip(
                _ARM_POSE_OFFSETS, _ARM_ACTION_OFFSETS):
            pose_valid = current_valid[..., pose_offset:pose_offset + 7].all(
                dim=-1, keepdim=True)
            gripper_valid = current_valid[..., pose_offset + 7:pose_offset + 8]
            result[..., action_offset:action_offset + 7] = torch.cat((
                pose_valid.expand(-1, 6), gripper_valid), dim=-1)[:, None]
        return result & self.action_stat_valid

    def encode_inference_state(
        self,
        current_eef_pose_abs: torch.Tensor,
        current_eef_pose_abs_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_shapes(
            current_eef_pose_abs, current_eef_pose_abs_valid)
        current, current_valid = self._canonicalize(
            current_eef_pose_abs, current_eef_pose_abs_valid)
        return self._normalize(
            current,
            current_valid,
            self.state_q01,
            self.state_q99,
            self.state_stat_valid,
        )

    def decode_predictions(
        self,
        normalized_action: torch.Tensor,
        current_eef_pose_abs: torch.Tensor,
        current_eef_pose_abs_valid: torch.Tensor,
        action_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode normalized Action32 into absolute WXYZ EEF target chunks."""
        self._validate_shapes(
            current_eef_pose_abs, current_eef_pose_abs_valid)
        if (normalized_action.ndim != 3
                or normalized_action.shape[0] != current_eef_pose_abs.shape[0]
                or normalized_action.shape[-1] != INTERNAL_ACTION_DIM):
            raise ValueError("normalized_action must have shape (B,T,32)")
        if action_valid is None:
            action_valid = self.inference_action_valid(
                current_eef_pose_abs,
                current_eef_pose_abs_valid,
                int(normalized_action.shape[1]),
            )
        elif action_valid.shape != normalized_action.shape or action_valid.dtype != torch.bool:
            raise ValueError("action_valid must be boolean with shape (B,T,32)")
        action_valid = action_valid & self.action_stat_valid
        raw = self._denormalize(
            normalized_action.float(), self.action_q01, self.action_q99)
        current, current_valid = self._canonicalize(
            current_eef_pose_abs, current_eef_pose_abs_valid)
        batch, horizon = normalized_action.shape[:2]
        output = raw.new_zeros(batch, horizon, ABSOLUTE_EEF_POSE_DIM)
        output_valid = torch.zeros_like(output, dtype=torch.bool)
        for pose_offset, action_offset in zip(
                _ARM_POSE_OFFSETS, _ARM_ACTION_OFFSETS):
            current_arm = current[:, None, pose_offset:pose_offset + 8]
            current_mask = current_valid[:, None, pose_offset:pose_offset + 8]
            delta = raw[..., action_offset:action_offset + 7]
            delta_valid = action_valid[..., action_offset:action_offset + 7]
            pose_valid = current_mask[..., :7].all(
                dim=-1, keepdim=True) & delta_valid[..., :6].all(
                    dim=-1, keepdim=True)
            translation = current_arm[..., :3] + delta[..., :3]
            quaternion = _quaternion_multiply_wxyz(
                _rotation_vector_to_quaternion_wxyz(delta[..., 3:6]),
                current_arm[..., 3:7],
            )
            gripper_valid = delta_valid[..., 6:7]
            gripper = ((1.0 - delta[..., 6:7]) * 0.5).clamp(0.0, 1.0)
            output[..., pose_offset:pose_offset + 8] = torch.cat((
                translation, quaternion, gripper), dim=-1)
            output_valid[..., pose_offset:pose_offset + 8] = torch.cat((
                pose_valid.expand(batch, horizon, 7), gripper_valid), dim=-1)
        output = torch.where(output_valid, output, torch.zeros_like(output))
        return output, output_valid
