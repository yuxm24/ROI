"""State20/Action20 contract shared by training and serving.

The public dataset and inference boundary is a homogeneous dual-arm State20::

    [left xyz(m), rotation6d(6), gripper, right ...]

``rotation6d`` is the first two **columns** of a proper rotation matrix,
flattened column-by-column.  It is continuous for learning and has neither the
``q``/``-q`` ambiguity of quaternions nor the singularities of Euler angles.
Quaternions may be used transiently by a source decoder, but they must be
converted before reaching this module or persistent training storage. Raw
state rows are never interpolated or resampled.

The codec expresses absolute targets relative to the current observation.
Decode applies the inverse transform against the same observation. The
benchmark adapter supplies the command alignment and validity masks.
"""

from __future__ import annotations

import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from zvla.utils.shared_metadata import read_shared_text, shared_metadata_exists

STATE_TRAJECTORY_CONTRACT = "zvla-state20-absolute-rotation6d-eef-state-target-v1"
INTERNAL_STATE_ACTION_CONTRACT = (
    "zvla-action20-anchor-relative-rotation6d-eef-state-target-v1")
STATE_CODEC_STATS_FORMAT = (
    "zvla-action20-state20-per-robot-q01-q99-linear-stats-v1")
MODEL_SIDE_STATE_NORMALIZATION_ID = (
    "model_side_action20_state20_per_robot_q01_q99_clip_v1")
ACTION20_COORDINATE_CONVENTION = (
    "translation_p_future_minus_p_anchor_in_upstream_canonical_frame;"
    "rotation_R_anchor_transpose_matmul_R_future;"
    "rotation6d_first_two_matrix_columns_column_major;"
    "gripper_absolute_upstream_canonical")
STATE_TRAJECTORY_DIM = 20
INTERNAL_STATE_ACTION_DIM = 20

STATE_TRAJECTORY_LAYOUT = (
    "arm0.position.x_m",
    "arm0.position.y_m",
    "arm0.position.z_m",
    "arm0.orientation.rotation6d.column0.x",
    "arm0.orientation.rotation6d.column0.y",
    "arm0.orientation.rotation6d.column0.z",
    "arm0.orientation.rotation6d.column1.x",
    "arm0.orientation.rotation6d.column1.y",
    "arm0.orientation.rotation6d.column1.z",
    "arm0.gripper.position_upstream_canonical",
    "arm1.position.x_m",
    "arm1.position.y_m",
    "arm1.position.z_m",
    "arm1.orientation.rotation6d.column0.x",
    "arm1.orientation.rotation6d.column0.y",
    "arm1.orientation.rotation6d.column0.z",
    "arm1.orientation.rotation6d.column1.x",
    "arm1.orientation.rotation6d.column1.y",
    "arm1.orientation.rotation6d.column1.z",
    "arm1.gripper.position_upstream_canonical",
)

INTERNAL_STATE_ACTION_LAYOUT = (
    "arm0.anchor_relative.translation.x_m",
    "arm0.anchor_relative.translation.y_m",
    "arm0.anchor_relative.translation.z_m",
    "arm0.anchor_relative.rotation6d.column0.x",
    "arm0.anchor_relative.rotation6d.column0.y",
    "arm0.anchor_relative.rotation6d.column0.z",
    "arm0.anchor_relative.rotation6d.column1.x",
    "arm0.anchor_relative.rotation6d.column1.y",
    "arm0.anchor_relative.rotation6d.column1.z",
    "arm0.target_gripper.position_upstream_canonical",
    "arm1.anchor_relative.translation.x_m",
    "arm1.anchor_relative.translation.y_m",
    "arm1.anchor_relative.translation.z_m",
    "arm1.anchor_relative.rotation6d.column0.x",
    "arm1.anchor_relative.rotation6d.column0.y",
    "arm1.anchor_relative.rotation6d.column0.z",
    "arm1.anchor_relative.rotation6d.column1.x",
    "arm1.anchor_relative.rotation6d.column1.y",
    "arm1.anchor_relative.rotation6d.column1.z",
    "arm1.target_gripper.position_upstream_canonical",
)

_ARM_OFFSETS = (0, 10)
_EPS = 1e-8

if len(STATE_TRAJECTORY_LAYOUT) != STATE_TRAJECTORY_DIM:
    raise RuntimeError("State20 layout width drifted")
if len(INTERNAL_STATE_ACTION_LAYOUT) != INTERNAL_STATE_ACTION_DIM:
    raise RuntimeError("private state-action layout width drifted")


def _fp32_robot_boundary(method):
    """Make State20 coordinate math independent of an outer autocast scope."""

    @wraps(method)
    def wrapped(self, first_tensor: torch.Tensor, *args, **kwargs):
        with torch.autocast(device_type=first_tensor.device.type, enabled=False):
            return method(self, first_tensor, *args, **kwargs)

    return wrapped


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Project two candidate columns onto SO(3) with Gram--Schmidt.

    The final dimension is ``[c0.x,c0.y,c0.z,c1.x,c1.y,c1.z]`` and the return
    shape appends ``(3,3)`` with basis vectors stored as matrix columns.
    """
    if value.shape[-1] != 6:
        raise ValueError("rotation6d must have a final width of six")
    candidate0 = value[..., :3]
    candidate1 = value[..., 3:6]
    column0 = candidate0 / torch.linalg.vector_norm(
        candidate0, dim=-1, keepdim=True).clamp_min(_EPS)
    orthogonal1 = candidate1 - (column0 * candidate1).sum(
        dim=-1, keepdim=True) * column0
    column1 = orthogonal1 / torch.linalg.vector_norm(
        orthogonal1, dim=-1, keepdim=True).clamp_min(_EPS)
    column2 = torch.linalg.cross(column0, column1, dim=-1)
    return torch.stack((column0, column1, column2), dim=-1)


def matrix_to_rotation_6d(value: torch.Tensor) -> torch.Tensor:
    """Return the first two columns of ``(...,3,3)`` column-by-column."""
    if value.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must have shape (...,3,3)")
    return torch.cat((value[..., :, 0], value[..., :, 1]), dim=-1)


def _strict_matrix(
    document: Mapping[str, Any],
    key: str,
    *,
    rows: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if key not in document:
        raise KeyError(f"state codec stats are missing {key!r}")
    value = torch.as_tensor(document[key], dtype=dtype)
    if tuple(value.shape) != (rows, STATE_TRAJECTORY_DIM):
        raise ValueError(
            f"state codec stats {key} must have shape "
            f"({rows},{STATE_TRAJECTORY_DIM}), got {tuple(value.shape)}")
    if dtype != torch.bool and not torch.isfinite(value).all():
        raise ValueError(f"state codec stats {key} contains NaN or Inf")
    return value


class CanonicalStateTrajectoryCodec(nn.Module):
    """Encode absolute State20 chunks and decode absolute desired states.

    Translation uses ``p_target - p_anchor`` in the canonical arm-base frame.
    Rotation uses ``R_anchor.T @ R_target`` represented as rotation6D.  The
    upstream-canonical gripper position remains an absolute target state.  The
    same anchor is used for every future row; predictions are never integrated
    autoregressively.  Per-robot q01/q99 statistics map valid features to
    ``[-1,1]`` and clip exactly once at the model boundary.
    """

    def __init__(
        self,
        stats: Optional[Mapping[str, Any]] = None,
        *,
        embodiments: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        expected_embodiments = (
            None if embodiments is None else tuple(embodiments))
        if stats is None:
            selected = expected_embodiments or ("default", )
            lows = [[-1.0] * STATE_TRAJECTORY_DIM for _ in selected]
            highs = [[1.0] * STATE_TRAJECTORY_DIM for _ in selected]
            valid = [[True] * STATE_TRAJECTORY_DIM for _ in selected]
            constant = [[False] * STATE_TRAJECTORY_DIM for _ in selected]
            stats = {
                "format": STATE_CODEC_STATS_FORMAT,
                "input_contract": STATE_TRAJECTORY_CONTRACT,
                "internal_action_contract": INTERNAL_STATE_ACTION_CONTRACT,
                "normalization": "per_robot_q01_q99_linear_clip_minus1_plus1",
                "robot_key": "robot_type",
                "embodiments": list(selected),
                "state_layout": list(STATE_TRAJECTORY_LAYOUT),
                "action_layout": list(INTERNAL_STATE_ACTION_LAYOUT),
                "coordinate_convention": ACTION20_COORDINATE_CONVENTION,
                "state_q01": lows,
                "state_q99": highs,
                "state_stat_valid": valid,
                "state_degenerate": constant,
                "action_q01": lows,
                "action_q99": highs,
                "action_stat_valid": valid,
                "action_degenerate": constant,
            }
        self._validate_document_header(stats)
        raw_embodiments = stats.get("embodiments")
        if (not isinstance(raw_embodiments, list) or not raw_embodiments
                or any(not isinstance(value, str) or not value
                       for value in raw_embodiments)
                or len(set(raw_embodiments)) != len(raw_embodiments)):
            raise ValueError(
                "state codec stats embodiments must be unique non-empty strings")
        selected = tuple(raw_embodiments)
        if (expected_embodiments is not None
                and selected != expected_embodiments):
            raise ValueError(
                "state codec embodiment order differs from the model config: "
                f"{selected} != {expected_embodiments}")
        rows = len(selected)
        state_q01 = _strict_matrix(
            stats, "state_q01", rows=rows, dtype=torch.float32)
        state_q99 = _strict_matrix(
            stats, "state_q99", rows=rows, dtype=torch.float32)
        state_valid = _strict_matrix(
            stats, "state_stat_valid", rows=rows, dtype=torch.bool)
        state_degenerate = _strict_matrix(
            stats, "state_degenerate", rows=rows, dtype=torch.bool)
        action_q01 = _strict_matrix(
            stats, "action_q01", rows=rows, dtype=torch.float32)
        action_q99 = _strict_matrix(
            stats, "action_q99", rows=rows, dtype=torch.float32)
        action_valid = _strict_matrix(
            stats, "action_stat_valid", rows=rows, dtype=torch.bool)
        action_degenerate = _strict_matrix(
            stats, "action_degenerate", rows=rows, dtype=torch.bool)
        if torch.any(state_valid & ~state_degenerate &
                     ((state_q99 - state_q01) < 1e-6)):
            raise ValueError("non-degenerate State20 q01/q99 span is too small")
        if torch.any(action_valid & ~action_degenerate &
                     ((action_q99 - action_q01) < 1e-6)):
            raise ValueError("non-degenerate Action20 q01/q99 span is too small")

        canonical = json.dumps(
            dict(stats),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = bytes.fromhex(hashlib.sha256(canonical).hexdigest())
        self.embodiments = selected
        self.register_buffer("state_q01", state_q01)
        self.register_buffer("state_q99", state_q99)
        self.register_buffer("state_stat_valid", state_valid)
        self.register_buffer("state_degenerate", state_degenerate)
        self.register_buffer("action_q01", action_q01)
        self.register_buffer("action_q99", action_q99)
        self.register_buffer("action_stat_valid", action_valid)
        self.register_buffer("action_degenerate", action_degenerate)
        self.register_buffer("contract_version",
                             torch.tensor(3, dtype=torch.int64))
        self.register_buffer("stats_sha256",
                             torch.tensor(list(digest), dtype=torch.uint8))
        for name, payload in (
                ("state_layout_sha256", STATE_TRAJECTORY_LAYOUT),
                ("action_layout_sha256", INTERNAL_STATE_ACTION_LAYOUT),
                ("coordinate_convention_sha256", ACTION20_COORDINATE_CONVENTION),
        ):
            encoded = json.dumps(payload, separators=(",", ":")).encode()
            self.register_buffer(
                name,
                torch.tensor(list(hashlib.sha256(encoded).digest()),
                             dtype=torch.uint8),
            )

    @staticmethod
    def _validate_document_header(stats: Mapping[str, Any]) -> None:
        expected = {
            "format": STATE_CODEC_STATS_FORMAT,
            "input_contract": STATE_TRAJECTORY_CONTRACT,
            "internal_action_contract": INTERNAL_STATE_ACTION_CONTRACT,
            "normalization": "per_robot_q01_q99_linear_clip_minus1_plus1",
            "robot_key": "robot_type",
        }
        actual = {key: stats.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"state codec stats contract mismatch: {actual} != {expected}")
        semantic_expected = {
            "state_layout": list(STATE_TRAJECTORY_LAYOUT),
            "action_layout": list(INTERNAL_STATE_ACTION_LAYOUT),
            "coordinate_convention": ACTION20_COORDINATE_CONVENTION,
        }
        if {key: stats.get(key) for key in semantic_expected} != semantic_expected:
            raise ValueError(
                "state codec layout/coordinate convention differs from Action20")

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        expected_dataset_manifest_fingerprint: Optional[str] = None,
        expected_embodiments: Optional[Tuple[str, ...]] = None,
    ) -> "CanonicalStateTrajectoryCodec":
        path = Path(path)
        if not shared_metadata_exists(path):
            raise FileNotFoundError(f"state codec stats are missing: {path}")
        value = json.loads(read_shared_text(path))
        if not isinstance(value, Mapping):
            raise TypeError("state codec stats must be a JSON object")
        if (expected_dataset_manifest_fingerprint is not None
                and value.get("dataset_manifest_fingerprint")
                != expected_dataset_manifest_fingerprint):
            raise ValueError(
                "state codec stats dataset manifest fingerprint differs")
        return cls(value, embodiments=expected_embodiments)

    @property
    def stats_fingerprint(self) -> str:
        return bytes(self.stats_sha256.tolist()).hex()

    def encode_state20(
        self,
        absolute_state: torch.Tensor,
        valid_mask: torch.Tensor,
        robot_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Public State20 encoder; ``robot_type`` is the registry row ID."""
        return self.encode_inference_state(
            absolute_state, valid_mask.bool(), robot_type)

    def encode_action20(
        self,
        anchor_state: torch.Tensor,
        future_states: torch.Tensor,
        valid_mask: Any,
        robot_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Public Action20 encoder using one anchor for the complete chunk.

        ``valid_mask`` is the explicit ``(anchor_valid, future_valid)`` pair.
        Keeping both masks together prevents a missing arm from being confused
        with a physically valid zero displacement.
        """
        if (not isinstance(valid_mask, (tuple, list))
                or len(valid_mask) != 2):
            raise TypeError(
                "Action20 valid_mask must be (anchor_valid, future_valid)")
        raw, raw_valid, _, _ = self.encode_raw_training_targets(
            anchor_state,
            valid_mask[0].bool(),
            future_states,
            valid_mask[1].bool(),
        )
        q01, q99, stats_valid, degenerate = self._select_statistics(
            robot_type,
            batch=raw.shape[0],
            q01=self.action_q01,
            q99=self.action_q99,
            valid=self.action_stat_valid,
            degenerate=self.action_degenerate,
            add_time_axis=True,
        )
        encoded, encoded_valid = self._normalize(
            raw, raw_valid, q01, q99, stats_valid, degenerate)
        return encoded.float(), encoded_valid

    def decode_action20(
        self,
        anchor_state: torch.Tensor,
        normalized_action: torch.Tensor,
        robot_type: torch.Tensor,
    ) -> torch.Tensor:
        """Public inverse used by serving; returns absolute State20 targets."""
        anchor_valid = torch.isfinite(anchor_state)
        value, _ = self.decode_predictions(
            normalized_action,
            anchor_state,
            anchor_valid,
            robot_type,
        )
        return value

    def _select_statistics(
        self,
        embodiment_id: torch.Tensor,
        *,
        batch: int,
        q01: torch.Tensor,
        q99: torch.Tensor,
        valid: torch.Tensor,
        degenerate: torch.Tensor,
        add_time_axis: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if embodiment_id.ndim != 1 or tuple(embodiment_id.shape) != (batch, ):
            raise ValueError(f"embodiment_id must have shape ({batch},)")
        if embodiment_id.dtype not in {
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        }:
            raise TypeError("embodiment_id must use an integer dtype")
        indices = embodiment_id.to(device=q01.device, dtype=torch.long)
        # Dataset/preflight validates the closed ID range once. Avoid a pair
        # of device-to-host synchronizations in every training step here;
        # index_select still fails rather than selecting an invalid row.
        selected = (
            q01.index_select(0, indices),
            q99.index_select(0, indices),
            valid.index_select(0, indices),
            degenerate.index_select(0, indices),
        )
        if add_time_axis:
            selected = tuple(value[:, None, :] for value in selected)
        return selected

    @staticmethod
    def _validate_shapes(
        current: torch.Tensor,
        current_valid: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_valid: Optional[torch.Tensor] = None,
    ) -> None:
        if current.ndim != 2 or current.shape[-1] != STATE_TRAJECTORY_DIM:
            raise ValueError("current_state_abs must have shape (B,20)")
        if current_valid.shape != current.shape or current_valid.dtype != torch.bool:
            raise ValueError(
                "current_state_abs_valid must be boolean with shape (B,20)")
        if target is not None:
            if (target.ndim != 3 or target.shape[0] != current.shape[0]
                    or target.shape[-1] != STATE_TRAJECTORY_DIM):
                raise ValueError("target_state_abs must have shape (B,T,20)")
            if (target_valid is None or target_valid.shape != target.shape
                    or target_valid.dtype != torch.bool):
                raise ValueError(
                    "target_state_abs_valid must be boolean with shape (B,T,20)"
                )

    @staticmethod
    def _canonicalize(
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        arms = []
        masks = []
        for offset in _ARM_OFFSETS:
            # State20 is persisted as float32, but projection back onto SO(3)
            # is part of the physical-coordinate codec rather than the model
            # boundary.  Keep that projection (and the values subsequently
            # used to build relative transforms/statistics) in float64.  The
            # public encode/decode methods perform the single float32 cast at
            # their return boundary.
            arm = value[..., offset:offset + 10].double()
            mask = valid[..., offset:offset + 10].bool()
            pose_valid = mask[..., :9].all(dim=-1, keepdim=True)
            translation = torch.where(pose_valid, arm[..., :3],
                                      torch.zeros_like(arm[..., :3]))
            matrix = rotation_6d_to_matrix(arm[..., 3:9])
            identity = torch.eye(
                3, device=matrix.device, dtype=matrix.dtype).expand_as(matrix)
            matrix = torch.where(pose_valid[..., None], matrix, identity)
            rotation = matrix_to_rotation_6d(matrix)
            gripper_valid = mask[..., 9:10]
            gripper = torch.where(gripper_valid, arm[..., 9:10],
                                  torch.zeros_like(arm[..., 9:10]))
            arms.append(torch.cat((translation, rotation, gripper), dim=-1))
            masks.append(
                torch.cat((
                    pose_valid.expand(*pose_valid.shape[:-1], 9),
                    gripper_valid,
                ),
                          dim=-1))
        return torch.cat(arms, dim=-1), torch.cat(masks, dim=-1)

    @staticmethod
    def _normalize(
        value: torch.Tensor,
        valid: torch.Tensor,
        q01: torch.Tensor,
        q99: torch.Tensor,
        stats_valid: torch.Tensor,
        degenerate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = valid & stats_valid
        span = (q99 - q01).clamp_min(1e-6)
        normalized = (2.0 * (value - q01) / span - 1.0).clamp(-1.0, 1.0)
        normalized = torch.where(degenerate, torch.zeros_like(normalized),
                                 normalized)
        return torch.where(mask, normalized,
                           torch.zeros_like(normalized)), mask

    @staticmethod
    def _denormalize(
        value: torch.Tensor,
        q01: torch.Tensor,
        q99: torch.Tensor,
        degenerate: torch.Tensor,
    ) -> torch.Tensor:
        decoded = q01 + (value.clamp(-1.0, 1.0) + 1.0) * 0.5 * (q99 - q01)
        return torch.where(degenerate, q01, decoded)

    def _normalize_action(self, *args):
        # Action scaling may change independently of the State20 condition.
        return self._normalize(*args)

    def _denormalize_action(self, *args):
        return self._denormalize(*args)

    @_fp32_robot_boundary
    def encode_raw_training_targets(
        self,
        current_state_abs: torch.Tensor,
        current_state_abs_valid: torch.Tensor,
        target_state_abs: torch.Tensor,
        target_state_abs_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return relative Action20 and canonical absolute State20."""
        self._validate_shapes(
            current_state_abs,
            current_state_abs_valid,
            target_state_abs,
            target_state_abs_valid,
        )
        current, current_valid = self._canonicalize(current_state_abs,
                                                    current_state_abs_valid)
        target, target_valid = self._canonicalize(target_state_abs,
                                                  target_state_abs_valid)
        batch, horizon = target.shape[:2]
        # Keep every relative transform in float64 through statistics/math;
        # encode_training_targets performs the one model-boundary FP32 cast.
        raw = torch.zeros(
            batch,
            horizon,
            INTERNAL_STATE_ACTION_DIM,
            dtype=torch.float64,
            device=target.device,
        )
        raw_valid = torch.zeros_like(raw, dtype=torch.bool)
        for offset in _ARM_OFFSETS:
            current_arm = current[:, None, offset:offset + 10]
            current_mask = current_valid[:, None, offset:offset + 10]
            target_arm = target[..., offset:offset + 10]
            target_mask = target_valid[..., offset:offset + 10]
            pose_valid = (
                current_mask[..., :9].all(dim=-1, keepdim=True)
                & target_mask[..., :9].all(dim=-1, keepdim=True))
            # Robot-coordinate transforms use float64 even under BF16 autocast;
            # only the cache/model boundary is rounded to float32.
            current_rotation = rotation_6d_to_matrix(
                current_arm[..., 3:9].double())
            target_rotation = rotation_6d_to_matrix(
                target_arm[..., 3:9].double())
            relative_rotation = current_rotation.transpose(-1, -2) @ target_rotation
            gripper_valid = current_mask[..., 9:10] & target_mask[..., 9:10]
            raw[..., offset:offset + 10] = torch.cat((
                target_arm[..., :3].double() - current_arm[..., :3].double(),
                matrix_to_rotation_6d(relative_rotation),
                target_arm[..., 9:10].double(),
            ),
                                                     dim=-1)
            raw_valid[..., offset:offset + 10] = torch.cat((
                pose_valid.expand(batch, horizon, 9),
                gripper_valid,
            ),
                                                           dim=-1)
        raw = torch.where(raw_valid, raw, torch.zeros_like(raw))
        return raw, raw_valid, current, current_valid

    @_fp32_robot_boundary
    def encode_training_targets(
        self,
        current_state_abs: torch.Tensor,
        current_state_abs_valid: torch.Tensor,
        target_state_abs: torch.Tensor,
        target_state_abs_valid: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw, raw_valid, current, current_valid = self.encode_raw_training_targets(
            current_state_abs,
            current_state_abs_valid,
            target_state_abs,
            target_state_abs_valid,
        )
        action_q01, action_q99, action_stats_valid, action_degenerate = self._select_statistics(
            embodiment_id,
            batch=raw.shape[0],
            q01=self.action_q01,
            q99=self.action_q99,
            valid=self.action_stat_valid,
            degenerate=self.action_degenerate,
            add_time_axis=True,
        )
        state_q01, state_q99, state_stats_valid, state_degenerate = self._select_statistics(
            embodiment_id,
            batch=current.shape[0],
            q01=self.state_q01,
            q99=self.state_q99,
            valid=self.state_stat_valid,
            degenerate=self.state_degenerate,
            add_time_axis=False,
        )
        action, action_valid = self._normalize_action(
            raw, raw_valid, action_q01, action_q99, action_stats_valid,
            action_degenerate)
        state, state_valid = self._normalize(
            current, current_valid, state_q01, state_q99, state_stats_valid,
            state_degenerate)
        return action.float(), action_valid, state.float(), state_valid

    @_fp32_robot_boundary
    def encode_inference_state(
        self,
        current_state_abs: torch.Tensor,
        current_state_abs_valid: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_shapes(current_state_abs, current_state_abs_valid)
        current, current_valid = self._canonicalize(current_state_abs,
                                                    current_state_abs_valid)
        q01, q99, stats_valid, degenerate = self._select_statistics(
            embodiment_id,
            batch=current.shape[0],
            q01=self.state_q01,
            q99=self.state_q99,
            valid=self.state_stat_valid,
            degenerate=self.state_degenerate,
            add_time_axis=False,
        )
        value, value_valid = self._normalize(
            current, current_valid, q01, q99, stats_valid, degenerate)
        return value.float(), value_valid

    @_fp32_robot_boundary
    def inference_action_valid(
        self,
        current_state_abs: torch.Tensor,
        current_state_abs_valid: torch.Tensor,
        embodiment_id: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        self._validate_shapes(current_state_abs, current_state_abs_valid)
        if horizon <= 0:
            raise ValueError("state target horizon must be positive")
        _, current_valid = self._canonicalize(current_state_abs,
                                              current_state_abs_valid)
        result = torch.zeros(
            current_valid.shape[0],
            horizon,
            INTERNAL_STATE_ACTION_DIM,
            dtype=torch.bool,
            device=current_valid.device)
        for offset in _ARM_OFFSETS:
            pose_valid = current_valid[..., offset:offset + 9].all(
                dim=-1, keepdim=True)
            gripper_valid = current_valid[..., offset + 9:offset + 10]
            result[...,
                   offset:offset + 10] = torch.cat(
                       (pose_valid.expand(-1, 9), gripper_valid),
                       dim=-1)[:, None]
        _, _, stats_valid, _ = self._select_statistics(
            embodiment_id,
            batch=current_valid.shape[0],
            q01=self.action_q01,
            q99=self.action_q99,
            valid=self.action_stat_valid,
            degenerate=self.action_degenerate,
            add_time_axis=True,
        )
        return result & stats_valid

    @_fp32_robot_boundary
    def decode_predictions(
        self,
        normalized_action: torch.Tensor,
        current_state_abs: torch.Tensor,
        current_state_abs_valid: torch.Tensor,
        embodiment_id: torch.Tensor,
        action_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode private predictions into absolute desired State20 chunks."""
        self._validate_shapes(current_state_abs, current_state_abs_valid)
        if (normalized_action.ndim != 3
                or normalized_action.shape[0] != current_state_abs.shape[0]
                or normalized_action.shape[-1] != INTERNAL_STATE_ACTION_DIM):
            raise ValueError("normalized_action must have shape (B,T,20)")
        if action_valid is None:
            action_valid = self.inference_action_valid(
                current_state_abs,
                current_state_abs_valid,
                embodiment_id,
                int(normalized_action.shape[1]),
            )
        elif (action_valid.shape != normalized_action.shape
              or action_valid.dtype != torch.bool):
            raise ValueError(
                "action_valid must be boolean with shape (B,T,20)")
        q01, q99, stats_valid, degenerate = self._select_statistics(
            embodiment_id,
            batch=normalized_action.shape[0],
            q01=self.action_q01,
            q99=self.action_q99,
            valid=self.action_stat_valid,
            degenerate=self.action_degenerate,
            add_time_axis=True,
        )
        action_valid = action_valid & stats_valid
        raw = self._denormalize_action(normalized_action.float(), q01, q99,
                                degenerate)
        current, current_valid = self._canonicalize(current_state_abs,
                                                    current_state_abs_valid)
        batch, horizon = normalized_action.shape[:2]
        output = raw.new_zeros(batch, horizon, STATE_TRAJECTORY_DIM)
        output_valid = torch.zeros_like(output, dtype=torch.bool)
        for offset in _ARM_OFFSETS:
            current_arm = current[:, None, offset:offset + 10]
            current_mask = current_valid[:, None, offset:offset + 10]
            delta = raw[..., offset:offset + 10]
            delta_valid = action_valid[..., offset:offset + 10]
            pose_valid = (
                current_mask[..., :9].all(dim=-1, keepdim=True)
                & delta_valid[..., :9].all(dim=-1, keepdim=True))
            absolute_rotation = (
                rotation_6d_to_matrix(current_arm[..., 3:9].double()) @
                rotation_6d_to_matrix(delta[..., 3:9].double()))
            gripper_valid = current_mask[..., 9:10] & delta_valid[..., 9:10]
            output[..., offset:offset + 10] = torch.cat((
                current_arm[..., :3] + delta[..., :3],
                matrix_to_rotation_6d(absolute_rotation).float(),
                delta[..., 9:10],
            ),
                                                        dim=-1)
            output_valid[..., offset:offset + 10] = torch.cat((
                pose_valid.expand(batch, horizon, 9),
                gripper_valid,
            ),
                                                              dim=-1)
        output = torch.where(output_valid, output, torch.zeros_like(output))
        return output, output_valid


__all__ = [
    "ACTION20_COORDINATE_CONVENTION",
    "Action20State20Codec",
    "CanonicalStateTrajectoryCodec",
    "INTERNAL_STATE_ACTION_CONTRACT",
    "INTERNAL_STATE_ACTION_DIM",
    "INTERNAL_STATE_ACTION_LAYOUT",
    "MODEL_SIDE_STATE_NORMALIZATION_ID",
    "STATE_CODEC_STATS_FORMAT",
    "STATE_TRAJECTORY_CONTRACT",
    "STATE_TRAJECTORY_DIM",
    "STATE_TRAJECTORY_LAYOUT",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
]

# The descriptive v5.2 name and the historical import resolve to one exact
# implementation; there is no second codec path whose math can drift.
Action20State20Codec = CanonicalStateTrajectoryCodec
