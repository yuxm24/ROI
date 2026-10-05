"""RoboDojo dual-arm absolute-target to Zerith VA source-action conversion.

The official EE LeRobot release stores both ``observation.state`` and
``action`` as 16-D vectors in this order::

    left  [x, y, z, qw, qx, qy, qz, gripper_opening],
    right [x, y, z, qw, qx, qy, qz, gripper_opening].

Unlike the RoboTwin observation-only demonstrations, every RoboDojo row has
an explicit absolute EE command.  Row ``t`` must therefore be converted from
``state[t]`` to ``action[t]`` without shifting the target by one row.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .robotwin import (
    normalize_quaternion_wxyz,
    opening_to_libero_gripper,
    relative_world_rotation_vector,
)


ROBODOJO_OFFICIAL_EE_DIM = 16
ROBODOJO_SOURCE_ACTION_DIM = 14
ROBODOJO_STATE20_DIM = 20
_EPS = 1e-12


def _validate_absolute_ee(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != ROBODOJO_OFFICIAL_EE_DIM:
        raise ValueError(f"{name} must have shape (T, 16), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    result = array.copy()
    result[:, 3:7] = normalize_quaternion_wxyz(result[:, 3:7])
    result[:, 11:15] = normalize_quaternion_wxyz(result[:, 11:15])
    opening_to_libero_gripper(result[:, 7])
    opening_to_libero_gripper(result[:, 15])
    result[:, 7] = np.clip(result[:, 7], 0.0, 1.0)
    result[:, 15] = np.clip(result[:, 15], 0.0, 1.0)
    return result


def absolute_ee_targets_to_source_actions(
    states: np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert aligned RoboDojo state/target rows to 14-D delta EE actions.

    All source rows are valid because the official dataset records an action
    target for every observation, including the final row.  Any temporal tail
    padding needed by the VAE is added later by preprocessing and receives an
    all-false feature mask there.
    """
    current = _validate_absolute_ee(states, name="RoboDojo states")
    target = _validate_absolute_ee(targets, name="RoboDojo targets")
    if current.shape != target.shape:
        raise ValueError(
            "RoboDojo state/action shapes must match: "
            f"{current.shape} != {target.shape}"
        )

    actions = np.empty(
        (current.shape[0], ROBODOJO_SOURCE_ACTION_DIM),
        dtype=np.float64,
    )
    actions[:, 0:3] = target[:, 0:3] - current[:, 0:3]
    actions[:, 3:6] = relative_world_rotation_vector(
        current[:, 3:7], target[:, 3:7]
    )
    actions[:, 6] = opening_to_libero_gripper(target[:, 7])
    actions[:, 7:10] = target[:, 8:11] - current[:, 8:11]
    actions[:, 10:13] = relative_world_rotation_vector(
        current[:, 11:15], target[:, 11:15]
    )
    actions[:, 13] = opening_to_libero_gripper(target[:, 15])
    valid = np.ones(actions.shape, dtype=np.bool_)
    return actions.astype(np.float32), valid


def absolute_ee_target_to_source_action(
    state: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Single-row convenience wrapper around the batched transform."""
    state_array = np.asarray(state)
    target_array = np.asarray(target)
    if state_array.shape != (ROBODOJO_OFFICIAL_EE_DIM,):
        raise ValueError(
            f"RoboDojo state must have shape (16,), got {state_array.shape}"
        )
    if target_array.shape != (ROBODOJO_OFFICIAL_EE_DIM,):
        raise ValueError(
            f"RoboDojo target must have shape (16,), got {target_array.shape}"
        )
    actions, _valid = absolute_ee_targets_to_source_actions(
        state_array[None], target_array[None]
    )
    return actions[0]


def absolute_ee_to_state20(value: np.ndarray) -> np.ndarray:
    """Convert RoboDojo ``xyz+wxyz+opening`` rows to canonical State20.

    The conversion is absolute and frame preserving: each arm keeps xyz,
    stores the first two *columns* of its rotation matrix as rotation6d, and
    maps the upstream opening convention (0=close, 1=open) to the model
    convention (+1=close, -1=open).
    """
    pose = _validate_absolute_ee(value, name="RoboDojo EE pose")
    result = np.empty((pose.shape[0], ROBODOJO_STATE20_DIM), dtype=np.float32)
    for source_offset, target_offset in ((0, 0), (8, 10)):
        result[:, target_offset:target_offset + 3] = pose[:, source_offset:source_offset + 3]
        quaternion = pose[:, source_offset + 3:source_offset + 7]
        w, x, y, z = quaternion.T
        # Quaternion-to-matrix in the repository's wxyz convention.
        matrix = np.stack((
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ), axis=-1).reshape(-1, 3, 3)
        result[:, target_offset + 3:target_offset + 9] = np.concatenate(
            (matrix[:, :, 0], matrix[:, :, 1]), axis=-1)
        result[:, target_offset + 9] = opening_to_libero_gripper(
            pose[:, source_offset + 7])
    return result


def absolute_ee_row_to_state20(value: np.ndarray) -> np.ndarray:
    """Single-row convenience wrapper for :func:`absolute_ee_to_state20`."""
    row = np.asarray(value)
    if row.shape != (ROBODOJO_OFFICIAL_EE_DIM,):
        raise ValueError(f"RoboDojo EE row must have shape (16,), got {row.shape}")
    return absolute_ee_to_state20(row[None])[0]


def state20_to_absolute_ee(value: np.ndarray) -> np.ndarray:
    """Invert canonical State20 rows to RoboDojo's 16-D EE convention.

    The model predicts absolute State20 targets. Serving reconstructs the
    absolute quaternion/opening target first, then computes a 14-D delta
    against the previous predicted target. Rotation6D is projected using the
    same column layout as :func:`absolute_ee_to_state20`.
    """
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != ROBODOJO_STATE20_DIM:
        raise ValueError(
            f"State20 rows must have shape (T,20), got {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError("State20 rows contain non-finite values")

    def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(max(trace + 1.0, _EPS))
            quat = np.array((
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ))
        elif matrix[0, 0] >= matrix[1, 1] and matrix[0, 0] >= matrix[2, 2]:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], _EPS))
            quat = np.array((
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ))
        elif matrix[1, 1] >= matrix[2, 2]:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], _EPS))
            quat = np.array((
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ))
        else:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], _EPS))
            quat = np.array((
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ))
        norm = float(np.linalg.norm(quat))
        if norm <= _EPS:
            raise ValueError("State20 rotation6D produced a zero quaternion")
        quat /= norm
        return quat if quat[0] >= 0.0 else -quat

    output = np.empty((rows.shape[0], ROBODOJO_OFFICIAL_EE_DIM), dtype=np.float32)
    for source_offset, target_offset in ((0, 0), (10, 8)):
        output[:, target_offset:target_offset + 3] = rows[:, source_offset:source_offset + 3]
        for index, row in enumerate(rows):
            first = row[source_offset + 3:source_offset + 6]
            second = row[source_offset + 6:source_offset + 9]
            first_norm = np.linalg.norm(first)
            if first_norm <= _EPS:
                raise ValueError("State20 rotation6D first column is degenerate")
            column0 = first / first_norm
            orthogonal = second - column0 * float(np.dot(column0, second))
            second_norm = np.linalg.norm(orthogonal)
            if second_norm <= _EPS:
                raise ValueError("State20 rotation6D columns are degenerate")
            column1 = orthogonal / second_norm
            column2 = np.cross(column0, column1)
            matrix = np.stack((column0, column1, column2), axis=1)
            output[index, target_offset + 3:target_offset + 7] = matrix_to_quaternion(matrix)
        # State20 uses -1=open,+1=close; RoboDojo stores 0=closed,1=open.
        output[:, target_offset + 7] = np.clip(
            (1.0 - rows[:, source_offset + 9]) * 0.5, 0.0, 1.0)
    return output


def build_state20_target_windows(
    states20: np.ndarray,
    anchor_indices: np.ndarray,
    *,
    horizon: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create State20 target rows ``t+1..t+horizon`` and aligned masks."""
    values = np.asarray(states20, dtype=np.float32)
    anchors = np.asarray(anchor_indices, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != ROBODOJO_STATE20_DIM:
        raise ValueError(f"states20 must have shape (T,20), got {values.shape}")
    if anchors.ndim != 1 or np.any((anchors < 0) | (anchors >= len(values))):
        raise ValueError("anchor_indices must select rows in states20")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    current = values[anchors].copy()
    target = np.zeros((len(anchors), horizon, ROBODOJO_STATE20_DIM), np.float32)
    current_valid = np.ones_like(current, dtype=np.bool_)
    target_valid = np.zeros_like(target, dtype=np.bool_)
    for row, anchor in enumerate(anchors.tolist()):
        end = min(len(values), int(anchor) + 1 + horizon)
        count = max(0, end - int(anchor) - 1)
        if count:
            target[row, :count] = values[int(anchor) + 1:end]
            target_valid[row, :count] = True
    return current, target, current_valid, target_valid
