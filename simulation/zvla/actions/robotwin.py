"""RoboTwin dual-arm action conversion shared by training and evaluation.

RoboTwin records absolute end-effector poses in world coordinates as
``xyz + wxyz`` and gripper opening targets in ``[0, 1]``. Zerith VA trains on
the same semantic action representation used by LIBERO: world-frame
translation deltas, world-frame left-composed rotation vectors, and a
gripper command where ``-1`` is open and ``+1`` is closed.

The source-space RoboTwin action order is::

    left  [dx, dy, dz, rx, ry, rz, gripper],
    right [dx, dy, dz, rx, ry, rz, gripper]

The legacy path below uses dual-arm end-effector deltas.  The native-qpos
path keeps the official 14-D joint target exactly as recorded by RoboTwin::

    left  [joint_0, ..., joint_5, gripper_opening],
    right [joint_0, ..., joint_5, gripper_opening]

For both representations row ``t`` is the command executed from observation
``t``.  Native qpos therefore uses the official causal target ``q[t + 1]``;
the final observation has no target and is feature-masked.  This module is
simulator-independent so preprocessing and deployment call the same
versioned transforms.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


ROBOTWIN_POSE_DIM = 7
ROBOTWIN_SOURCE_ACTION_DIM = 14
ROBOTWIN_QPOS_DIM = 14
ROBOTWIN_OFFICIAL_EE_ACTION_DIM = 16

_EPS = 1e-12


def _as_finite_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Normalize one or more WXYZ quaternions."""
    quaternion = _as_finite_array(quaternion, name="quaternion")
    if quaternion.shape[-1:] != (4,):
        raise ValueError(
            "quaternion last dimension must be 4 (wxyz), got "
            f"{quaternion.shape}"
        )
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= _EPS):
        raise ValueError("quaternion contains a zero-norm value")
    return quaternion / norm


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for broadcastable WXYZ quaternions."""
    left = normalize_quaternion_wxyz(left)
    right = normalize_quaternion_wxyz(right)
    left, right = np.broadcast_arrays(left, right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    result = np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )
    return normalize_quaternion_wxyz(result)


def quaternion_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion_wxyz(quaternion)
    result = quaternion.copy()
    result[..., 1:] *= -1.0
    return result


def rotation_vector_to_quaternion_wxyz(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors to shortest-path WXYZ quaternions."""
    vector = _as_finite_array(rotation_vector, name="rotation vector")
    if vector.shape[-1:] != (3,):
        raise ValueError(
            f"rotation vector last dimension must be 3, got {vector.shape}"
        )
    angle = np.linalg.norm(vector, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(
        angle <= 1e-8,
        0.5 - angle * angle / 48.0,
        np.sin(half) / np.maximum(angle, _EPS),
    )
    quaternion = np.concatenate((np.cos(half), vector * scale), axis=-1)
    return normalize_quaternion_wxyz(quaternion)


def quaternion_wxyz_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    """Convert WXYZ quaternions to rotation vectors with norm in ``[0, pi]``."""
    quaternion = normalize_quaternion_wxyz(quaternion)
    quaternion = np.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    xyz = quaternion[..., 1:]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(
        xyz_norm,
        np.clip(quaternion[..., :1], 0.0, 1.0),
    )
    scale = np.where(
        xyz_norm <= 1e-8,
        2.0,
        angle / np.maximum(xyz_norm, _EPS),
    )
    return xyz * scale


def relative_world_rotation_vector(
    current_wxyz: np.ndarray,
    target_wxyz: np.ndarray,
) -> np.ndarray:
    """Return ``r`` such that ``R_target = Exp(r) @ R_current``."""
    delta = quaternion_multiply_wxyz(
        target_wxyz,
        quaternion_conjugate_wxyz(current_wxyz),
    )
    return quaternion_wxyz_to_rotation_vector(delta)


def apply_world_rotation_vector(
    current_wxyz: np.ndarray,
    rotation_vector: np.ndarray,
) -> np.ndarray:
    """Apply a world-frame, left-composed rotation delta."""
    delta = rotation_vector_to_quaternion_wxyz(rotation_vector)
    return quaternion_multiply_wxyz(delta, current_wxyz)


def opening_to_libero_gripper(opening: np.ndarray) -> np.ndarray:
    """Map RoboTwin ``0=closed, 1=open`` to ``-1=open, +1=closed``."""
    opening = _as_finite_array(opening, name="gripper opening")
    if np.any((opening < -1e-6) | (opening > 1.0 + 1e-6)):
        raise ValueError("RoboTwin gripper opening must be in [0, 1]")
    return 1.0 - 2.0 * np.clip(opening, 0.0, 1.0)


def libero_gripper_to_opening(command: np.ndarray) -> np.ndarray:
    """Invert :func:`opening_to_libero_gripper`, clipping model overshoot."""
    command = _as_finite_array(command, name="gripper command")
    return np.clip((1.0 - command) * 0.5, 0.0, 1.0)


def _validate_pose_trajectory(value: np.ndarray, *, name: str) -> np.ndarray:
    pose = _as_finite_array(value, name=name)
    if pose.ndim != 2 or pose.shape[1] != ROBOTWIN_POSE_DIM:
        raise ValueError(f"{name} must have shape (T, 7), got {pose.shape}")
    pose = pose.copy()
    pose[:, 3:] = normalize_quaternion_wxyz(pose[:, 3:])
    return pose


def _validate_gripper_trajectory(
    value: np.ndarray,
    *,
    length: int,
    name: str,
) -> np.ndarray:
    gripper = _as_finite_array(value, name=name)
    if gripper.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {gripper.shape}")
    opening_to_libero_gripper(gripper)
    return np.clip(gripper, 0.0, 1.0)


def validate_qpos_trajectory(value: np.ndarray) -> np.ndarray:
    """Validate the official dual-arm qpos order and gripper convention.

    RoboTwin stores six arm joints plus one normalized opening target for
    each arm.  The opening dimensions are indices 6 and 13, where 0 is fully
    closed and 1 is fully open.
    """
    qpos = _as_finite_array(value, name="qpos trajectory")
    if qpos.ndim != 2 or qpos.shape[1] != ROBOTWIN_QPOS_DIM:
        raise ValueError(
            "qpos trajectory must have shape (T, 14), got "
            f"{qpos.shape}"
        )
    if qpos.shape[0] == 0:
        raise ValueError("RoboTwin qpos trajectory must contain an observation")
    for index, name in ((6, "left"), (13, "right")):
        opening = qpos[:, index]
        if np.any((opening < -1e-6) | (opening > 1.0 + 1e-6)):
            raise ValueError(
                f"RoboTwin {name} gripper qpos must be in [0, 1]"
            )
    result = qpos.copy()
    result[:, [6, 13]] = np.clip(result[:, [6, 13]], 0.0, 1.0)
    return result.astype(np.float32)


def qpos_trajectory_to_source_actions(
    qpos_trajectory: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build causal native-qpos targets and a strict terminal feature mask.

    The observation at row ``t`` is paired with the official absolute qpos
    target at row ``t + 1``.  The terminal row is exact zero with an
    all-false mask instead of duplicating the last command.
    """
    qpos = validate_qpos_trajectory(qpos_trajectory)
    actions = np.zeros_like(qpos, dtype=np.float32)
    valid = np.zeros_like(qpos, dtype=np.bool_)
    if qpos.shape[0] > 1:
        actions[:-1] = qpos[1:]
        valid[:-1] = True
    return actions, valid


def source_qpos_action_to_official_qpos(
    source_action: np.ndarray,
) -> np.ndarray:
    """Validate one predicted qpos target for ``take_action(..., 'qpos')``.

    Arm joints are already in the official absolute target space.  Only the
    two normalized gripper openings are clipped to their physical contract;
    action normalization supplies the learned joint-range bound.
    """
    action = _as_finite_array(source_action, name="source qpos action")
    if action.shape != (ROBOTWIN_QPOS_DIM,):
        raise ValueError(
            f"source qpos action must have shape (14,), got {action.shape}"
        )
    result = action.astype(np.float32, copy=True)
    result[[6, 13]] = np.clip(result[[6, 13]], 0.0, 1.0)
    return result


def absolute_trajectory_to_source_actions(
    left_pose_wxyz: np.ndarray,
    left_opening: np.ndarray,
    right_pose_wxyz: np.ndarray,
    right_opening: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert absolute observations into causal RoboTwin source actions.

    Row ``t`` moves observation ``t`` to observation ``t+1``. The final row
    has no causal target and is exact zero with an all-false feature mask.
    """
    left_pose = _validate_pose_trajectory(left_pose_wxyz, name="left pose")
    right_pose = _validate_pose_trajectory(right_pose_wxyz, name="right pose")
    if left_pose.shape[0] != right_pose.shape[0]:
        raise ValueError("left and right pose trajectories must have equal length")
    length = int(left_pose.shape[0])
    if length == 0:
        raise ValueError("RoboTwin trajectory must contain at least one observation")
    left_gripper = _validate_gripper_trajectory(
        left_opening,
        length=length,
        name="left gripper opening",
    )
    right_gripper = _validate_gripper_trajectory(
        right_opening,
        length=length,
        name="right gripper opening",
    )

    actions = np.zeros((length, ROBOTWIN_SOURCE_ACTION_DIM), dtype=np.float64)
    valid = np.zeros_like(actions, dtype=np.bool_)
    if length > 1:
        actions[:-1, 0:3] = left_pose[1:, :3] - left_pose[:-1, :3]
        actions[:-1, 3:6] = relative_world_rotation_vector(
            left_pose[:-1, 3:],
            left_pose[1:, 3:],
        )
        actions[:-1, 6] = opening_to_libero_gripper(left_gripper[1:])
        actions[:-1, 7:10] = right_pose[1:, :3] - right_pose[:-1, :3]
        actions[:-1, 10:13] = relative_world_rotation_vector(
            right_pose[:-1, 3:],
            right_pose[1:, 3:],
        )
        actions[:-1, 13] = opening_to_libero_gripper(right_gripper[1:])
        valid[:-1] = True
    return actions.astype(np.float32), valid


def source_action_to_official_ee_action(
    source_action: np.ndarray,
    current_left_pose_wxyz: np.ndarray,
    current_right_pose_wxyz: np.ndarray,
) -> np.ndarray:
    """Convert one predicted 14-D delta to RoboTwin's official 16-D EE target."""
    action = _as_finite_array(source_action, name="source action")
    if action.shape != (ROBOTWIN_SOURCE_ACTION_DIM,):
        raise ValueError(f"source action must have shape (14,), got {action.shape}")
    left = _as_finite_array(current_left_pose_wxyz, name="current left pose")
    right = _as_finite_array(current_right_pose_wxyz, name="current right pose")
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError("current left/right poses must each have shape (7,)")
    left = left.copy()
    right = right.copy()
    left[3:] = normalize_quaternion_wxyz(left[3:])
    right[3:] = normalize_quaternion_wxyz(right[3:])

    left_target = np.empty(7, dtype=np.float64)
    left_target[:3] = left[:3] + action[:3]
    left_target[3:] = apply_world_rotation_vector(left[3:], action[3:6])
    right_target = np.empty(7, dtype=np.float64)
    right_target[:3] = right[:3] + action[7:10]
    right_target[3:] = apply_world_rotation_vector(right[3:], action[10:13])

    official = np.concatenate(
        (
            left_target,
            np.atleast_1d(libero_gripper_to_opening(action[6])),
            right_target,
            np.atleast_1d(libero_gripper_to_opening(action[13])),
        )
    )
    if official.shape != (ROBOTWIN_OFFICIAL_EE_ACTION_DIM,):
        raise AssertionError(f"unexpected official EE action shape {official.shape}")
    return official.astype(np.float32)


def reconstruct_absolute_trajectory(
    initial_left_pose_wxyz: np.ndarray,
    initial_left_opening: float,
    initial_right_pose_wxyz: np.ndarray,
    initial_right_opening: float,
    source_actions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate source actions for offline contract verification."""
    actions = _as_finite_array(source_actions, name="source actions")
    if actions.ndim != 2 or actions.shape[1] != ROBOTWIN_SOURCE_ACTION_DIM:
        raise ValueError(f"source actions must have shape (N, 14), got {actions.shape}")

    left_poses = [np.asarray(initial_left_pose_wxyz, dtype=np.float64)]
    right_poses = [np.asarray(initial_right_pose_wxyz, dtype=np.float64)]
    left_openings = [float(initial_left_opening)]
    right_openings = [float(initial_right_opening)]
    for action in actions:
        official = source_action_to_official_ee_action(
            action,
            left_poses[-1],
            right_poses[-1],
        ).astype(np.float64)
        left_poses.append(official[:7])
        left_openings.append(float(official[7]))
        right_poses.append(official[8:15])
        right_openings.append(float(official[15]))
    return (
        np.stack(left_poses),
        np.asarray(left_openings),
        np.stack(right_poses),
        np.asarray(right_openings),
    )
