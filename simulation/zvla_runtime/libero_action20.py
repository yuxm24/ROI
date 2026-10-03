"""Loss-minimal LIBERO OSC-pose <-> State20 command conversion.

This module deliberately depends only on NumPy. The offline asset author,
the training-side compatibility export, and the LIBERO rollout client all use
this one implementation, while the simulator environment remains independent
from the much larger Z-VLA training import graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


LIBERO_STATE20_TARGET_SEMANTICS = (
    "libero_expert_action_at_each_native_row_transcoded_to_absolute_state20_command"
)

# robosuite 1.4 PandaGripper XML declares finger_joint1 in [0, 0.04] m
# and finger_joint2 in [-0.04, 0] m. Their signed difference is therefore the
# physical 0--0.08 m aperture.
PANDA_GRIPPER_MAX_APERTURE_M = 0.08


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(value, dtype=np.float64)
    return np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64
    )


def rotation_vector_to_matrix(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("rotation vector must be finite shape (3,)")
    theta = float(np.linalg.norm(vector))
    generator = _skew(vector)
    if theta < 1e-8:
        return (
            np.eye(3, dtype=np.float64)
            + generator
            + 0.5 * generator @ generator
        )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) / theta * generator
        + (1.0 - np.cos(theta)) / (theta * theta) * generator @ generator
    )


def matrix_to_rotation_vector(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix must be finite shape (3,3)")
    u, _, vt = np.linalg.svd(matrix)
    matrix = u @ vt
    if np.linalg.det(matrix) < 0.0:
        u[:, -1] *= -1.0
        matrix = u @ vt
    cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cosine))
    vee = np.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ),
        dtype=np.float64,
    )
    if theta < 1e-8:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0.0:
            axis = -axis
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * vee


def matrix_to_rotation6d(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must have shape (3,3)")
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def rotation6d_to_matrix(value: Sequence[float]) -> np.ndarray:
    candidate = np.asarray(value, dtype=np.float64)
    if candidate.shape != (6,) or not np.isfinite(candidate).all():
        raise ValueError("rotation6d must be finite shape (6,)")
    first = candidate[:3].copy()
    first /= max(np.linalg.norm(first), 1e-12)
    second = candidate[3:] - first * np.dot(first, candidate[3:])
    second /= max(np.linalg.norm(second), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=1)


def pack_single_arm_state20(
    position: Sequence[float],
    rotation: np.ndarray,
    gripper: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros(20, dtype=np.float32)
    valid = np.zeros(20, dtype=np.bool_)
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("EEF position must be finite shape (3,)")
    if not np.isfinite(gripper):
        raise ValueError("gripper must be finite")
    result[:3] = position
    result[3:9] = matrix_to_rotation6d(rotation)
    result[9] = float(gripper)
    valid[:10] = True
    return result, valid


@dataclass(frozen=True)
class LiberoAction20Adapter:
    """Exact LIBERO convention used by conversion and simulator serving."""

    position_scale_m: float = 0.05
    rotation_scale_rad: float = 0.5

    def observation_state20(
        self, observation_state: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        observation = np.asarray(observation_state, dtype=np.float64)
        if observation.shape != (8,) or not np.isfinite(observation).all():
            raise ValueError("LIBERO observation state must be finite shape (8,)")
        opening = float(
            np.clip(
                (observation[6] - observation[7])
                / PANDA_GRIPPER_MAX_APERTURE_M,
                0.0,
                1.0,
            )
        )
        return pack_single_arm_state20(
            observation[:3],
            rotation_vector_to_matrix(observation[3:6]),
            1.0 - 2.0 * opening,
        )

    def command_state20(
        self,
        observation_state: Sequence[float],
        action: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        observation = np.asarray(observation_state, dtype=np.float64)
        expert = np.asarray(action, dtype=np.float64)
        if (
            observation.shape != (8,)
            or expert.shape != (7,)
            or not np.isfinite(observation).all()
            or not np.isfinite(expert).all()
        ):
            raise ValueError(
                "LIBERO command source must be finite 8D state/7D action"
            )
        current_rotation = rotation_vector_to_matrix(observation[3:6])
        command_rotation = (
            rotation_vector_to_matrix(self.rotation_scale_rad * expert[3:6])
            @ current_rotation
        )
        return pack_single_arm_state20(
            observation[:3] + self.position_scale_m * expert[:3],
            command_rotation,
            float(expert[6]),
        )

    def demonstration_chunk(
        self,
        observation_states: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return an anchor State20 and same-row absolute command targets."""
        observations = np.asarray(observation_states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        if (
            observations.ndim != 2
            or observations.shape[1] != 8
            or actions.ndim != 2
            or actions.shape[1] != 7
            or len(observations) != len(actions)
            or not len(actions)
        ):
            raise ValueError("LIBERO chunk must contain aligned (H,8)/(H,7) rows")
        anchor, anchor_valid = self.observation_state20(observations[0])
        packed = [
            self.command_state20(state, action)
            for state, action in zip(observations, actions, strict=True)
        ]
        return (
            anchor,
            anchor_valid,
            np.stack([value for value, _ in packed]).astype(np.float32),
            np.stack([mask for _, mask in packed]).astype(np.bool_),
        )

    def absolute_target_to_simulator_action(
        self,
        target_state20: Sequence[float],
        target_valid: Sequence[bool],
        observation_state: Sequence[float],
    ) -> np.ndarray:
        target = np.asarray(target_state20, dtype=np.float64)
        valid = np.asarray(target_valid, dtype=np.bool_)
        current = np.asarray(observation_state, dtype=np.float64)
        if (
            target.shape != (20,)
            or valid.shape != (20,)
            or current.shape != (8,)
            or not valid[:10].all()
        ):
            raise ValueError(
                "LIBERO execution requires a complete first-arm State20"
            )
        current_rotation = rotation_vector_to_matrix(current[3:6])
        target_rotation = rotation6d_to_matrix(target[3:9])
        result = np.empty(7, dtype=np.float64)
        result[:3] = (target[:3] - current[:3]) / self.position_scale_m
        result[3:6] = (
            matrix_to_rotation_vector(target_rotation @ current_rotation.T)
            / self.rotation_scale_rad
        )
        result[6] = target[9]
        # The only clipping operation belongs at the simulator boundary.
        return np.clip(result, -1.0, 1.0).astype(np.float32)


__all__ = [
    "LIBERO_STATE20_TARGET_SEMANTICS",
    "LiberoAction20Adapter",
    "PANDA_GRIPPER_MAX_APERTURE_M",
    "matrix_to_rotation6d",
    "matrix_to_rotation_vector",
    "pack_single_arm_state20",
    "rotation6d_to_matrix",
    "rotation_vector_to_matrix",
]
