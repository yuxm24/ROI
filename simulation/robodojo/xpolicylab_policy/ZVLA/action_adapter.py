"""RoboDojo observation and action conversion for Z-VLA."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_CAMERA_NAMES = (
    "observation/head_camera/rgb",
    "observation/left_camera/rgb",
    "observation/right_camera/rgb",
)
ROBODOJO_CAMERA_KEYS = (
    "cam_head",
    "cam_left_wrist",
    "cam_right_wrist",
)


def _load_shared_action_adapter():
    default_root = Path(__file__).resolve().parents[3]
    repo_root = Path(os.environ.get("ZVLA_REPO_ROOT", default_root)).resolve()
    path = repo_root / "zvla" / "actions" / "robotwin.py"
    if not path.is_file():
        raise FileNotFoundError(f"Z-VLA dual-arm action adapter not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "_zvla_robodojo_shared_action_adapter",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Z-VLA action adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SHARED = _load_shared_action_adapter()


def _finite_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _gripper_opening(value: Any, *, name: str) -> float:
    array = _finite_vector(value, name=name, length=1)
    opening = float(array[0])
    if not -1e-6 <= opening <= 1.0 + 1e-6:
        raise ValueError(f"{name} must be normalized to [0, 1], got {opening}")
    return float(np.clip(opening, 0.0, 1.0))


def extract_pose_state(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = observation.get("state")
    if not isinstance(state, dict):
        raise ValueError("RoboDojo observation has no state mapping")
    left_pose = _finite_vector(
        state.get("left_ee_pose"), name="left_ee_pose", length=7
    )
    right_pose = _finite_vector(
        state.get("right_ee_pose"), name="right_ee_pose", length=7
    )
    left_opening = _gripper_opening(
        state.get("left_ee_joint_state"), name="left_ee_joint_state"
    )
    right_opening = _gripper_opening(
        state.get("right_ee_joint_state"), name="right_ee_joint_state"
    )
    packed = np.concatenate(
        (
            left_pose,
            np.asarray([left_opening], dtype=np.float32),
            right_pose,
            np.asarray([right_opening], dtype=np.float32),
        )
    )
    return left_pose, right_pose, packed


def extract_prompt(observation: dict[str, Any]) -> str:
    for key in ("instruction", "prompt", "task", "language_instruction"):
        value = observation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    raise ValueError("RoboDojo observation has no non-empty instruction")


def extract_camera_frame(observation: dict[str, Any], camera_key: str) -> np.ndarray:
    vision = observation.get("vision")
    if not isinstance(vision, dict) or camera_key not in vision:
        raise ValueError(f"RoboDojo observation is missing vision/{camera_key}")
    camera = vision[camera_key]
    if not isinstance(camera, dict) or "color" not in camera:
        raise ValueError(f"RoboDojo observation is missing vision/{camera_key}/color")
    image = np.asarray(camera["color"])
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(
            f"vision/{camera_key}/color must be HWC RGB(A), got {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"vision/{camera_key}/color must be uint8, got {image.dtype}"
        )
    return np.ascontiguousarray(image[..., :3])


def pack_camera_sequence(observations: list[dict[str, Any]]) -> np.ndarray:
    if not observations:
        raise ValueError("camera sequence must contain at least one observation")
    cameras = np.stack(
        [
            np.stack(
                [extract_camera_frame(observation, key) for key in ROBODOJO_CAMERA_KEYS],
                axis=0,
            )
            for observation in observations
        ],
        axis=0,
    )
    return np.ascontiguousarray(cameras, dtype=np.uint8)


def source_action_to_robodojo(
    source_action: Any,
    observation: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Convert a Z-VLA 14-D delta into one official RoboDojo EE action."""
    action = _finite_vector(source_action, name="source_action", length=14)
    left_pose, right_pose, _ = extract_pose_state(observation)
    target = _SHARED.source_action_to_official_ee_action(
        action,
        left_pose,
        right_pose,
    )
    return {
        "left_ee_pose": target[:7],
        "left_ee_joint_state": target[7:8],
        "right_ee_pose": target[8:15],
        "right_ee_joint_state": target[15:16],
    }


def robodojo_action_to_vector(action: dict[str, Any]) -> np.ndarray:
    """Pack one official RoboDojo EE action into its canonical 16-D order."""
    if not isinstance(action, dict):
        raise TypeError("RoboDojo action must be a mapping")
    left_pose = _finite_vector(
        action.get("left_ee_pose"), name="left_ee_pose target", length=7
    )
    right_pose = _finite_vector(
        action.get("right_ee_pose"), name="right_ee_pose target", length=7
    )
    left_opening = _gripper_opening(
        action.get("left_ee_joint_state"), name="left_ee_joint_state target"
    )
    right_opening = _gripper_opening(
        action.get("right_ee_joint_state"), name="right_ee_joint_state target"
    )
    return np.ascontiguousarray(
        np.concatenate(
            (
                left_pose,
                np.asarray([left_opening], dtype=np.float32),
                right_pose,
                np.asarray([right_opening], dtype=np.float32),
            )
        ),
        dtype=np.float32,
    )


def robodojo_vector_to_action(target: Any) -> dict[str, np.ndarray]:
    """Unpack one absolute RoboDojo 16-D EE target.

    State20 inference predicts absolute targets.  Keeping that target beside
    the legacy 14-D source-delta response lets the simulator apply each row
    against the *actual* pose observed immediately before that row.  This is
    important when the controller has a small tracking error: integrating
    deltas between predicted rows would otherwise compound that error.
    """
    vector = _finite_vector(target, name="absolute EE target", length=16)
    return {
        "left_ee_pose": vector[:7],
        "left_ee_joint_state": vector[7:8],
        "right_ee_pose": vector[8:15],
        "right_ee_joint_state": vector[15:16],
    }
