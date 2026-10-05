"""Original native HighLevel IK in the client; SDK remains in joint mode 3."""

import importlib
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .mapping import control19, decode_command, encode_state
from .native_ik import BodyYawArmIK


def load_planner(config):
    # Explicit trusted external provider; never fetched or executed from the network.
    root = Path(config["planner_source"]).expanduser().resolve(strict=True)
    sys.path.insert(0, str(root))
    parts = config["planner_factory"].split(":")
    if (
        len(parts) != 2
        or not all(p.isidentifier() for p in parts[0].split("."))
        or not parts[1].isidentifier()
    ):
        raise ValueError(
            "planner_factory must use module:function syntax; see the Zerith VA README, public SDK and IK"
        )
    module, function = parts
    planner = getattr(importlib.import_module(module), function)()
    validate_planner(planner)
    return planner


def validate_planner(planner):
    """Reject inconsistent joint/frame metadata before constructing the solver."""
    for attribute in (
        "lower",
        "upper",
        "arm",
        "arm_control_indices",
        "root_from_body",
        "body_eef_poses",
        "eef_names",
    ):
        if not hasattr(planner, attribute):
            raise ValueError(
                f"Planner is missing {attribute}; see the Zerith VA README, public SDK and IK"
            )
    if np.asarray(planner.lower).shape != (19,) or np.asarray(planner.upper).shape != (19,):
        raise ValueError("Planner limits must be control19")
    if not np.isfinite(np.r_[planner.lower, planner.upper]).all() or np.any(
        planner.lower >= planner.upper
    ):
        raise ValueError("Invalid planner limits")
    indices = np.asarray(planner.arm_control_indices)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or len(set(indices.tolist())) != len(indices)
        or not set(range(3, 17)) <= set(indices.tolist()) <= set(range(3, 19))
    ):
        raise ValueError(
            "IK indices must include both arms exactly once, optional head, and no waist"
        )
    transform = np.asarray(planner.root_from_body, dtype=float)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("root_from_body must be a finite rigid transform with a proper rotation")
    if tuple(planner.eef_names) != ("left_end_effector_link", "right_end_effector_link"):
        raise ValueError("Planner EEF order must match the H1 left/right end-effector links")
    if not callable(planner.body_eef_poses):
        raise ValueError("Planner body_eef_poses must be callable")


class Adapter:
    def __init__(self, planner, engine=None):
        self.planner = planner
        self.ik = BodyYawArmIK(planner) if engine is None else engine
        self.last = None

    def state(self, measured):
        return encode_state(self.planner.body_eef_poses(control19(measured)), measured)

    def reset(self, measured):
        self.ik.reset(measured)
        self.last = None

    def prepare(self, command, valid, measured, head):
        template = np.asarray(measured, float).copy()
        template[19:21] = head
        hands, target = decode_command(command, valid, template)
        actual = np.asarray(self.planner.body_eef_poses(control19(measured)))
        distances = np.linalg.norm(hands[:, :3] - actual[:, :3], axis=1)
        angles = (
            Rotation.from_quat(actual[:, 3:]).inv() * Rotation.from_quat(hands[:, 3:])
        ).magnitude()
        if np.max(distances) > 0.10 or np.max(angles) > 0.5:
            raise ValueError("EEF discontinuity: explicit transition required")
        solved = self.ik.solve(hands, target, measured)
        previous = measured if self.last is None else self.last
        if np.any(np.abs(solved[16:19] - previous[16:19]) > [0.03, 0.2, 0.2]):
            raise ValueError("Waist command discontinuity")
        return solved

    def accept(self, sent):
        self.ik.accept(sent)
        self.last = np.asarray(sent).copy()
