# SPDX-License-Identifier: Apache-2.0
# Adapted from the deployment-side original HighLevel IK wrapper.
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation


def sdk21(control, template):
    """Map control19 to SDK21, retaining grippers from the command template."""
    from .mapping import control19

    out = np.asarray(template, dtype=float).copy()
    control19(out)
    q = np.asarray(control, dtype=float)
    if q.shape != (19,) or not np.isfinite(q).all():
        raise ValueError("invalid solved control19")
    (out[:7], out[8:15], out[16:19], out[19:21]) = (q[3:10], q[10:17], q[:3], q[17:])
    return out


def original_solver(planner):
    """Load the external C++ HighLevel backend with continuity weight 3."""
    from processors.fast_ik_backend import FastCollisionModel, load_urdf, solve_highlevel

    expected = dict(
        SMOOTH_WEIGHT=3.0,
        REST_WEIGHT=0.01,
        MAX_ITERATIONS=5,
        DAMPING=0.001,
        POS_WEIGHT=60.0,
        ORI_WEIGHT=20.0,
        LIMIT_WEIGHT=20.0,
        LIMIT_MARGIN=0.15,
        LIMIT_MARGIN_WEIGHT=0.0,
        SELF_COLLISION=1,
        SELF_COLLISION_WEIGHT=3.0,
        SELF_COLLISION_MARGIN=0.01,
    )
    for name, value in expected.items():
        key = "PYROKI_FAST_HIGHLEVEL_" + name
        if key in os.environ and float(os.environ[key]) != value:
            raise ValueError(
                f"{key}={os.environ[key]} conflicts with validated EEF profile {value}"
            )
    collision = FastCollisionModel.from_urdf(load_urdf(planner.arm.urdf_path))

    def solve(positions, wxyz, seed):
        return solve_highlevel(
            robot=planner.arm,
            collision=collision,
            target_link_names=planner.eef_names,
            target_positions=positions,
            target_wxyzs=wxyz,
            prev_cfg=seed,
            home_cfg=seed,
            continuity_weight=3.0,
        )

    return solve


class BodyYawArmIK:
    """Solve bodyyaw EEF targets with a waist-fixed model; commit only after send."""

    def __init__(
        self,
        planner,
        solve=None,
        *,
        max_solves=8,
        max_s=0.02,
        position_tolerance=0.001,
        rotation_tolerance=0.01,
    ):
        self.planner = planner
        self.solve_native = original_solver(planner) if solve is None else solve
        self.previous = None
        self.last_diagnostics = None
        if (
            max_solves < 1
            or not np.isfinite([max_s, position_tolerance, rotation_tolerance]).all()
            or min(max_s, position_tolerance, rotation_tolerance) <= 0
        ):
            raise ValueError("invalid bounded IK settings")
        (self.max_solves, self.max_s) = (max_solves, max_s)
        (self.position_tolerance, self.rotation_tolerance) = (
            position_tolerance,
            rotation_tolerance,
        )

    def reset(self, measured=None):
        """Seed from measured joints on entry/recovery without publishing targets."""
        from .mapping import control19

        self.previous = None if measured is None else control19(measured).copy()
        self.last_diagnostics = None

    def solve(self, hands, action, measured):
        """Solve both arms, then check FK residuals, limits and command continuity.

        Args:
            hands: Two bodyyaw poses, each [x, y, z, qx, qy, qz, qw].
            action: SDK21 template holding waist, gripper and head targets.
            measured: Current SDK21 feedback for initial seeding.

        Returns:
            SDK21 targets. Call accept() only after every motor send succeeds.

        Raises:
            ValueError: Invalid targets, solver output or convergence failure.
        """
        from .mapping import control19

        self.last_diagnostics = None
        planner = self.planner
        hands = np.asarray(hands, dtype=float)
        (q, actual) = (control19(action), control19(measured))
        if hands.shape != (2, 7) or not np.isfinite(hands).all():
            raise ValueError("expected finite bodyyaw EEF (2,7)")
        if np.any(np.abs(np.linalg.norm(hands[:, 3:], axis=1) - 1) > 0.01):
            raise ValueError("EEF quaternion must be normalized")
        previous = actual if self.previous is None else self.previous
        body_targets = np.tile(np.eye(4), (2, 1, 1))
        body_targets[:, :3, :3] = Rotation.from_quat(hands[:, 3:]).as_matrix()
        body_targets[:, :3, 3] = hands[:, :3]
        targets = planner.root_from_body @ body_targets
        wxyz = Rotation.from_matrix(targets[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        indices = np.asarray(planner.arm_control_indices)
        seed = previous[indices].copy()
        started = time.monotonic()
        (history, projections, native_s) = ([], [], 0.0)
        for iteration in range(self.max_solves):
            if time.monotonic() - started > self.max_s:
                raise ValueError("IK convergence time budget exceeded; no command sent")
            call_started = time.monotonic()
            solved = np.asarray(self.solve_native(targets[:, :3, 3], wxyz, seed), dtype=float)
            native_s += time.monotonic() - call_started
            if solved.shape != seed.shape or not np.isfinite(solved).all():
                raise ValueError("invalid original HighLevel IK output")
            output = q.copy()
            for i, control_index in enumerate(indices):
                if 3 <= control_index < 17:
                    output[control_index] = solved[i]
            clipped = np.clip(output[3:17], planner.lower[3:17], planner.upper[3:17])
            delta = np.abs(clipped - output[3:17])
            if np.any(delta) and np.max(delta) <= 0.0001:
                projections.append(
                    dict(
                        iteration=iteration + 1,
                        max_correction_rad=float(np.max(delta)),
                        indices=(np.flatnonzero(delta) + 3).tolist(),
                    )
                )
                output[3:17] = clipped
            self.last_diagnostics = dict(
                ik_solves=iteration + 1,
                ik_s=native_s,
                solved_control19=output.tolist(),
                ik_converged=False,
                limit_projections=projections,
            )
            if np.any(output < planner.lower - 1e-06) or np.any(output > planner.upper + 1e-06):
                bad = np.flatnonzero(
                    (output < planner.lower - 1e-06) | (output > planner.upper + 1e-06)
                )
                self.last_diagnostics["outside_limits"] = [
                    dict(
                        index=int(i),
                        value=float(output[i]),
                        lower=float(planner.lower[i]),
                        upper=float(planner.upper[i]),
                    )
                    for i in bad
                ]
                raise ValueError(
                    f"solved joint/waist/head target outside limits: {self.last_diagnostics['outside_limits']}"
                )
            if np.any(np.abs(output[3:17] - previous[3:17]) > 0.25):
                raise ValueError("solved arm target discontinuity >0.25 rad")
            solved_hands = np.asarray(planner.body_eef_poses(output))
            if solved_hands.shape != (2, 7) or not np.isfinite(solved_hands).all():
                raise ValueError("invalid solved FK result")
            pos_error = np.linalg.norm(solved_hands[:, :3] - hands[:, :3], axis=1)
            rot_error = (
                Rotation.from_quat(hands[:, 3:]).inv() * Rotation.from_quat(solved_hands[:, 3:])
            ).magnitude()
            history.append(dict(position_m=pos_error.tolist(), rotation_rad=rot_error.tolist()))
            converged = bool(
                np.all(pos_error <= self.position_tolerance)
                and np.all(rot_error <= self.rotation_tolerance)
            )
            elapsed = time.monotonic() - started
            self.last_diagnostics = dict(
                ik_s=native_s,
                ik_total_s=elapsed,
                ik_solves=iteration + 1,
                ik_position_error_m=pos_error.tolist(),
                ik_rotation_error_rad=rot_error.tolist(),
                solved_control19=output.tolist(),
                ik_residual_warning=not converged,
                ik_converged=converged,
                ik_residual_history=history,
                limit_projections=projections,
                ik_profile="bounded-eef-v2: smooth=3, home=iteration_seed",
                ik_position_tolerance_m=self.position_tolerance,
                ik_rotation_tolerance_rad=self.rotation_tolerance,
            )
            if elapsed > self.max_s:
                raise ValueError("IK convergence time budget exceeded; no command sent")
            if converged:
                return sdk21(output, action)
            seed = output[indices].copy()
        raise ValueError(
            f"IK did not converge: position_m={pos_error.tolist()}, rotation_rad={rot_error.tolist()}; no command sent"
        )

    def accept(self, sent):
        """Commit after all SDK sends succeed; acknowledgement is not physical arrival."""
        from .mapping import control19

        self.previous = control19(sent).copy()
