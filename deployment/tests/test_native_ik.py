from types import SimpleNamespace

import numpy as np
import pytest
from zwam.h1.mapping import control19
from zwam.h1.native_ik import BodyYawArmIK


def planner():
    def poses(q):
        return np.array([np.r_[q[3:6], 0, 0, 0, 1], np.r_[q[10:13], 0, 0, 0, 1]])

    return SimpleNamespace(
        lower=np.full(19, -2.0),
        upper=np.full(19, 2.0),
        arm_control_indices=np.arange(3, 17),
        root_from_body=np.eye(4),
        body_eef_poses=poses,
    )


def solve(positions, _, seed):
    seed = seed.copy()
    seed[:3], seed[7:10] = positions
    return seed


def test_ik_seed_committed_only_after_send():
    p = planner()
    engine = BodyYawArmIK(p, solve=solve, max_s=1)
    measured = np.zeros(21)
    engine.reset(measured)
    target = np.array([[0.02, 0.03, 0.04, 0, 0, 0, 1], [0.02, -0.03, 0.04, 0, 0, 0, 1]])
    solved = engine.solve(target, measured, measured)
    np.testing.assert_array_equal(engine.previous, np.zeros(19))
    np.testing.assert_allclose(p.body_eef_poses(control19(solved)), target)
    engine.accept(solved)
    np.testing.assert_allclose(engine.previous, control19(solved))


def test_ik_failure_does_not_commit():
    p = planner()
    engine = BodyYawArmIK(p, solve=lambda *args: np.full(14, np.nan), max_s=1)
    measured = np.zeros(21)
    engine.reset(measured)
    with pytest.raises(ValueError, match="invalid original"):
        engine.solve(p.body_eef_poses(np.zeros(19)), measured, measured)
    np.testing.assert_array_equal(engine.previous, np.zeros(19))


def test_ik_rejects_joint_branch_jump():
    p = planner()
    engine = BodyYawArmIK(p, solve=solve, max_s=1)
    measured = np.zeros(21)
    target = p.body_eef_poses(np.zeros(19))
    target[0, 0] = 0.3
    with pytest.raises(ValueError, match="discontinuity"):
        engine.solve(target, measured, measured)
