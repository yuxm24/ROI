import time

import numpy as np
import pytest
from zwam.h1.runtime import control_loop, offline_check
from zwam.h1.sdk import SDKEnvironment


class Env:
    fault = None
    sent = 0
    closed = 0

    def ready(self):
        pass

    def state(self):
        return np.zeros(21)

    def images(self):
        return np.zeros((3, 4, 4, 3), np.uint8)

    def send(self, target):
        self.sent += 1

    def close(self):
        self.closed += 1


class Adapter:
    resets = 0
    calls = 0
    accepted = 0

    def reset(self, _):
        self.resets += 1

    def state(self, _):
        return np.zeros(23, np.float32)

    def prepare(self, *_):
        self.calls += 1
        if self.calls in (1, 3):
            raise ValueError("IK convergence failed")
        return np.zeros(21)

    def accept(self, _):
        self.accepted += 1


class Policy:
    resets = 0

    def reset(self, *_):
        self.resets += 1

    def infer(self, *_):
        return np.zeros((4, 23), np.float32), np.ones((4, 23), bool)


def test_ik_pause_resume_never_deinitializes(monkeypatch):
    monkeypatch.setattr("zwam.h1.runtime.time.sleep", lambda _: None)
    answers = iter(["", "q"])
    env, adapter, policy = Env(), Adapter(), Policy()
    control_loop(env, adapter, policy, "task", read_input=lambda _: next(answers))
    assert env.closed == 0
    assert env.sent == adapter.accepted == 1
    assert policy.resets == adapter.resets == 2


def test_default_check_does_not_import_sdk(monkeypatch, capsys):
    def forbidden(*_, **__):
        raise AssertionError("Hardware construction")

    monkeypatch.setattr(SDKEnvironment, "initialize", forbidden)
    offline_check()
    assert "no SDK" in capsys.readouterr().out


def test_partial_send_latches():
    class Robot:
        def isRobotConnected(self):
            return True

        def getInitState(self):
            return 2

        def getCurrentMode(self):
            return 3

        def setMotorControl_low(self, i, _):
            return i < 2

    class Binding:
        class MotorControl:
            pass

    env = SDKEnvironment({})
    env.robot, env.bindings, env.indices = Robot(), Binding(), list(range(21))
    with pytest.raises(RuntimeError, match="partial-publication"):
        env.send(np.zeros(21))
    with pytest.raises(RuntimeError, match="partial-publication"):
        env.ready()


def test_public_sdk_motor_control_type_is_accepted():
    class Robot:
        def isRobotConnected(self):
            return True

        def getInitState(self):
            return 2

        def getCurrentMode(self):
            return 3

        def setMotorControl_low(self, _, control):
            assert control.Position == 0.5
            return True

    class Binding:
        class Motor_Control:
            pass

    env = SDKEnvironment({})
    env.robot, env.bindings, env.indices = Robot(), Binding(), [0] * 21
    env.send(np.full(21, 0.5))
    assert env.fault is None


def test_public_camera_wall_clock_timestamps_are_checked():
    class Camera:
        def __init__(self, stamp):
            self.stamp = stamp

        def get_latest_frame(self, _):
            return np.zeros((4, 4, 3), np.uint8), self.stamp

    env = SDKEnvironment({})
    env.camera = Camera(time.time() - 0.02)
    assert env.images(timeout=0.1).shape == (3, 4, 4, 3)

    env.camera = Camera(time.time() - 1.0)
    with pytest.raises(RuntimeError, match="stale"):
        env.images(timeout=0.01)
