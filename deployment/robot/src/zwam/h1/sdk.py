"""External licensed H1 SDK adapter; no hardware imports at module import time."""

import importlib
import sys
import time
from pathlib import Path

import numpy as np

from ..contract import SDK_CAMERAS
from .mapping import control19


class SDKEnvironment:
    def __init__(self, config):
        self.config = config
        self.robot = self.camera = None
        self.init_requested = False
        self.fault = None
        self.last_stamps = np.full(3, -np.inf)

    def initialize(self):
        root = Path(self.config["sdk_root"]).expanduser().resolve(strict=True)
        camera_root = (
            Path(self.config.get("camera_sdk_root", root)).expanduser().resolve(strict=True)
        )
        sys.path[:0] = [str(camera_root), str(root), str(root / "lib")]
        b = importlib.import_module("lib_h1_sdk_python")
        self.bindings = b
        self.indices = [
            getattr(b.EtherCAT_Motor_Index, f"MOTOR_{side}_ARM_{i}")
            for side in ("LEFT", "RIGHT")
            for i in range(1, 9)
        ]
        self.indices += [
            getattr(b.EtherCAT_Motor_Index, name)
            for name in (
                "MOTOR_LIFT",
                "MOTOR_WAIST_DOWN",
                "MOTOR_WAIST_UP",
                "MOTOR_HEAD_DOWN",
                "MOTOR_HEAD_UP",
            )
        ]
        self.robot = b.H1Robot()
        self.robot.robot_connect()
        if not self.robot.isRobotConnected() or int(self.robot.getInitState()) not in (0, 4):
            raise RuntimeError("Not connected or already initialized; refusing takeover")
        if (
            not self.robot.switchControlMode(b.MotorControlMode.GRAVITY_COMPENSATION_LEVEL)
            or int(self.robot.getCurrentMode()) != 3
        ):
            raise RuntimeError("SDK mode3 switch failed before initialization")
        camera_module = (
            "camera_client"
            if any(camera_root.glob("camera_client*.so"))
            or (camera_root / "camera_client.py").is_file()
            else "lib.camera_client"
        )
        camera_cls = importlib.import_module(camera_module).CameraClient
        self.camera = camera_cls(grpc_target=self.config.get("camera_address", "localhost:50051"))
        self.camera.start()
        self.images(timeout=15)
        self.init_requested = True
        if not self.robot.robot_init() or int(self.robot.getInitState()) != 2:
            raise RuntimeError("SDK initialization failed; inspect robot locally")

    def ready(self):
        if self.fault:
            raise RuntimeError(self.fault)
        if (
            not self.robot.isRobotConnected()
            or int(self.robot.getInitState()) != 2
            or int(self.robot.getCurrentMode()) != 3
        ):
            raise RuntimeError("Robot is not initialized in SDK joint mode3")

    def state(self):
        if self.robot is None or not self.robot.isRobotConnected():
            raise RuntimeError("Disconnected SDK")
        values = []
        for index in self.indices:
            ok, state = self.robot.getMotorState(index)
            if not ok:
                raise RuntimeError("Motor feedback failed")
            values.append(state.Position_Actual)
        out = np.asarray(values, np.float32)
        control19(out)
        return out

    def images(self, timeout=1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frames, stamps = [], []
            for name in SDK_CAMERAS:
                result = None
                for alias in (name, "rs/" + name, "rs." + name, "rs_" + name):
                    result = self.camera.get_latest_frame(alias)
                    if result is not None:
                        break
                if result is None:
                    break
                frame, stamp = result
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError("Camera must provide uint8 BGR images")
                frames.append(frame[..., ::-1].copy())
                stamps.append(stamp)
            if len(stamps) == 3:
                times = np.asarray(stamps, float)
                monotonic_age = time.monotonic() - times
                wall_age = time.time() - times
                fresh = any(np.all((age >= 0) & (age <= 0.25)) for age in (monotonic_age, wall_age))
                if (
                    np.isfinite(times).all()
                    and np.all(times > self.last_stamps)
                    and fresh
                    and np.ptp(times) <= 0.10
                ):
                    images = np.stack(frames)
                    self.last_stamps = times
                    return images
            time.sleep(0.002)
        raise RuntimeError("Missing/stale/unsynchronized camera frames")

    def send(self, command):
        self.ready()
        control19(command)
        for i, (index, value) in enumerate(zip(self.indices, command, strict=False)):
            control_type = getattr(self.bindings, "Motor_Control", None)
            if control_type is None:
                control_type = self.bindings.MotorControl
            control = control_type()
            control.Position = float(value)
            try:
                accepted = self.robot.setMotorControl_low(index, control)
            except Exception:
                accepted = False
            if not accepted:
                self.fault = f"SDK partial-publication fault at slot {i}; no automatic resume"
                raise RuntimeError(self.fault)

    def close(self):
        """Explicit operator exit only. Deinitialization may move hardware."""
        try:
            if self.robot is not None and self.init_requested:
                if not self.robot.robot_deinit() or int(self.robot.getInitState()) not in (0, 4):
                    raise RuntimeError(
                        "Deinitialization failed; inspect locally, use physical E-stop if needed"
                    )
        finally:
            if self.camera is not None:
                self.camera.stop()
            self.camera = self.robot = None
