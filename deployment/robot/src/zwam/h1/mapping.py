"""Command23 (bodyyaw absolute) <-> measured SDK joint feedback."""

import numpy as np
from scipy.spatial.transform import Rotation


def control19(sdk):
    sdk = np.asarray(sdk, dtype=float)
    if sdk.shape != (21,) or not np.isfinite(sdk).all():
        raise ValueError("Expected finite SDK21")
    return np.r_[sdk[16:19], sdk[:7], sdk[8:15], sdk[19:21]]


def rotation6d(value):
    value = np.asarray(value, float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("Invalid rotation6D")
    x, second = value[:3], value[3:]
    if np.linalg.norm(x) < 1e-6:
        raise ValueError("Degenerate rotation column0")
    x = x / np.linalg.norm(x)
    y = second - x * np.dot(x, second)
    if np.linalg.norm(y) < 1e-6:
        raise ValueError("Degenerate rotation column1")
    y = y / np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


def encode_state(hands, sdk):
    control19(sdk)
    hands = np.asarray(hands, float)
    if hands.shape != (2, 7) or not np.isfinite(hands).all():
        raise ValueError("FK must return two xyz+xyzw poses in bodyyaw")
    if np.any(np.abs(np.linalg.norm(hands[:, 3:], axis=1) - 1) > 0.01):
        raise ValueError("Invalid FK quaternion")
    if np.any(np.asarray(sdk)[[7, 15]] < 0) or np.any(np.asarray(sdk)[[7, 15]] > 1):
        raise ValueError("Gripper observation must be normalized 0..1")
    result = np.zeros(23, np.float32)
    for i, offset in enumerate((0, 10)):
        r = Rotation.from_quat(hands[i, 3:]).as_matrix()
        result[offset : offset + 3] = hands[i, :3]
        result[offset + 3 : offset + 9] = np.r_[r[:, 0], r[:, 1]]
        result[offset + 9] = sdk[(7, 15)[i]]
    result[20:] = sdk[16:19]
    return result


def decode_command(command, valid, template):
    command, valid = np.asarray(command), np.asarray(valid)
    control19(template)
    if (
        command.shape != (23,)
        or valid.shape != (23,)
        or valid.dtype != np.bool_
        or not valid.all()
        or not np.isfinite(command).all()
    ):
        raise ValueError("All Command23 dimensions must be finite and valid")
    if np.any(command[[9, 19]] < 0) or np.any(command[[9, 19]] > 1.5):
        raise ValueError("Gripper commands outside 0..1.5; not rescaled or silently clipped")
    hands = [
        np.r_[
            command[o : o + 3], Rotation.from_matrix(rotation6d(command[o + 3 : o + 9])).as_quat()
        ]
        for o in (0, 10)
    ]
    target = np.asarray(template, float).copy()
    target[[7, 15]] = command[[9, 19]]
    target[16:19] = command[20:23]
    return np.array(hands), target
