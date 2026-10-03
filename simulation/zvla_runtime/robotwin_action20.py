"""NumPy-only RoboTwin reported-EEF <-> anchor-relative Action20 ABI.

Input poses are the *reported* world EEF poses (xyz, wxyz), not URDF link
poses. No TCP correction, LIBERO scaling, joint conversion or decimal rounding
belongs here. This describes EEF targets, not lossless joint-path replay.
All geometry stays float64; callers cast only at cache/model boundaries.
"""
from __future__ import annotations

import numpy as np

ROBOT_TYPE = "robotwin2_aloha_agilex"
CONTRACT = "robotwin2-reported-world-eef-action20-offset1-tail-v1"
# The current RoboTwin standardized release stores command targets explicitly
# under ``/action``.  It is numerically close to the next reported state, but
# it is a different source contract and must never be silently shifted by one
# row.  Keep the historical CONTRACT unchanged for old caches.
OFFICIAL_COMMAND_CONTRACT = "robotwin2-reported-world-eef-action20-command-offset0-tail-v1"
# Public State20 input identity for the standardized action-command producer.
# This is deliberately distinct from ``STATE_TRAJECTORY_CONTRACT``: the model
# tensor layout is identical, but the target owner and timeline are not.
OFFICIAL_STATE_TRAJECTORY_CONTRACT = (
    "zvla-state20-absolute-rotation6d-eef-action-command-v1")


def _finite(value, width):
    result = np.asarray(value, dtype=np.float64)
    if result.shape[-1:] != (width,) or not np.isfinite(result).all():
        raise ValueError(f"expected finite (...,{width}) array")
    return result


def quaternion_to_matrix(value):
    q = _finite(value, 4)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("zero quaternion")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def matrix_to_quaternion(value):
    """Stable at pi, with deterministic sign; rejects non-rotations."""
    r = np.asarray(value, dtype=np.float64)
    if (r.shape[-2:] != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-8, rtol=0)):
        raise ValueError("expected proper rotation matrices")
    result = np.empty(r.shape[:-2]+(4,), np.float64)
    for mat, out in zip(r.reshape(-1, 3, 3), result.reshape(-1, 4)):
        # Symmetric quaternion eigenproblem in xyzw order, stable for all angles.
        xx, xy, xz, yx, yy, yz, zx, zy, zz = mat.reshape(-1)
        k = np.array([[xx-yy-zz, xy+yx, xz+zx, zy-yz],
                      [xy+yx, yy-xx-zz, yz+zy, xz-zx],
                      [xz+zx, yz+zy, zz-xx-yy, yx-xy],
                      [zy-yz, xz-zx, yx-xy, xx+yy+zz]]) / 3
        _, vectors = np.linalg.eigh(k)
        out[:] = vectors[:, -1][[3, 0, 1, 2]]
        pivot = 0 if abs(out[0]) > 1e-12 else int(np.argmax(abs(out)))
        if out[pivot] < 0:
            out *= -1
    return result


def matrix_to_rot6d(r):
    r = np.asarray(r, dtype=np.float64)
    if r.shape[-2:] != (3, 3) or not np.isfinite(r).all():
        raise ValueError("expected finite rotation (...,3,3)")
    return np.concatenate((r[..., :, 0], r[..., :, 1]), axis=-1)


def rot6d_to_matrix(value):
    v = _finite(value, 6)
    n = np.linalg.norm(v[..., :3], axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("degenerate first rotation column")
    first = v[..., :3] / n
    second = v[..., 3:] - first * np.sum(first*v[..., 3:], axis=-1, keepdims=True)
    n = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("collinear rotation columns")
    second /= n
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def pack_reported_state(left_pose, left_opening, right_pose, right_opening):
    """Return float64 State20. Opening is the recorded drive command, not qpos."""
    arms = []
    for pose, opening in ((left_pose, left_opening), (right_pose, right_opening)):
        pose = _finite(pose, 7)
        g = np.asarray(opening, dtype=np.float64)
        if g.shape == pose.shape[:-1]+(1,):
            g = g[..., 0]
        if g.shape != pose.shape[:-1] or not np.isfinite(g).all() or np.any((g < 0) | (g > 1)):
            raise ValueError("recorded opening must match pose rows and lie in [0,1]")
        arms.append(np.concatenate((pose[..., :3], matrix_to_rot6d(quaternion_to_matrix(pose[..., 3:])),
                                    (1-2*g)[..., None]), axis=-1))
    if arms[0].shape != arms[1].shape:
        raise ValueError("left and right trajectory shapes differ")
    return np.concatenate(arms, axis=-1)


def relative_action20(anchor_state, target_states):
    """Unnormalized shared-codec semantics, used for offline parity tests."""
    anchor, future = np.broadcast_arrays(_finite(anchor_state, 20), _finite(target_states, 20))
    result = np.empty_like(future)
    for start in (0, 10):
        r0 = rot6d_to_matrix(anchor[..., start+3:start+9])
        rh = rot6d_to_matrix(future[..., start+3:start+9])
        result[..., start:start+3] = future[..., start:start+3] - anchor[..., start:start+3]
        result[..., start+3:start+9] = matrix_to_rot6d(r0.swapaxes(-1, -2) @ rh)
        result[..., start+9] = future[..., start+9]
    return result


def absolute_targets(anchor_state, relative_actions):
    anchor, action = np.broadcast_arrays(_finite(anchor_state, 20), _finite(relative_actions, 20))
    result = np.empty_like(action)
    for start in (0, 10):
        r0 = rot6d_to_matrix(anchor[..., start+3:start+9])
        rel = rot6d_to_matrix(action[..., start+3:start+9])
        result[..., start:start+3] = anchor[..., start:start+3] + action[..., start:start+3]
        result[..., start+3:start+9] = matrix_to_rot6d(r0 @ rel)
        result[..., start+9] = action[..., start+9]
    return result


def official_ee_commands(anchor_state, relative_actions):
    """Decode a whole chunk against ONE anchor; clip only simulator opening.

    Input must already be denormalized by the checkpoint's shared MinMax codec.
    Return (...,16): [Lxyz,Lwxyz,Lopening,Rxyz,Rwxyz,Ropening]. No xyz clip.
    """
    return absolute_state_to_official_ee(absolute_targets(anchor_state, relative_actions))


def absolute_state_to_official_ee(target_states):
    """Simulator boundary for shared codec.decode_predictions absolute output."""
    target = _finite(target_states, 20)
    arms = []
    for start in (0, 10):
        arms.append(np.concatenate((target[..., start:start+3],
                    matrix_to_quaternion(rot6d_to_matrix(target[..., start+3:start+9])),
                    np.clip((1-target[..., start+9:start+10])/2, 0, 1)), axis=-1))
    return np.concatenate(arms, axis=-1)


def anchor_rows(native_rows):
    if not isinstance(native_rows, (int, np.integer)) or native_rows < 2:
        raise ValueError("an episode needs at least two rows")
    return np.arange(0, native_rows-1, 4, dtype=np.int64)


def target_window(native_rows, anchor_index):
    """Return indices and independent supervision masks, never attention masks."""
    anchors = anchor_rows(native_rows)
    if not isinstance(anchor_index, (int, np.integer)) or not 0 <= anchor_index < len(anchors):
        raise IndexError("anchor outside episode")
    source = int(anchors[anchor_index])
    targets = source + np.arange(1, 17, dtype=np.int64)
    future_rgb = source + np.arange(4, 17, 4, dtype=np.int64)
    return dict(anchor_row=source, target_rows=targets,
                action_label_valid=targets < native_rows,
                future_rgb_rows=future_rgb, world_label_valid=future_rgb < native_rows)


def padded_targets(states, anchor_index):
    states = _finite(states, 20)
    if states.ndim != 2:
        raise ValueError("one episode (T,20) required")
    window = target_window(len(states), anchor_index)
    valid = window['action_label_valid']
    targets = np.zeros((16, 20), np.float64)
    targets[valid] = states[window['target_rows'][valid]]
    # The zero-padded invalid rows must not be passed to geometric conversion.
    actions = np.zeros_like(targets)
    actions[valid] = relative_action20(states[window['anchor_row']], targets[valid])
    return targets, actions, np.broadcast_to(valid[:, None], targets.shape).copy(), window
