"""MetaWorld applied-command and image conventions."""

import hashlib, json

from pathlib import Path

PROTOCOL = 'metaworld311-mt50-applied4-v52-v1'

OFFICIAL_EVALUATION = 'mt50-all50goals-h500-no-warmup-v1'

ROBOT = 'metaworld_sawyer_applied4'

STATS_FORMAT = PROTOCOL + '-state-minmax'

STATE_SLOTS = (0, 1, 2, 9)

ACTION_LAYOUT = ('command.dx', 'command.dy', 'command.dz', 'command.gripper')

PHYSICAL_DT = .0125

VIDEO_DT = .05

ALIASES = {'nut-assembly-v3': 'assembly-v3', 'nut-disassemble-v3': 'disassemble-v3',
           'door-v3': 'door-open-v3', 'peg-insertion-side-v3': 'peg-insert-side-v3',
           'sweep-into-goal-v3': 'sweep-into-v3'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def state20(state):
    import numpy as np
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != 4 or not np.isfinite(state).all():
        raise ValueError('MetaWorld observations must be finite EEF xyz + opening')
    result = np.zeros(state.shape[:-1] + (20,), np.float32)
    result[..., list(STATE_SLOTS)] = state
    return result

def stored_image(raw_rgb):
    """Online render -> stored-image domain. Never apply to decoded dataset video."""
    import cv2
    import numpy as np
    frame = np.asarray(raw_rgb)
    if frame.shape != (480, 480, 3) or frame.dtype != np.uint8:
        raise ValueError('expected 480x480 RGB render')
    return cv2.resize(cv2.rotate(frame, cv2.ROTATE_180)[80:400, 80:400], (480, 480),
                      interpolation=cv2.INTER_LINEAR)

def canonical_image(stored_rgb, size):
    from zvla.vision.letterbox import resize_with_pad_rgb, resize_canonical_canvas
    return resize_canonical_canvas(resize_with_pad_rgb(stored_rgb)[0], size)
