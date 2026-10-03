"""RoboTwin official policy plugin."""
import os
import numpy as np
from zvla.posttrain.client import PolicyClient
from zvla_runtime.robotwin_action20 import pack_reported_state


def get_model(usr_args):
    model = PolicyClient(os.environ['ZVLA_SERVER_URL'])
    reset_model(model)
    return model


def reset_model(model):
    model.frames = []
    model.initialized = False


def eval(TASK_ENV, model, observation):
    if not model.initialized:
        model.reset(TASK_ENV.get_instruction())
        model.initialized = True
    # One call executes four native commands and collects their four new frames.
    if not model.frames:
        model.frames = [observation]
    names = ('head_camera','left_camera','right_camera')
    frames = np.stack([np.stack([o['observation'][c]['rgb'] for c in names]) for o in model.frames])
    ee = model.frames[-1]['endpose']
    state = pack_reported_state(ee['left_endpose'],ee['left_gripper'],ee['right_endpose'],ee['right_gripper'])
    actions = model.infer(frames, state)
    model.frames = []
    for action in actions:
        TASK_ENV.take_action(action, action_type='ee')
        model.frames.append(TASK_ENV.get_obs())
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
