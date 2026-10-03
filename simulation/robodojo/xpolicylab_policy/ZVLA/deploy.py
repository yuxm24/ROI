"""Execute absolute EE targets through the official RoboDojo interface."""
from .action_adapter import robodojo_vector_to_action
from .model import FRAME_SEQUENCE_KEY


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name='prepare_case',obs={'eval_seed':TASK_ENV.eval_seed})
    model_client.call(func_name='reset')
    frames = [TASK_ENV.get_obs()]
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name='update_obs',obs={FRAME_SEQUENCE_KEY:frames})
        actions = model_client.call(func_name='get_action')
        frames = []
        for action in actions:
            TASK_ENV.take_action(robodojo_vector_to_action(action))
            if TASK_ENV.is_episode_end(): break
            frames.append(TASK_ENV.get_obs())
