"""XPolicyLab bridge to the public Z-VLA service."""
import os
from XPolicyLab.model_template import ModelTemplate
from zvla.posttrain.client import PolicyClient
from .action_adapter import extract_pose_state, extract_prompt, pack_camera_sequence
FRAME_SEQUENCE_KEY = 'zvla_frames'


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.client = PolicyClient(os.environ.get('ZVLA_SERVER_URL',model_cfg['server_url']))
        self.seed = int(model_cfg.get('seed',0))
        self.reset()

    def reset(self):
        self.initialized = False

    def prepare_case(self, obs):
        self.seed = int(obs['eval_seed'])

    def update_obs(self, obs):
        self.frames = obs[FRAME_SEQUENCE_KEY]

    def get_action(self):
        obs = self.frames[-1]
        if not self.initialized:
            self.client.reset(extract_prompt(obs),self.seed)
            self.initialized = True
        return self.client.infer(pack_camera_sequence(self.frames),extract_pose_state(obs)[2]).tolist()
