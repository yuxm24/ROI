"""Dependency-light absolute drive-target ABI (not achieved joint positions)."""
import numpy as np

ROBOT_TYPE = 'robotwin2_aloha_agilex_joint14'
CONTRACT = 'robotwin-drive-qpos14-absolute-offset1-tail-v1'
FORMAT = 'robotwin-joint14-fixed-minmax-state-action-v1'
RESPONSE_CONTRACT = CONTRACT+'-official-qpos14-fp32-v1'
LAYOUT = tuple([f'left.joint{i}.drive_target_rad' for i in range(6)] +
               ['left.gripper_opening'] + [f'right.joint{i}.drive_target_rad' for i in range(6)] +
               ['right.gripper_opening'] + [f'invalid.padding{i}' for i in range(6)])


def pack_joint14(value):
    value=np.asarray(value,dtype=np.float32)
    if value.shape[-1:]!=(14,) or not np.isfinite(value).all():
        raise ValueError('finite official Joint14 vector required')
    result=np.zeros((*value.shape[:-1],20),np.float32)
    result[...,:14]=value
    return result
