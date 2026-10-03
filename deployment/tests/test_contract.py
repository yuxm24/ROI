import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from zwam.contract import images_rgb, metadata, state23, validate_metadata
from zwam.h1.mapping import control19, decode_command, encode_state, rotation6d


def sample():
    sdk = np.zeros(21, np.float32)
    sdk[[7, 15]] = [0.2, 0.8]
    sdk[16:19] = [0.5, 0.1, -0.2]
    hands = np.array([[0.3, 0.2, 0.1, 0, 0, 0, 1], [0.3, -0.2, 0.1, 0, 0, 0, 1]])
    return sdk, hands


@pytest.mark.parametrize("seed", range(6))
def test_rotation_columns(seed):
    r = Rotation.random(random_state=seed).as_matrix()
    np.testing.assert_allclose(rotation6d(np.r_[r[:, 0], r[:, 1]]), r, atol=1e-12)


@pytest.mark.parametrize("bad", [np.zeros(6), [1, 0, 0, 2, 0, 0], [np.nan] * 6])
def test_bad_rotation(bad):
    with pytest.raises(ValueError):
        rotation6d(bad)


def test_joint_mapping():
    q = np.arange(21)
    np.testing.assert_array_equal(control19(q), np.r_[q[16:19], q[:7], q[8:15], q[19:21]])


def test_state_command_no_home_offset_or_double_gripper_scale():
    sdk, hands = sample()
    state = encode_state(hands, sdk)
    np.testing.assert_allclose(state[[9, 19]], [0.2, 0.8])
    state[[9, 19]] = [0.3, 1.2]
    decoded, command = decode_command(state, np.ones(23, bool), sdk)
    np.testing.assert_allclose(decoded, hands, atol=1e-6)
    np.testing.assert_allclose(command[[7, 15]], [0.3, 1.2])
    np.testing.assert_allclose(command[16:19], sdk[16:19])


@pytest.mark.parametrize("grip", [-0.1, 1.51, float("inf")])
def test_bad_gripper(grip):
    sdk, hands = sample()
    command = encode_state(hands, sdk)
    command[9] = grip
    with pytest.raises(ValueError):
        decode_command(command, np.ones(23, bool), sdk)


def test_metadata_rejects_wrong_mode():
    data = metadata()
    validate_metadata(data)
    data["action_horizon"] = 12
    with pytest.raises(ValueError):
        validate_metadata(data)


def test_state_dtype_invalid_mask():
    state, valid = np.zeros(23, np.float32), np.ones(23, bool)
    state23(state, valid)
    with pytest.raises(ValueError):
        state23(state.astype(np.float64), valid)
    valid[0], state[0] = False, 1
    with pytest.raises(ValueError):
        state23(state, valid)


def test_frame_count():
    frames = np.zeros((1, 3, 8, 8, 3), np.uint8)
    images_rgb(frames, 1)
    with pytest.raises(ValueError):
        images_rgb(frames, 4)
