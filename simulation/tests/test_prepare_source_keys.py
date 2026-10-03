import io
import json
import struct
import zipfile

import numpy as np
from PIL import Image
from zvla.posttrain.sources import libero_episodes, robotwin_episodes
from zvla.posttrain.tfrecord import TF_EXAMPLE


def jpeg(color):
    encoded = io.BytesIO()
    Image.new('RGB', (7, 5), color).save(encoded, format='JPEG', quality=100, subsampling=0)
    return encoded.getvalue()


def masked_crc32c(value):
    crc = 0xffffffff
    for byte in value:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82f63b78 if crc & 1 else 0)
    crc ^= 0xffffffff
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def tfrecord(example):
    payload = example.SerializeToString()
    size = struct.pack('<Q', len(payload))
    return size + struct.pack('<I', masked_crc32c(size)) + payload + struct.pack('<I', masked_crc32c(payload))


def test_libero_episode_keys_distinguish_same_filename_in_different_directories(tmp_path):
    example = TF_EXAMPLE()
    features = example.features.feature
    features['steps/observation/state'].float_list.value.extend([.1, .2, .3, 0, 0, 0, .04, 0])
    features['steps/action'].float_list.value.extend([0] * 7)
    features['steps/language_instruction'].bytes_list.value.append(b'move the block')
    for camera in ('image', 'wrist_image'):
        features['steps/observation/' + camera].bytes_list.value.append(jpeg((180, 50, 20)))
    filename = 'train.tfrecord-00000-of-00001'
    for task in ('task_a', 'task_b'):
        directory = tmp_path / task
        directory.mkdir()
        (directory / filename).write_bytes(tfrecord(example))

    records = list(libero_episodes(tmp_path))
    assert [row[0] for row in records] == [f'{task}/{filename}-0' for task in ('task_a', 'task_b')]
    assert len({row[0] for row in records}) == 2
    for _, prompts, state, target, sv, av, decode in records:
        assert prompts == ['move the block']
        assert state.shape == target.shape == sv.shape == av.shape == (1, 20)
        streams = decode()
        assert len(streams) == 2
        for frames in streams:
            assert frames.shape == (1, 5, 7, 3)
            assert frames.dtype == np.uint8
            np.testing.assert_allclose(frames[0, 0, 0], [180, 50, 20], atol=2)


def test_robotwin_episode_keys_distinguish_same_archive_and_decode_during_yield(tmp_path):
    import h5py

    poses = np.array([[.1, .2, .3, 1, 0, 0, 0], [.11, .2, .3, 1, 0, 0, 0]])
    data = io.BytesIO()
    with h5py.File(data, 'w') as handle:
        for group in ('state', 'action'):
            handle.create_dataset(group + '/left_ee_poses', data=poses)
            handle.create_dataset(group + '/right_ee_poses', data=poses)
            handle.create_dataset(group + '/left_ee_joint_states', data=np.full((2, 1), .2))
            handle.create_dataset(group + '/right_ee_joint_states', data=np.full((2, 1), .8))
        for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
            colors = handle.create_dataset('vision/' + camera + '/colors', shape=(2,), dtype=h5py.vlen_dtype(np.uint8))
            for index in range(2):
                colors[index] = np.frombuffer(jpeg((180, 50, 20)), dtype=np.uint8)
    member = 'dataset/data/episode_0000000.hdf5'
    for task in ('task_a', 'task_b'):
        directory = tmp_path / task
        directory.mkdir()
        with zipfile.ZipFile(directory / 'demo_clean.zip', 'w') as archive:
            archive.writestr(member, data.getvalue())
            archive.writestr('dataset/instruction/episode_0000000.json', json.dumps({'seen': ['move the block']}))

    keys = []
    for key, prompts, state, target, sv, av, decode in robotwin_episodes(tmp_path):
        keys.append(key)
        assert prompts == ['move the block']
        assert state.shape == target.shape == sv.shape == av.shape == (2, 20)
        np.testing.assert_allclose(state[:, :3], poses[:, :3])
        np.testing.assert_allclose(state[:, [9, 19]], [[.6, -.6], [.6, -.6]])
        assert sv.all() and av.all()
        # The adapter owns an open HDF5 handle only until the generator advances.
        streams = decode()
        assert len(streams) == 3
        for frames in streams:
            assert frames.shape == (2, 5, 7, 3)
            assert frames.dtype == np.uint8
            np.testing.assert_allclose(frames[0, 0, 0], [180, 50, 20], atol=2)
    assert keys == [f'{task}/demo_clean.zip::{member}' for task in ('task_a', 'task_b')]
    assert len(set(keys)) == 2
