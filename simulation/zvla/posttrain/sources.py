"""Readers for the five public benchmark releases."""
import io
import json
from pathlib import Path
import struct
import tempfile
import zipfile

import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(Path(path).read_text())


def libero_state(state):
    from zvla_runtime.libero_action20 import rotation_vector_to_matrix, matrix_to_rotation6d
    state = np.asarray(state, np.float64)
    result = np.zeros((len(state), 20), np.float32)
    result[:, :3] = state[:, :3]
    result[:, 3:9] = [matrix_to_rotation6d(rotation_vector_to_matrix(x[3:6])) for x in state]
    result[:, 9] = 1 - 2 * np.clip((state[:, 6] - state[:, 7]) / .08, 0, 1)
    return result


def convert_rows(bench, state, action):
    """Commands stay aligned with observations at the same row."""
    state, action = np.asarray(state), np.asarray(action)
    sv, av = np.zeros((len(state), 20), bool), np.zeros((len(action), 20), bool)
    targets = np.zeros((len(action), 20), np.float32)
    if bench in ('libero', 'libero-plus'):
        states = libero_state(state)
        sv[:, :10] = True
        targets[:, :7] = action
        av[:, :7] = True
    elif bench == 'metaworld':
        states = np.zeros((len(state), 20), np.float32)
        states[:, [0, 1, 2, 9]] = state[:, :4]
        sv[:, [0, 1, 2, 9]] = True
        targets[:, :4] = np.clip(action, -1, 1)
        av[:, :4] = True
    elif bench == 'robodojo':
        from zvla.actions.robodojo import absolute_ee_to_state20
        states, targets = absolute_ee_to_state20(state), absolute_ee_to_state20(action)
        sv[:] = av[:] = True
    else:
        states, targets = state.astype(np.float32), action.astype(np.float32)
        sv[:] = av[:] = True
    return states, targets, sv, av


def decode_video(path, start, count, fps):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(max(0, int(start / float(stream.time_base))), stream=stream)
        frames = []
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) + .5 / fps < start:
                continue
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise ValueError(f'Video has {len(frames)} frames, expected {count}: {path}')
    return np.stack(frames)


def lerobot_episodes(root, bench):
    import pyarrow.parquet as pq
    info = read_json(root / 'meta/info.json')
    cameras = {
        'libero-plus': ('observation.images.front', 'observation.images.wrist'),
        'robodojo': ('observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist'),
        'metaworld': tuple(k for k, v in info['features'].items() if v['dtype'] == 'video'),
    }[bench]
    v3 = info['codebase_version'] == 'v3.0'
    if v3:
        rows = [r for p in sorted((root / 'meta/episodes').rglob('*.parquet')) for r in pq.read_table(p).to_pylist()]
        bases = {}
        for row in rows:
            key = row['data/chunk_index'], row['data/file_index']
            bases[key] = min(bases.get(key, row['dataset_from_index']), row['dataset_from_index'])
    else:
        rows = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines()]
    # Respect the publisher's train split, including its original episode IDs.
    train_start, train_stop = map(int, info.get('splits', {'train': f'0:{len(rows)}'})['train'].split(':'))
    for row in rows:
        ep, n = row['episode_index'], row['length']
        if not train_start <= ep < train_stop:
            continue
        if v3:
            key = row['data/chunk_index'], row['data/file_index']
            path = root / info['data_path'].format(chunk_index=key[0], file_index=key[1])
            offset = row['dataset_from_index'] - bases[key]
        else:
            path = root / info['data_path'].format(episode_chunk=ep // info['chunks_size'], episode_index=ep)
            offset = 0
        table = pq.read_table(path, columns=['observation.state', 'action']).slice(offset, n)
        state, target, sv, av = convert_rows(bench, table['observation.state'].to_pylist(), table['action'].to_pylist())
        videos = []
        for camera in cameras:
            if v3:
                prefix = 'videos/' + camera
                path = root / info['video_path'].format(video_key=camera, chunk_index=row[prefix+'/chunk_index'], file_index=row[prefix+'/file_index'])
                start = row[prefix+'/from_timestamp']
            else:
                path = root / info['video_path'].format(video_key=camera, episode_chunk=ep // info['chunks_size'], episode_index=ep)
                start = 0.
            videos.append((path, start, n, info['fps']))
        yield str(ep), row['tasks'], state, target, sv, av, lambda videos=videos: [decode_video(*v) for v in videos]


def libero_episodes(root):
    from .tfrecord import TF_EXAMPLE
    for path in sorted(root.rglob('*.tfrecord-*')):
        with path.open('rb') as handle:
            while header := handle.read(12):
                offset = handle.tell() - 12
                size = struct.unpack('<Q', header[:8])[0]
                example = TF_EXAMPLE()
                example.ParseFromString(handle.read(size))
                handle.read(4)
                f = example.features.feature
                state = np.array(f['steps/observation/state'].float_list.value).reshape(-1, 8)
                action = np.array(f['steps/action'].float_list.value).reshape(-1, 7)
                state, target, sv, av = convert_rows('libero', state, action)
                streams = [list(f['steps/observation/'+key].bytes_list.value) for key in ('image', 'wrist_image')]
                def images(streams=streams):
                    return [np.stack([np.array(Image.open(io.BytesIO(x)).convert('RGB')) for x in stream]) for stream in streams]
                yield path.relative_to(root).as_posix()+'-'+str(offset), [f['steps/language_instruction'].bytes_list.value[0].decode()], state, target, sv, av, images


def robotwin_episodes(root):
    import h5py
    from zvla_runtime.robotwin_action20 import pack_reported_state
    camera_fields = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
    for archive in sorted(root.rglob('*.zip')):
        if 'demo_clean' not in str(archive):
            continue
        with zipfile.ZipFile(archive) as z:
            for member in sorted(n for n in z.namelist() if n.endswith('.hdf5')):
                episode = int(Path(member).stem.replace('episode_', ''))
                instruction = next(n for n in z.namelist() if n.endswith(f'/instruction/episode_{episode:07d}.json'))
                prompts = sorted(set(json.loads(z.read(instruction))['seen']))
                with tempfile.TemporaryFile() as tmp:
                    with z.open(member) as src:
                        import shutil
                        shutil.copyfileobj(src, tmp)
                    tmp.seek(0)
                    with h5py.File(tmp, 'r') as h:
                        arrays = []
                        for group in ('state', 'action'):
                            arrays.append(pack_reported_state(h[group+'/left_ee_poses'][:], h[group+'/left_ee_joint_states'][:],
                                h[group+'/right_ee_poses'][:], h[group+'/right_ee_joint_states'][:]).astype(np.float32))
                        state, target, sv, av = convert_rows('robotwin', *arrays)
                        def images(h=h):
                            import cv2
                            return [np.stack([cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
                                              for x in h['vision/'+cam+'/colors']]) for cam in camera_fields]
                        yield archive.relative_to(root).as_posix()+'::'+member, prompts, state, target, sv, av, images


def episodes(root, bench):
    root = Path(root)
    if bench == 'libero':
        return libero_episodes(root)
    if bench == 'robotwin':
        return robotwin_episodes(root)
    return lerobot_episodes(root, bench)
