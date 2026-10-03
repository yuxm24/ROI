"""Content-checked preparation receipts and atomic cache publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

CACHE_FORMAT = 'zvla-prepare-cache-v1'
EPISODE_FORMAT = 'zvla-prepare-episode-v1'
SUMMARY_FORMAT = 'zvla-prepare-summary-v1'
ROOT_RECEIPT = 'prepare_manifest.json'
SUMMARY_RECEIPT = 'summary_receipt.json'
ARRAY_NAMES = ('states', 'targets', 'state_valid', 'target_valid', 'video', 'geometry')
SUMMARY_NAMES = ('codec.json', 'tasks.json', 'text_embeddings.npy', 'dataset.json')


class PreparationIdentityChanged(ValueError):
    """Inputs changed, so newly generated completion markers are unsafe."""


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def identity_digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes(path, value):
    with Path(path).open('wb') as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path, value):
    write_bytes(path, json_bytes(value))


def atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'Preparation receipt is missing or invalid: {path}') from error


@contextmanager
def output_lock(output):
    """Keep a stable sibling lock inode; a killed process releases flock."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.parent / ('.' + output.name + '.prepare.lock')
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Output is already being prepared by another process: {output}') from error
        yield
    finally:
        os.close(descriptor)


def file_identity(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    after = path.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, key) != getattr(after, key) for key in keys):
        raise ValueError(f'Preparation cache changed while validating: {path}')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def array_identity(value):
    value = np.asarray(value)
    digest = hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()
    return {'dtype': value.dtype.str, 'shape': list(value.shape), 'sha256': digest}


def episode_identity(key, prompts, state, target, state_valid, target_valid,
                     streams, camera_count):
    if not isinstance(key, str) or not key:
        raise ValueError('Every preparation episode needs a non-empty source key')
    prompts = list(prompts)
    if not prompts or any(not isinstance(text, str) or not text.strip() for text in prompts):
        raise ValueError(f'Episode {key} has no valid instruction')
    arrays = dict(states=np.asarray(state), targets=np.asarray(target),
                  state_valid=np.asarray(state_valid), target_valid=np.asarray(target_valid))
    count = len(arrays['states'])
    if count <= 0:
        raise ValueError(f'Episode {key} is empty')
    for name, value in arrays.items():
        if value.shape != (count, 20):
            raise ValueError(f'Episode {key}: {name} must have shape ({count}, 20)')
        if name.endswith('_valid'):
            if value.dtype != np.bool_:
                raise ValueError(f'Episode {key}: {name} must be boolean')
        elif not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f'Episode {key}: {name} must contain finite floating-point values')
    if len(streams) != camera_count:
        raise ValueError(f'Episode {key} has {len(streams)} cameras, expected {camera_count}')
    for stream in streams:
        value = np.asarray(stream)
        if (value.ndim != 4 or value.shape[0] != count or value.shape[-1] != 3
                or value.dtype != np.uint8 or min(value.shape[1:3]) <= 0):
            raise ValueError(f'Episode {key} must have native-row-aligned uint8 RGB camera streams')
    result = {'key': key, 'prompts': prompts, 'native_rows': count,
              'arrays': {name: array_identity(value) for name, value in arrays.items()},
              'images': [array_identity(stream) for stream in streams]}
    result['identity_sha256'] = identity_digest(result)
    return result


def make_root_receipt(config, episodes):
    identity = {'config': config, 'episodes': episodes}
    return {'format': CACHE_FORMAT, 'identity': identity,
            'identity_sha256': identity_digest(identity)}


def existing_root_receipt(output, resume):
    output = Path(output)
    if output.exists() and not output.is_dir():
        raise ValueError(f'Preparation output is not a directory: {output}')
    if (output / 'episodes').is_symlink():
        raise ValueError('Preparation episode directory must not be a symlink')
    nonempty = output.is_dir() and any(output.iterdir())
    if not resume:
        if nonempty:
            raise ValueError(f'Preparation output is not empty: {output}. Use --resume with a matching new-format cache or choose a new output directory.')
        return None
    path = output / ROOT_RECEIPT
    if path.is_symlink():
        raise ValueError('Preparation root receipt must not be a symlink')
    if not path.is_file():
        raise ValueError('Cannot resume an old or missing preparation cache without identity receipts; choose a new output directory')
    receipt = read_json(path)
    if (not isinstance(receipt, dict) or receipt.get('format') != CACHE_FORMAT
            or receipt.get('identity_sha256') != identity_digest(receipt.get('identity'))):
        raise ValueError(f'Preparation root identity receipt is invalid: {path}')
    return receipt


def check_root_identity(existing, expected):
    if existing is not None and existing != expected:
        raise PreparationIdentityChanged('Preparation identity changed (source, episode inventory, configuration or model assets); use a new output directory')


def shard_path(output, index):
    return Path(output) / 'episodes' / f'{index:06d}'


def validate_shard(output, index, source, root_digest):
    """A marked shard is immutable; corruption is never silently regenerated."""
    shard = shard_path(output, index)
    if shard.parent.is_symlink() or shard.is_symlink():
        raise ValueError(f'Preparation cache shard must not be a symlink: {shard}')
    if (shard / '.complete').is_symlink() or (shard / 'receipt.json').is_symlink():
        raise ValueError(f'Preparation episode receipt and marker must not be symlinks: {shard}')
    if not (shard / '.complete').exists():
        return None
    receipt = read_json(shard / 'receipt.json')
    expected_header = {'format': EPISODE_FORMAT, 'root_identity_sha256': root_digest,
                       'source': source}
    if not isinstance(receipt, dict) or {name: receipt.get(name) for name in expected_header} != expected_header:
        raise ValueError(f'Completed episode identity is invalid: {shard}')
    if (shard / '.complete').read_text().strip() != identity_digest(receipt):
        raise ValueError(f'Completed episode marker is invalid: {shard}')
    files = receipt.get('files')
    if not isinstance(files, dict) or set(files) != {name + '.npy' for name in ARRAY_NAMES}:
        raise ValueError(f'Completed episode file inventory is invalid: {shard}')
    for name, expected in files.items():
        path = shard / name
        if path.is_symlink() or not path.is_file() or file_identity(path) != expected:
            raise ValueError(f'Completed episode file is missing or corrupt: {path}')
    return receipt


def publish_shard(output, index, source, root_digest, arrays):
    output = Path(output)
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError('Episode array inventory is incomplete')
    n = source['native_rows']
    count = (n + 3) // 4
    for name, value in arrays.items():
        value = np.asarray(value)
        expected = (source['arrays'][name] if name in source['arrays'] else
                    {'dtype': np.dtype(np.float16).str,
                     'shape': [count, 48, 42, 14] if name == 'video' else [count, 3, 49, 2048]})
        if list(value.shape) != expected['shape'] or value.dtype.str != expected['dtype']:
            raise ValueError(f'Prepared {name} has unexpected shape or dtype')
        if not np.isfinite(value).all():
            raise ValueError(f'Prepared {name} contains NaN or Inf')
        if name in source['arrays'] and array_identity(value) != source['arrays'][name]:
            raise ValueError(f'Prepared {name} no longer matches its source identity')
    staging = Path(tempfile.mkdtemp(prefix='.episode-staging-', dir=output))
    try:
        for name, value in arrays.items():
            with (staging / (name + '.npy')).open('wb') as handle:
                np.save(handle, value, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
        receipt = dict(format=EPISODE_FORMAT, root_identity_sha256=root_digest,
                       source=source,
                       files={name + '.npy': file_identity(staging / (name + '.npy'))
                              for name in ARRAY_NAMES})
        write_json(staging / 'receipt.json', receipt)
        write_bytes(staging / '.complete', (identity_digest(receipt) + '\n').encode())
        sync_directory(staging)
        target = shard_path(output, index)
        if target.parent.is_symlink():
            raise ValueError('Preparation episode directory must not be a symlink')
        target.parent.mkdir(exist_ok=True)
        sync_directory(output)
        if target.exists():
            if (target / '.complete').exists():
                raise ValueError(f'Refusing to replace a completed episode: {target}')
            if target.is_symlink() or not target.is_dir():
                raise ValueError(f'Invalid incomplete episode directory: {target}')
            shutil.rmtree(target)
        os.replace(staging, target)
        sync_directory(target.parent)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def validate_summary(output, root_digest, episode_receipts):
    output = Path(output)
    if (output / 'dataset.json').is_symlink() or (output / SUMMARY_RECEIPT).is_symlink():
        raise ValueError('Preparation summary and receipt must not be symlinks')
    if not (output / 'dataset.json').exists():
        return False
    if any(receipt is None for receipt in episode_receipts):
        raise ValueError('Completed preparation summary references an incomplete episode')
    receipt = read_json(output / SUMMARY_RECEIPT)
    if (not isinstance(receipt, dict) or receipt.get('format') != SUMMARY_FORMAT
            or receipt.get('root_identity_sha256') != root_digest
            or set(receipt.get('files', {})) != set(SUMMARY_NAMES)):
        raise ValueError('Completed preparation summary identity is invalid')
    for name, expected in receipt['files'].items():
        path = output / name
        if path.is_symlink() or not path.is_file() or file_identity(path) != expected:
            raise ValueError(f'Completed preparation summary is missing or corrupt: {path}')
    dataset = read_json(output / 'dataset.json')
    if dataset.get('summary_files') != {name: value for name, value in receipt['files'].items()
                                      if name != 'dataset.json'}:
        raise ValueError('Completed dataset summary content identity does not match its cache receipt')
    rows = dataset.get('episodes', [])
    if dataset.get('prepare_identity_sha256') != root_digest or len(rows) != len(episode_receipts):
        raise ValueError('Completed dataset identity does not match the preparation cache')
    for row, episode in zip(rows, episode_receipts, strict=True):
        if (row.get('source_identity_sha256') != episode['source']['identity_sha256']
                or row.get('output_identity_sha256') != identity_digest(episode['files'])):
            raise ValueError('Completed dataset episode identity does not match its cache receipt')
    return True


def publish_summary(output, staging, root_digest):
    """Publish the first summary once, with dataset.json as its commit marker."""
    output, staging = Path(output), Path(staging)
    if (output / 'dataset.json').exists():
        raise ValueError('Refusing to overwrite a committed preparation summary')
    receipt = {'format': SUMMARY_FORMAT, 'root_identity_sha256': root_digest,
               'files': {name: file_identity(staging / name) for name in SUMMARY_NAMES}}
    sync_directory(staging)
    for name in SUMMARY_NAMES[:-1]:
        os.replace(staging / name, output / name)
    atomic_json(output / SUMMARY_RECEIPT, receipt)
    os.replace(staging / 'dataset.json', output / 'dataset.json')
    sync_directory(output)


def cancel_new_shards(output, indices):
    for index in indices:
        shard = shard_path(output, index)
        (shard / '.complete').unlink(missing_ok=True)
        sync_directory(shard)


def clear_staging(output):
    """Remove only our uncommitted directories after successful preflight."""
    for path in Path(output).iterdir():
        if re.fullmatch(r'\.(episode|summary)-staging-[a-z0-9_]{8}', path.name):
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f'Invalid preparation staging directory: {path}')
            shutil.rmtree(path)
