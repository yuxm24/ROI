"""Rank-zero distribution of immutable training metadata.

Large production indexes live on CFS, but every trainer rank needs the same
bytes.  Reading a 52-MiB JSONL index independently from 152 processes creates
an avoidable metadata/read storm.  Megatron production startup therefore has
rank zero read each sealed file once, compresses the payload, broadcasts it
through NCCL, and installs a process-local read cache before datasets are
constructed.  Training shards themselves remain demand-paged from CFS.
"""

from __future__ import annotations

import hashlib
import errno
import os
import pickle
import time
import zlib
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist


_CACHE: dict[str, bytes] = {}
_FORMAT = "zvla-shared-metadata-v1"
_MAX_TOTAL_BYTES = 512 << 20
_TRANSIENT_ERRNOS = frozenset({
    errno.EAGAIN,
    errno.EBUSY,
    errno.EINTR,
    errno.EIO,
    errno.ENETDOWN,
    errno.ENETRESET,
    errno.ENETUNREACH,
    errno.ESTALE,
    errno.ETIMEDOUT,
})
_READ_ATTEMPTS = 3


def _key(path: str | Path) -> str:
    return os.path.realpath(os.fspath(path))


def has_shared_metadata(path: str | Path) -> bool:
    return _key(path) in _CACHE


def clear_shared_metadata() -> None:
    """Drop process-local startup bytes (primarily for isolated tests)."""
    _CACHE.clear()


def shared_metadata_exists(path: str | Path) -> bool:
    return has_shared_metadata(path) or Path(path).is_file()


def _read_file_bytes(path: Path) -> bytes:
    for attempt in range(_READ_ATTEMPTS):
        try:
            return path.read_bytes()
        except OSError as exc:
            if (exc.errno not in _TRANSIENT_ERRNOS
                    or attempt + 1 == _READ_ATTEMPTS):
                raise
            time.sleep(0.25 * 2**attempt)
    raise AssertionError("finite metadata retry loop did not return")


def read_shared_bytes(path: str | Path, *, consume: bool = False) -> bytes:
    key = _key(path)
    if key in _CACHE:
        return _CACHE.pop(key) if consume else _CACHE[key]
    return _read_file_bytes(Path(path))


def read_shared_text(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    consume: bool = False,
) -> str:
    return read_shared_bytes(path, consume=consume).decode(encoding)


def _build_archive(paths: tuple[str, ...]) -> bytes:
    entries = []
    total = 0
    for value in paths:
        path = Path(value)
        data = _read_file_bytes(path)
        total += len(data)
        if total > _MAX_TOTAL_BYTES:
            raise RuntimeError(
                "shared startup metadata exceeds the 512-MiB safety limit")
        entries.append({
            "path": value,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "compressed": zlib.compress(data, level=1),
        })
    return pickle.dumps({
        "format": _FORMAT,
        "entries": entries,
        "uncompressed_bytes": total,
        "error": None,
    }, protocol=5)


def _error_archive(exc: BaseException) -> bytes:
    return pickle.dumps({
        "format": _FORMAT,
        "entries": [],
        "uncompressed_bytes": 0,
        "error": f"{type(exc).__name__}: {exc}",
    }, protocol=5)


def _broadcast_bytes(payload: bytes | None, *, src: int = 0) -> bytes:
    if not dist.is_initialized():
        if payload is None:
            raise RuntimeError("single-process metadata payload is absent")
        return payload
    device = torch.device("cuda", torch.cuda.current_device())
    rank = dist.get_rank()
    size = torch.tensor(
        [len(payload) if rank == src and payload is not None else 0],
        dtype=torch.int64,
        device=device,
    )
    dist.broadcast(size, src=src)
    length = int(size.item())
    if length <= 0 or length > _MAX_TOTAL_BYTES:
        raise RuntimeError(f"invalid shared metadata archive size: {length}")
    if rank == src:
        assert payload is not None
        host = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        value = host.to(device=device)
    else:
        value = torch.empty(length, dtype=torch.uint8, device=device)
    dist.broadcast(value, src=src)
    return value.cpu().numpy().tobytes()


def broadcast_shared_metadata(
    paths: Iterable[str | Path],
    *,
    src: int = 0,
) -> dict[str, int]:
    """Read required files on ``src`` and cache identical bytes on all ranks."""
    normalized = tuple(dict.fromkeys(_key(path) for path in paths))
    payload = None
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == src:
        try:
            payload = _build_archive(normalized)
        except (OSError, RuntimeError, ValueError) as exc:
            payload = _error_archive(exc)
    archive = pickle.loads(_broadcast_bytes(payload, src=src))
    if not isinstance(archive, dict) or archive.get("format") != _FORMAT:
        raise RuntimeError("shared metadata archive format is invalid")
    if archive.get("error") is not None:
        raise RuntimeError(
            f"rank-zero shared metadata read failed: {archive['error']}")
    received = set()
    compressed_bytes = 0
    for entry in archive.get("entries", ()):
        path = str(entry["path"])
        compressed = entry["compressed"]
        compressed_bytes += len(compressed)
        data = zlib.decompress(compressed)
        if len(data) != int(entry["bytes"]):
            raise RuntimeError(f"shared metadata byte count differs: {path}")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"shared metadata checksum differs: {path}")
        _CACHE[path] = data
        received.add(path)
    if received != set(normalized):
        raise RuntimeError("shared metadata archive path inventory differs")
    return {
        "files": len(received),
        "uncompressed_bytes": int(archive["uncompressed_bytes"]),
        "compressed_bytes": compressed_bytes,
    }
