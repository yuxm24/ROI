"""Validate and load explicit local model assets."""

import hashlib
import re
from pathlib import Path

from .config import read_json


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def valid_digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("Expected a provider-supplied SHA-256 digest")
    return value.lower()


def check_visual_assets(assets, *, verify=True):
    """Validate the same source and asset contract for doctor and runtime."""
    from .source import source_sha256

    if not isinstance(assets, dict):
        raise ValueError("Visual assets configuration must be a JSON object")
    files = assets.get("vggt_files")
    if not isinstance(files, dict) or set(files) != {"config.json", "model.safetensors"}:
        raise ValueError("vggt_files must contain exactly config.json and model.safetensors")
    for name, package, required in (
        ("wan", "wan", ("__init__.py", "modules/__init__.py", "modules/vae2_2.py")),
        ("vggt", "vggt", ("models/vggt.py",)),
    ):
        root = Path(assets[f"{name}_source"]).resolve(strict=True) / package
        for relative in required:
            if not (root / relative).is_file():
                raise ValueError(f"Missing {name} source file: {relative}")
        expected = valid_digest(assets[f"{name}_source_sha256"])
        if verify and source_sha256(root) != expected:
            raise ValueError(f"{name} source checksum mismatch")
    selected = [(assets["wan_weights"], "Wan2.2_VAE.pth", assets["vae_sha256"])]
    selected.extend((assets["vggt_weights"], name, digest) for name, digest in files.items())
    verified = {}
    for root, name, expected in selected:
        expected = valid_digest(expected)
        root = Path(root).resolve(strict=True)
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Visual asset must be a file inside its configured root")
        verified[name] = verified_file(root, name, expected) if verify else path
    return verified


def verified_file(root, relative, expected):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Asset must be a file inside its configured root")
    if sha256(path) != valid_digest(expected):
        raise ValueError(f"Asset checksum mismatch: {relative}")
    return path


def load_text_bank(root):
    import numpy as np

    root = Path(root)
    manifest = read_json(root / "text_manifest.json")
    tasks_path = verified_file(root, "tasks.json", manifest["tasks_sha256"])
    bank_path = verified_file(root, "text_embeddings.npy", manifest["embeddings_sha256"])
    tasks = read_json(tasks_path)
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(t, str) or not t.strip() for t in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        raise ValueError("Unique nonempty tasks required")
    bank = np.load(bank_path, allow_pickle=False, mmap_mode="r")
    if (
        bank.shape != (len(tasks), 64, 4096)
        or bank.dtype != np.float16
        or not np.isfinite(bank).all()
    ):
        raise ValueError("Expected finite FP16 [tasks,64,4096] frozen text embeddings")
    lookup = {task: i for i, task in enumerate(tasks)}
    return tasks, lambda prompt: bank[lookup[prompt]]
