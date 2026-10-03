"""Content identities for repeatable public demonstration preparation.

Checks read local files without importing external model code or loading weights.
Paths identify files within each bundle; moving an unchanged bundle is allowed.
"""

from __future__ import annotations

import hashlib
import os
import platform
from importlib import metadata
from pathlib import Path

_IGNORED_DIRECTORIES = {
    ".git",
    ".cache",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}


def _files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Preparation input directory is missing: {root}")
    result = []
    for directory, folders, names in os.walk(root):
        folders[:] = sorted(name for name in folders if name not in _IGNORED_DIRECTORIES)
        for name in folders:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Preparation input contains a directory symlink: {path}")
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix not in (".pyc", ".pyo"):
                if not path.is_file():
                    raise ValueError(f"Preparation input is not a regular file: {path}")
                result.append(path)
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def file_identity(path):
    """Hash bytes, rejecting a file that changes while it is being read."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    after = path.stat()
    attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in attributes):
        raise ValueError(f"Preparation input changed while hashing: {path}")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def directory_identity(root):
    root = Path(root)
    paths = _files(root)
    if not paths:
        raise ValueError(f"Preparation input directory is empty: {root}")
    result = {path.relative_to(root).as_posix(): file_identity(path) for path in paths}
    if [path.relative_to(root).as_posix() for path in _files(root)] != list(result):
        raise ValueError(f"Preparation input inventory changed while hashing: {root}")
    return result


def _require_files(root, names):
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required preparation asset is missing: {path}")


def _runtime_identity():
    import torch

    packages = (
        "torch",
        "numpy",
        "Pillow",
        "opencv-python",
        "opencv-python-headless",
        "pyarrow",
        "av",
        "h5py",
        "protobuf",
        "transformers",
        "tokenizers",
        "sentencepiece",
        "ftfy",
        "einops",
        "safetensors",
        "huggingface-hub",
        "nvidia-cudnn-cu12",
        "nvidia-cudnn-cu13",
    )
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "packages": versions,
        "torch_cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
            )
        },
    }


def prepare_asset_identity(args):
    """Identify raw files, consumed encoder assets, code and software settings.

    Raw sources and encoder model directories must remain unchanged for the
    duration of a run. Administrative caches are excluded; model content hashes
    are always checked, regardless of any fast-validation environment override.
    """
    source = Path(args.source).resolve()
    wan_repo = Path(args.wan_repo).resolve()
    wan_model = Path(args.wan_model).resolve()
    vggt_repo = Path(args.vggt_repo).resolve()
    vggt_model = Path(args.vggt_model).resolve()
    output = Path(args.output).expanduser().resolve()
    roots = (
        source,
        wan_repo,
        wan_model,
        vggt_repo,
        vggt_model,
        (wan_repo / "wan").resolve(),
        (wan_model / "google/umt5-xxl").resolve(),
        (vggt_repo / "vggt").resolve(),
    )
    for root in roots:
        if output == root or root in output.parents or output in root.parents:
            raise ValueError(
                "Preparation output must be separate from source and encoder asset directories"
            )

    _require_files(
        wan_repo / "wan",
        (
            "__init__.py",
            "modules/__init__.py",
            "modules/vae2_2.py",
            "modules/t5.py",
            "modules/tokenizers.py",
        ),
    )
    _require_files(vggt_repo / "vggt", ("models/vggt.py", "models/aggregator.py"))
    wan_weights = ("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth")
    _require_files(wan_model, wan_weights)
    _require_files(wan_model / "google/umt5-xxl", ("spiece.model", "tokenizer.json"))
    if not any(
        (vggt_model / name).is_file()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    ):
        raise FileNotFoundError(f"VGGT model directory has no local weights: {vggt_model}")

    simulation = Path(__file__).resolve().parents[2]
    project = {}
    for relative in ("zvla/posttrain", "zvla/vision", "zvla/actions", "zvla_runtime"):
        root = simulation / relative
        project.update(
            {
                relative + "/" + path.relative_to(root).as_posix(): file_identity(path)
                for path in _files(root)
                if path.suffix == ".py"
            }
        )
    return {
        "format": "zvla-prepare-input-assets-v1",
        "source": directory_identity(source),
        "wan_source": directory_identity(wan_repo / "wan"),
        "wan_weights": {name: file_identity(wan_model / name) for name in wan_weights},
        "tokenizer": directory_identity(wan_model / "google/umt5-xxl"),
        "vggt_source": directory_identity(vggt_repo / "vggt"),
        "vggt_model": directory_identity(vggt_model),
        "project_code": project,
        "runtime": _runtime_identity(),
    }
