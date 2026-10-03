"""Check real file identities without importing external encoders or CUDA."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from zvla.posttrain import asset_identity


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    args = SimpleNamespace(
        source=tmp_path / "raw",
        output=tmp_path / "prepared",
        wan_repo=tmp_path / "wan-repo",
        wan_model=tmp_path / "wan-model",
        vggt_repo=tmp_path / "vggt-repo",
        vggt_model=tmp_path / "vggt-model",
    )
    files = {
        "source": ("meta/info.json", "videos/episode.mp4"),
        "wan_repo": (
            "wan/__init__.py",
            "wan/modules/__init__.py",
            "wan/modules/vae2_2.py",
            "wan/modules/t5.py",
            "wan/modules/tokenizers.py",
        ),
        "wan_model": (
            "Wan2.2_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "google/umt5-xxl/spiece.model",
            "google/umt5-xxl/tokenizer.json",
            "google/umt5-xxl/tokenizer_config.json",
        ),
        "vggt_repo": ("vggt/models/vggt.py", "vggt/models/aggregator.py"),
        "vggt_model": ("model.safetensors", "config.json"),
    }
    for root, names in files.items():
        for name in names:
            path = getattr(args, root) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"original")
    monkeypatch.setattr(asset_identity, "_runtime_identity", lambda: {"python": "test"})
    return args


@pytest.mark.parametrize(
    "root,relative",
    (
        ("source", "videos/episode.mp4"),
        ("source", "meta/info.json"),
        ("wan_repo", "wan/modules/vae2_2.py"),
        ("wan_model", "Wan2.2_VAE.pth"),
        ("wan_model", "models_t5_umt5-xxl-enc-bf16.pth"),
        ("wan_model", "google/umt5-xxl/spiece.model"),
        ("wan_model", "google/umt5-xxl/tokenizer_config.json"),
        ("vggt_repo", "vggt/models/aggregator.py"),
        ("vggt_model", "model.safetensors"),
        ("vggt_model", "config.json"),
    ),
)
def test_same_size_same_mtime_content_changes_identity(local_inputs, root, relative, monkeypatch):
    monkeypatch.setenv("ZVLA_FAST_ASSET_VALIDATION", "1")
    before = asset_identity.prepare_asset_identity(local_inputs)
    path = getattr(local_inputs, root) / relative
    stamp = path.stat()
    path.write_bytes(b"modified")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert path.stat().st_size == stamp.st_size
    assert asset_identity.prepare_asset_identity(local_inputs) != before
    assert not local_inputs.output.exists()


def test_file_inventory_changes_and_administrative_cache(local_inputs):
    before = asset_identity.prepare_asset_identity(local_inputs)
    ignored = local_inputs.wan_repo / "wan/__pycache__/module.pyc"
    ignored.parent.mkdir()
    ignored.write_bytes(b"cached")
    assert asset_identity.prepare_asset_identity(local_inputs) == before
    extra = local_inputs.wan_model / "google/umt5-xxl/added_tokens.json"
    extra.write_text("{}")
    assert asset_identity.prepare_asset_identity(local_inputs) != before


@pytest.mark.parametrize("root", ("source", "wan_repo", "wan_model", "vggt_repo", "vggt_model"))
def test_output_inside_inputs_is_rejected_before_writes(local_inputs, root):
    local_inputs.output = getattr(local_inputs, root) / "prepared"
    with pytest.raises(ValueError, match="separate"):
        asset_identity.prepare_asset_identity(local_inputs)
    assert not local_inputs.output.exists()


def test_missing_required_weight_is_rejected(local_inputs):
    (local_inputs.wan_model / "Wan2.2_VAE.pth").unlink()
    with pytest.raises(FileNotFoundError, match="Wan2.2_VAE"):
        asset_identity.prepare_asset_identity(local_inputs)
    assert not local_inputs.output.exists()


def test_directory_symlink_does_not_silently_hide_inputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "episode.mp4").write_bytes(b"video")
    (raw / "videos").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="directory symlink"):
        asset_identity.directory_identity(raw)


def test_file_symlinks_are_identified_by_their_content(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"weights")
    (root / "model.safetensors").symlink_to(blob)
    assert asset_identity.directory_identity(root)[
        "model.safetensors"
    ] == asset_identity.file_identity(blob)


def test_runtime_identity_does_not_initialize_cuda():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from zvla.posttrain.asset_identity import _runtime_identity
def refuse_cuda(*args, **kwargs):
    raise AssertionError("CPU identity check must not initialize CUDA")
torch.cuda._lazy_init = refuse_cuda
identity = _runtime_identity()
assert identity["packages"]["torch"]
assert not torch.cuda.is_initialized()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
