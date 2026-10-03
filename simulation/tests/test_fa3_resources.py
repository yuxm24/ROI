"""CPU checks for the pinned FA3 resources in source and installed packages."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
import torch

SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "fa2_custom_mask"


def load_fa3(package):
    spec = importlib.util.spec_from_file_location(
        "fa3_resource_test", package / "attention_fa3.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def installed_package(tmp_path, monkeypatch):
    monkeypatch.delenv("ZVLA_FA3_CUTLASS_ROOT", raising=False)

    def cuda_access(*args, **kwargs):
        pytest.fail("resource validation must not access CUDA")

    for name in ("is_available", "current_device", "get_device_name",
                 "get_device_capability"):
        monkeypatch.setattr(torch.cuda, name, cuda_access)
    package = tmp_path / "site-packages" / "fa2_custom_mask"
    shutil.copytree(
        SOURCE_PACKAGE, package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".build_fa3"),
    )
    return package


def test_source_and_installed_resources_match_original_pin(installed_package):
    source = load_fa3(SOURCE_PACKAGE)
    installed = load_fa3(installed_package)
    assert source.flash_attention3_source_fingerprint() == source._PINNED_SOURCE_FINGERPRINT
    assert installed._CUTLASS_ROOT == installed_package / "vendor" / "flash_attn_cutlass"
    assert installed.flash_attention3_source_fingerprint() == source._PINNED_SOURCE_FINGERPRINT
    assert installed.flash_attention3_build_fingerprint() == source.flash_attention3_build_fingerprint()
    assert all(path.is_file() for _, path in installed._fingerprinted_files())


def test_matching_cutlass_override_preserves_identity(installed_package, tmp_path, monkeypatch):
    bundled = installed_package / "vendor" / "flash_attn_cutlass"
    external = tmp_path / "external-cutlass"
    shutil.move(str(bundled), external)
    monkeypatch.setenv("ZVLA_FA3_CUTLASS_ROOT", str(external))
    module = load_fa3(installed_package)
    assert module._CUTLASS_ROOT == external
    assert module.flash_attention3_source_fingerprint() == module._PINNED_SOURCE_FINGERPRINT


@pytest.mark.parametrize("missing", [
    "vendor/flash_attn_hopper",
    "vendor/flash_attn_cutlass/include",
    "vendor/flash_attn_cutlass/LICENSE.txt",
    "fa3_bindings.cpp",
])
def test_missing_resources_fail_closed(installed_package, missing):
    path = installed_package / missing
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    module = load_fa3(installed_package)
    with pytest.raises(FileNotFoundError, match="build resources are incomplete") as error:
        module.flash_attention3_source_fingerprint()
    assert "fetch_fa3_build_dependencies.py" not in str(error.value)


@pytest.mark.parametrize("changed", [
    "fa3_fwd.cu",
    "vendor/flash_attn_hopper/mask.h",
    "vendor/flash_attn_cutlass/include/cute/tensor.hpp",
])
def test_compiler_input_drift_is_rejected(installed_package, changed):
    path = installed_package / changed
    path.write_bytes(path.read_bytes() + b"\n// Changed compiler input.\n")
    module = load_fa3(installed_package)
    with pytest.raises(RuntimeError, match="source tree differs from the pinned"):
        module.flash_attention3_source_fingerprint()


def test_new_attribution_preserves_compiler_identity(installed_package):
    hopper = installed_package / "vendor" / "flash_attn_hopper"
    (hopper / "AUTHORS").write_text("Upstream contributor attribution.\n")
    (hopper / "NOTICE").write_text("Additional distribution notices.\n")
    (installed_package / "NOTICE").write_text("Package distribution notices.\n")
    module = load_fa3(installed_package)
    assert module.flash_attention3_source_fingerprint() == module._PINNED_SOURCE_FINGERPRINT


def test_build_identity_includes_runtime_abi(installed_package, monkeypatch):
    module = load_fa3(installed_package)
    original = module.flash_attention3_build_fingerprint()
    monkeypatch.setattr(torch, "__version__", str(torch.__version__) + "+different-build")
    module.flash_attention3_build_fingerprint.cache_clear()
    assert module.flash_attention3_build_fingerprint() != original
