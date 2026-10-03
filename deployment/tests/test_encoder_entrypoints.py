"""Reject incompatible frozen encoder interfaces before deployment side effects."""

import builtins
import copy
import json
from pathlib import Path

import pytest
from test_doctor import server_files, write_json
from test_encoder_contract import SUPPORTED_INTERFACE
from test_h1_pose_contract import body_contract, bundle_manifest, write_tiny_bundle
from zwam import artifact, doctor, model

INCOMPATIBLE_INTERFACES = [
    pytest.param({"text_dim": 2048}, "text_dim", id="text-width"),
    pytest.param({"text_prefix_length": 32}, "text_prefix_length", id="text-prefix"),
    pytest.param({"latent_channels": 16}, "latent_channels", id="vae-channels"),
    pytest.param({"canonical_latent_height": 24}, "canonical_latent_height", id="vae-height"),
    pytest.param({"canonical_latent_width": 16}, "canonical_latent_width", id="vae-width"),
    pytest.param({"geometry_feature_dim": 1024}, "geometry_feature_dim", id="geometry-width"),
    pytest.param(
        {"geometry_tokens_per_camera": 64}, "geometry_tokens_per_camera", id="geometry-grid"
    ),
    pytest.param({"camera_slots": 4}, "camera_slots", id="camera-slots"),
    pytest.param({"max_active_camera_slots": 2}, "max_active_camera_slots", id="camera-capacity"),
    pytest.param({"patch_size": 1}, "patch_size", id="patch-size"),
    pytest.param({"text_length": 63}, "text_length", id="short-text-context"),
    pytest.param({"text_dim": 4096.0}, "text_dim", id="non-integer-size"),
    pytest.param(None, "config", id="missing-config"),
]


def refuse(*_, **__):
    pytest.fail("An incompatible encoder interface reached model assets, imports or a GPU")


@pytest.mark.parametrize("entrypoint", ["prepare", "policy", "session"])
@pytest.mark.parametrize("config,field", INCOMPATIBLE_INTERFACES)
def test_incompatible_interface_fails_before_import_weights_gpu_or_output(
    tmp_path, monkeypatch, entrypoint, config, field
):
    manifest = bundle_manifest(body_contract())
    if config is None:
        manifest.pop("config")
    else:
        manifest["config"] = copy.deepcopy(config)
    if entrypoint == "prepare":
        manifest["format"] = "fam-wam-policy-v5.2-action20-linear-g4d-teacher"
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    before = manifest_path.read_bytes()
    original_read = model.read_json

    def read_manifest_only(path):
        if Path(path) != manifest_path:
            refuse()
        return original_read(path)

    monkeypatch.setattr(model, "read_json", read_manifest_only)
    monkeypatch.setattr(model, "verified_file", refuse)
    monkeypatch.setattr(model, "check_visual_assets", refuse)
    monkeypatch.setattr(model, "load_text_bank", refuse)
    monkeypatch.setattr(model, "move_policy", refuse)
    monkeypatch.setattr(artifact, "verified_file", refuse)
    monkeypatch.setattr(artifact.shutil, "copyfile", refuse)
    original_import = builtins.__import__

    def import_without_models(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "safetensors", "zwam_core", "wan", "vggt"}:
            refuse()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_models)
    with pytest.raises(ValueError, match=f"Encoder interface.*{field}"):
        if entrypoint == "prepare":
            artifact.prepare(tmp_path, tmp_path / "prepared")
        elif entrypoint == "policy":
            model.load_policy(tmp_path, "cuda:0")
        else:
            model.load_session({"artifact": str(tmp_path), "device": "cuda:0"})
    assert manifest_path.read_bytes() == before
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("verify_assets", [False, True], ids=["paths", "hashes"])
@pytest.mark.parametrize("config,field", INCOMPATIBLE_INTERFACES[:-1])
def test_doctor_rejects_interface_before_policy_assets_in_both_modes(
    tmp_path, monkeypatch, verify_assets, config, field
):
    path = server_files(tmp_path)
    manifest_path = tmp_path / "policy/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"] = copy.deepcopy(config)
    write_json(manifest_path, manifest)
    original_check_asset = doctor.check_asset

    def check_asset(root, name, expected, verify):
        if name == "model.safetensors" and Path(root) == tmp_path / "policy":
            refuse()
        return original_check_asset(root, name, expected, verify)

    monkeypatch.setattr(doctor, "check_asset", check_asset)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _: object())
    records = doctor.run_checks("server", path, verify_assets=verify_assets)
    record = next(record for record in records if record["check"] == "policy bundle")
    assert record["status"] == "FAIL"
    assert "Encoder interface" in record["detail"]
    assert field in record["detail"]


@pytest.mark.parametrize("omit_defaults", [False, True], ids=["explicit", "defaulted"])
def test_supported_encoder_sizes_keep_small_internal_models_and_logical_text_padding(
    tmp_path, omit_defaults
):
    from zwam.encoder_contract import validate_encoder_contract

    manifest = write_tiny_bundle(tmp_path)
    # A longer logical text context adds zero suffix rows to the compact 64-row bank.
    manifest["config"]["text_length"] = 128
    if omit_defaults:
        for field in SUPPORTED_INTERFACE:
            manifest["config"].pop(field)
    write_json(tmp_path / "manifest.json", manifest)
    original = copy.deepcopy(manifest)
    validate_encoder_contract(manifest["config"])
    policy, loaded = model.load_policy(tmp_path, "cpu")
    assert loaded == manifest == original
    assert policy.config.text_length == 128
    assert policy.config.student_dim == policy.config.reader_dim == 32
    assert policy.config.student_layers == 4
    assert policy.config.reader_layers == 2
    assert policy.config.camera_rope_bases == (0, 4, 8)
    assert next(policy.parameters()).device.type == "cpu"
    # The model constructs the correct frozen VAE/three-camera output grid.
    assert tuple(policy.world.fixed_future_noise.shape) == (1, 4, 48, 42, 14)
