"""Keep H1 hand and waist unit declarations aligned with the deployment contract."""

import builtins
import copy
import json
from pathlib import Path

import pytest
from test_doctor import server_files, write_json
from test_h1_pose_contract import (
    body_contract,
    bundle_manifest,
    exported_manifest,
    full_stats,
    write_tiny_bundle,
)
from zwam import artifact, doctor, model
from zwam.assets import sha256
from zwam.contract import (
    H1_WAIST_LAYOUT,
    H1_WAIST_SEMANTICS,
    H1_WAIST_UNITS,
    metadata,
    validate_h1_command_contract,
)

SWAPPED_JOINTS = ["waist_joint2.position", "waist_joint1.position", "waist_joint3.position"]
INVALID_DECLARATIONS = [
    pytest.param("position_unit", None, id="missing-position-unit"),
    pytest.param("position_unit", "mm", id="hand-position-in-mm"),
    pytest.param("waist_units", None, id="missing-waist-units"),
    pytest.param("waist_units", ["source_native"] * 3, id="unresolved-native-units"),
    pytest.param("waist_units", ["rad", "m", "rad"], id="swapped-lift-and-pitch-units"),
    pytest.param("waist_units", ["m", "rad"], id="missing-yaw-unit"),
    pytest.param("waist_joint_order", SWAPPED_JOINTS, id="swapped-joints"),
    pytest.param("waist_semantics", ["lift", "yaw", "pitch"], id="swapped-pitch-and-yaw"),
]


def changed_contract(field, value):
    contract = body_contract()
    if value is None:
        contract.pop(field, None)
    else:
        contract[field] = copy.deepcopy(value)
    return contract


def refuse(*_, **__):
    pytest.fail("Invalid H1 units reached a model asset, copy, or GPU operation")


@pytest.mark.parametrize("field,value", INVALID_DECLARATIONS)
def test_shared_contract_rejects_incompatible_units_and_order_without_mutating(field, value):
    contract = changed_contract(field, value)
    original = copy.deepcopy(contract)
    with pytest.raises(ValueError, match="H1"):
        validate_h1_command_contract(contract)
    assert contract == original


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("position_unit", "mm", id="position-unit"),
        pytest.param("waist_units", ["source_native"] * 3, id="unresolved-waist-units"),
        pytest.param("waist_joint_order", SWAPPED_JOINTS, id="joint-order"),
    ],
)
def test_prepare_rejects_units_before_import_weights_or_copy_and_leaves_no_output(
    tmp_path, monkeypatch, field, value
):
    manifest = bundle_manifest(changed_contract(field, value))
    manifest["format"] = "zerith-va-policy-action20-linear-g4d-teacher"
    write_json(tmp_path / "manifest.json", manifest)
    original = (tmp_path / "manifest.json").read_bytes()
    monkeypatch.setattr(artifact, "verified_file", refuse)
    monkeypatch.setattr(artifact.shutil, "copyfile", refuse)
    original_import = builtins.__import__

    def import_without_models(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "safetensors", "zwam_core", "wan", "vggt"}:
            pytest.fail(f"Invalid H1 units caused a model dependency import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_models)
    with pytest.raises(ValueError, match="H1"):
        artifact.prepare(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert (tmp_path / "manifest.json").read_bytes() == original


@pytest.mark.parametrize(
    "entrypoint,field,value",
    [
        pytest.param("policy", "waist_units", None, id="policy-missing-units"),
        pytest.param(
            "policy", "waist_semantics", ["lift", "yaw", "pitch"], id="policy-swapped-semantics"
        ),
        pytest.param("session", "waist_units", ["rad", "m", "rad"], id="session-swapped-units"),
        pytest.param("session", "waist_units", ["m", "rad"], id="session-missing-yaw-unit"),
    ],
)
def test_loaders_reject_units_before_weights_other_asset_reads_or_gpu(
    tmp_path, monkeypatch, entrypoint, field, value
):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, bundle_manifest(changed_contract(field, value)))
    original_read_json = model.read_json

    def only_manifest(path):
        if Path(path) != manifest_path:
            refuse()
        return original_read_json(path)

    monkeypatch.setattr(model, "read_json", only_manifest)
    monkeypatch.setattr(model, "verified_file", refuse)
    monkeypatch.setattr(model, "check_visual_assets", refuse)
    monkeypatch.setattr(model, "load_text_bank", refuse)
    monkeypatch.setattr(model, "move_policy", refuse)
    monkeypatch.setattr(safetensors, "load_file", refuse)
    monkeypatch.setattr(torch, "device", refuse)
    monkeypatch.setattr(torch.cuda, "is_available", refuse)
    with pytest.raises(ValueError, match="H1"):
        if entrypoint == "policy":
            model.load_policy(tmp_path, "cuda:0")
        else:
            model.load_session({"artifact": str(tmp_path), "device": "cuda:0"})


@pytest.mark.parametrize("verify_assets", [False, True], ids=["paths", "hashes"])
def test_doctor_rejects_unresolved_waist_units_before_policy_asset_check(
    tmp_path, monkeypatch, verify_assets
):
    path = server_files(tmp_path)
    manifest_path = tmp_path / "policy/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["normalization_stats"]["command_contract"]["waist_units"] = ["source_native"] * 3
    write_json(manifest_path, manifest)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _: object())
    original_check_asset = doctor.check_asset

    def check_asset(root, name, expected, verify):
        if name == "model.safetensors" and Path(root) == tmp_path / "policy":
            refuse()
        return original_check_asset(root, name, expected, verify)

    monkeypatch.setattr(doctor, "check_asset", check_asset)
    records = doctor.run_checks("server", path, verify_assets=verify_assets)
    policy = next(record for record in records if record["check"] == "policy bundle")
    assert policy["status"] == "FAIL"
    assert "waist" in policy["detail"]


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("waist_units", ["rad", "m", "rad"], id="swapped-units"),
        pytest.param("waist_joint_order", SWAPPED_JOINTS, id="swapped-joints"),
    ],
)
def test_codec_rejects_unit_and_joint_order_mismatch(field, value):
    from zwam_core.actions.h1 import H1Command23Codec

    with pytest.raises(ValueError, match="H1"):
        H1Command23Codec(full_stats(changed_contract(field, value)))


@pytest.mark.parametrize("declare_semantics", [False, True], ids=["legacy-fields", "named-axes"])
def test_correct_units_preserve_evidence_and_unverified_status_when_prepared(
    tmp_path, declare_semantics
):
    from zwam_core.actions.h1 import H1Command23Codec, digest

    contract = body_contract()
    if declare_semantics:
        contract["waist_semantics"] = ["lift", "pitch", "yaw"]
    original = copy.deepcopy(contract)
    validate_h1_command_contract(contract)
    assert contract == original
    assert H1_WAIST_UNITS == ("m", "rad", "rad")
    assert H1_WAIST_LAYOUT == tuple(f"waist_joint{i}.position" for i in (1, 2, 3))
    assert H1_WAIST_SEMANTICS == ("lift", "pitch", "yaw")
    assert metadata()["waist_units"] == ["m", "rad", "rad"]
    stats = full_stats(contract)
    assert H1Command23Codec(stats).stats_fingerprint == digest(stats)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"synthetic copy-only asset")
    write_json(tmp_path / "manifest.json", exported_manifest(stats, sha256(weights)))
    prepared = artifact.prepare(tmp_path, tmp_path / "prepared")
    saved = json.loads((tmp_path / "prepared/manifest.json").read_text())
    assert prepared["normalization_stats"] == saved["normalization_stats"] == stats
    assert saved["normalization_stats"]["command_contract"] == original
    assert saved["normalization_stats"]["command_contract"]["physical_semantics_verified"] is False
    assert (tmp_path / "prepared/model.safetensors").read_bytes() == weights.read_bytes()


def test_manifest_only_unit_relabel_cannot_override_checkpoint_statistics(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from zwam_core.actions.h1 import digest

    manifest = write_tiny_bundle(tmp_path)
    stats = manifest["normalization_stats"]
    stats["command_contract"]["waist_units"] = ["source_native"] * 3
    weights_path = tmp_path / "model.safetensors"
    weights = safetensors.load_file(str(weights_path), device="cpu")
    # Represent a historical checkpoint with an unresolved unit declaration frozen into its hash.
    weights["state_codec.stats_sha256"] = torch.tensor(
        list(bytes.fromhex(digest(stats))), dtype=torch.uint8
    )
    safetensors.save_file(weights, str(weights_path))
    manifest["weights_sha256"] = sha256(weights_path)
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="H1"):
        model.load_policy(tmp_path, "cpu")

    frozen_weights = weights_path.read_bytes()
    stats["command_contract"]["waist_units"] = ["m", "rad", "rad"]
    write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(model, "move_policy", refuse)
    with pytest.raises(ValueError, match="Normalization/checkpoint mismatch: stats_sha256"):
        model.load_policy(tmp_path, "cpu")
    assert weights_path.read_bytes() == frozen_weights
