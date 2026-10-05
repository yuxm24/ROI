"""Reject incompatible H1 frames before preparing or loading policy assets."""

import builtins
import copy
import json

import pytest
from test_doctor import server_files, write_json
from zwam import artifact, doctor, model
from zwam.assets import sha256
from zwam.contract import CAMERAS, H1_POSE_FRAME, metadata, validate_h1_pose_contract


def body_contract():
    return {
        "pose_frame": "body_yaw_link",
        "position_unit": "m",
        "waist_units": ["m", "rad", "rad"],
        "waist_joint_order": [f"waist_joint{i}.position" for i in (1, 2, 3)],
        "action_row_offset": 0,
        "time_alignment_evidence": "synthetic fixture: paired state and action rows",
        "frame_evidence": "  synthetic fixture: source states and targets use body_yaw_link  ",
        "physical_semantics_verified": False,
        "conversion": "none",
    }


INVALID_CONTRACTS = [
    pytest.param(None, id="missing-contract"),
    pytest.param({}, id="empty-contract"),
    pytest.param({"frame_evidence": "declared source"}, id="missing-frame"),
    pytest.param({"pose_frame": None, "frame_evidence": "declared source"}, id="null-frame"),
    pytest.param({"pose_frame": "", "frame_evidence": "declared source"}, id="empty-frame"),
    pytest.param(
        {"pose_frame": "source_native_shared_action_state_frame", "frame_evidence": "native"},
        id="source-native-frame",
    ),
    pytest.param({"pose_frame": "world", "frame_evidence": "declared source"}, id="other-frame"),
    pytest.param(
        {"pose_frame": " body_yaw_link", "frame_evidence": "declared source"},
        id="frame-must-match-exactly",
    ),
    pytest.param({"pose_frame": "body_yaw_link"}, id="missing-evidence"),
    pytest.param({"pose_frame": "body_yaw_link", "frame_evidence": None}, id="null-evidence"),
    pytest.param({"pose_frame": "body_yaw_link", "frame_evidence": ""}, id="empty-evidence"),
    pytest.param({"pose_frame": "body_yaw_link", "frame_evidence": " \n\t"}, id="blank-evidence"),
    pytest.param(
        {"pose_frame": "body_yaw_link", "frame_evidence": ["source"]}, id="nontext-evidence"
    ),
]


def refuse(*_, **__):
    pytest.fail("An incompatible frame reached an asset loader or GPU operation")


def bundle_manifest(contract):
    # Invalid-frame cases need only a manifest: no model file should be inspected.
    return {
        "format": "zwam-h1-inference-v1",
        "config": {},
        "normalization_stats": {"command_contract": copy.deepcopy(contract)},
    }


@pytest.mark.parametrize("contract", INVALID_CONTRACTS)
def test_prepare_rejects_frame_before_weights_and_creates_no_destination(
    tmp_path, monkeypatch, contract
):
    source, destination = tmp_path / "source", tmp_path / "destination"
    manifest = bundle_manifest(contract)
    manifest["format"] = "zerith-va-policy-action20-linear-g4d-teacher"
    write_json(source / "manifest.json", manifest)
    monkeypatch.setattr(artifact, "verified_file", refuse)
    before = (source / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="H1 pose"):
        artifact.prepare(source, destination)

    assert not destination.exists()
    assert (source / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("loader", ["policy", "session"])
@pytest.mark.parametrize("contract", INVALID_CONTRACTS)
def test_loaders_reject_frame_before_gpu_weights_or_external_assets(
    tmp_path, monkeypatch, contract, loader
):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    write_json(tmp_path / "manifest.json", bundle_manifest(contract))
    monkeypatch.setattr(torch.cuda, "is_available", refuse)
    monkeypatch.setattr(torch, "device", refuse)
    monkeypatch.setattr(safetensors, "load_file", refuse)
    monkeypatch.setattr(model, "verified_file", refuse)
    monkeypatch.setattr(model, "check_visual_assets", refuse)
    monkeypatch.setattr(model, "load_text_bank", refuse)
    monkeypatch.setattr(model, "move_policy", refuse)

    with pytest.raises(ValueError, match="H1 pose"):
        if loader == "policy":
            model.load_policy(tmp_path, "cuda:0")
        else:
            # No visual/text paths: checking either before the frame would also fail.
            model.load_session({"artifact": str(tmp_path), "device": "cuda:0"})


@pytest.mark.parametrize("entrypoint", ["prepare", "policy", "session"])
def test_wrong_frame_requires_no_torch_model_or_vendor_imports(tmp_path, monkeypatch, entrypoint):
    manifest = bundle_manifest({"pose_frame": "world", "frame_evidence": "declared source"})
    if entrypoint == "prepare":
        manifest["format"] = "zerith-va-policy-action20-linear-g4d-teacher"
    write_json(tmp_path / "manifest.json", manifest)
    original_import = builtins.__import__

    def import_without_models(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "safetensors", "zwam_core", "wan", "vggt"}:
            pytest.fail(f"Wrong frame caused a model dependency import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_models)
    with pytest.raises(ValueError, match="H1 pose_frame"):
        if entrypoint == "prepare":
            artifact.prepare(tmp_path, tmp_path / "prepared")
        elif entrypoint == "policy":
            model.load_policy(tmp_path, "cuda:0")
        else:
            model.load_session({"artifact": str(tmp_path), "device": "cuda:0"})
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("verify_assets", [False, True], ids=["paths", "hashes"])
@pytest.mark.parametrize("contract", INVALID_CONTRACTS)
def test_doctor_reports_incompatible_frame_in_both_asset_modes(
    tmp_path, monkeypatch, contract, verify_assets
):
    path = server_files(tmp_path)
    manifest_path = tmp_path / "policy/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["normalization_stats"]["command_contract"] = copy.deepcopy(contract)
    write_json(manifest_path, manifest)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _: object())
    original_check_asset = doctor.check_asset

    def check_asset(root, name, expected, verify):
        if name == "model.safetensors" and str(root) == str(tmp_path / "policy"):
            refuse()
        return original_check_asset(root, name, expected, verify)

    monkeypatch.setattr(doctor, "check_asset", check_asset)
    records = doctor.run_checks("server", path, verify_assets=verify_assets)
    policy = next(record for record in records if record["check"] == "policy bundle")
    assert policy["status"] == "FAIL"
    assert "H1 pose" in policy["detail"]


def tiny_config():
    from zwam_core.models.zerith_va.config import ZerithVAConfig

    # Keep the real encoder interface with a small internal model for CPU loading.
    return ZerithVAConfig.tiny(
        latent_channels=48,
        text_dim=4096,
        text_length=64,
        text_prefix_length=64,
        canonical_latent_height=42,
        canonical_latent_width=14,
        geometry_feature_dim=2048,
        geometry_tokens_per_camera=49,
        camera_slots=3,
        max_active_camera_slots=3,
        patch_size=2,
        action_flow_steps=10,
    )


def full_stats(contract):
    from zwam_core.actions.h1 import H1_FORMAT, WAIST_LAYOUT, digest
    from zwam_core.actions.mixed_native7 import MINMAX_FORMAT, MINMAX_NORMALIZATION, NATIVE7_LAYOUT
    from zwam_core.actions.state_trajectory import (
        ACTION20_COORDINATE_CONVENTION,
        INTERNAL_STATE_ACTION_LAYOUT,
        STATE_TRAJECTORY_LAYOUT,
    )

    state_min, state_max = [-1.0] * 20, [1.0] * 20
    action_min, action_max = [-1.0] * 20, [1.0] * 20
    for column in (9, 19):
        state_min[column] = action_min[column] = 0.0
        action_max[column] = 1.5
    arms = {
        "format": MINMAX_FORMAT,
        "normalization": MINMAX_NORMALIZATION,
        "state_layout": list(STATE_TRAJECTORY_LAYOUT),
        "action_layout": list(INTERNAL_STATE_ACTION_LAYOUT),
        "native7_layout": list(NATIVE7_LAYOUT),
        "coordinate_convention": ACTION20_COORDINATE_CONVENTION,
        "embodiments": ["h1"],
        "state_min": [state_min],
        "state_max": [state_max],
        "action_min": [action_min],
        "action_max": [action_max],
        "state_stat_valid": [[True] * 20],
        "action_stat_valid": [[True] * 20],
        "state_degenerate": [[False] * 20],
        "action_degenerate": [[False] * 20],
    }
    return {
        "format": H1_FORMAT,
        "embodiments": ["h1"],
        "waist_layout": WAIST_LAYOUT,
        "arms_minmax": arms,
        "arms_minmax_sha256": digest(arms),
        "command_contract": copy.deepcopy(contract),
        "state_waist_min": [[-1.0] * 3],
        "state_waist_max": [[1.0] * 3],
        "action_waist_min": [[-1.0] * 3],
        "action_waist_max": [[1.0] * 3],
    }


def exported_manifest(stats, weights_digest):
    return {
        "format": "zerith-va-policy-action20-linear-g4d-teacher",
        "config": tiny_config().as_dict(),
        "normalization_stats": stats,
        "action_schema": {
            "width": 23,
            "execute_rows": [0, 1, 2, 3],
            "primitive_dt": 1 / 30,
            "video_dt": 4 / 30,
        },
        "camera_schema": {
            "profiles": {
                "fam_wam_raw_3canonical_view_v1": {
                    "source_camera_names": list(CAMERAS),
                    "source_to_canonical": [0, 1, 2],
                }
            }
        },
        "weights": {"sha256": weights_digest},
        "vae_encoder": {"sha256": "a" * 64},
    }


@pytest.mark.parametrize("contract", INVALID_CONTRACTS)
def test_codec_rejects_missing_frame_evidence_or_incompatible_frame(contract):
    from zwam_core.actions.h1 import H1Command23Codec

    if isinstance(contract, dict):
        declared = body_contract()
        declared.pop("pose_frame")
        declared.pop("frame_evidence")
        declared.update(contract)
    else:
        declared = contract
    stats = full_stats(declared)
    original = copy.deepcopy(stats)
    with pytest.raises(ValueError, match="H1"):
        H1Command23Codec(stats)
    assert stats == original


@pytest.mark.parametrize("legacy_metadata", [False, True], ids=["canonical", "legacy-metadata"])
def test_body_frame_preserves_unverified_physical_status_and_source_evidence(
    tmp_path, legacy_metadata
):
    from zwam_core.actions.h1 import H1Command23Codec, digest

    contract = body_contract()
    original = copy.deepcopy(contract)
    validate_h1_pose_contract(contract)
    assert H1_POSE_FRAME == metadata()["frame"] == "body_yaw_link"
    assert contract == original
    stats = full_stats(contract)
    codec = H1Command23Codec(stats)
    assert codec.stats_fingerprint == digest(stats)

    source = tmp_path / "source"
    source.mkdir()
    weights = source / "model.safetensors"
    weights.write_bytes(b"synthetic copy-only asset")
    manifest = exported_manifest(stats, sha256(weights))
    if legacy_metadata:
        from zwam_core.models.zerith_va.config import LEGACY_TEACHER_ARCHITECTURE_VERSION

        manifest["format"] = artifact.LEGACY_EXPORT_FORMAT
        manifest["config"]["architecture_version"] = LEGACY_TEACHER_ARCHITECTURE_VERSION
    write_json(source / "manifest.json", manifest)
    prepared = artifact.prepare(source, tmp_path / "prepared")
    saved = json.loads((tmp_path / "prepared/manifest.json").read_text())

    assert prepared["normalization_stats"] == saved["normalization_stats"] == stats
    assert saved["normalization_stats"]["command_contract"] == original
    assert saved["normalization_stats"]["command_contract"]["physical_semantics_verified"] is False
    assert saved["config"]["architecture_version"] == tiny_config().architecture_version
    assert (tmp_path / "prepared/model.safetensors").read_bytes() == weights.read_bytes()


def write_tiny_bundle(root, *, legacy_native=False):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from zwam_core.actions.h1 import SOURCE_NATIVE_CONTRACT, H1Command23Codec, digest
    from zwam_core.models.zerith_va.runtime import ZerithVATeacherPolicy

    config = tiny_config()
    stats = full_stats(body_contract())
    policy = ZerithVATeacherPolicy(config)
    policy.state_codec = H1Command23Codec(stats)
    if legacy_native:
        # Model a historical checkpoint without asking the new codec to accept its frame.
        stats["command_contract"] = copy.deepcopy(SOURCE_NATIVE_CONTRACT)
        policy.state_codec.stats_sha256.copy_(
            torch.tensor(list(bytes.fromhex(digest(stats))), dtype=torch.uint8)
        )
    weights = root / "model.safetensors"
    safetensors.save_file(
        {key: value.contiguous().clone() for key, value in policy.state_dict().items()},
        str(weights),
    )
    manifest = {
        "format": "zwam-h1-inference-v1",
        "config": config.as_dict(),
        "normalization_stats": stats,
        "weights_sha256": sha256(weights),
        "vae_sha256": "a" * 64,
    }
    write_json(root / "manifest.json", manifest)
    return json.loads((root / "manifest.json").read_text())


@pytest.mark.parametrize("legacy_metadata", [False, True], ids=["canonical", "legacy-metadata"])
def test_body_frame_full_checkpoint_loads_on_cpu_and_keeps_original_contract(
    tmp_path, legacy_metadata
):
    manifest = write_tiny_bundle(tmp_path)
    if legacy_metadata:
        from zwam_core.models.zerith_va.config import LEGACY_TEACHER_ARCHITECTURE_VERSION

        manifest["config"]["architecture_version"] = LEGACY_TEACHER_ARCHITECTURE_VERSION
        write_json(tmp_path / "manifest.json", manifest)
    policy, loaded = model.load_policy(tmp_path, "cpu")
    assert loaded == manifest
    assert next(policy.parameters()).device.type == "cpu"
    assert policy.config.architecture_version == tiny_config().architecture_version
    assert loaded["normalization_stats"]["command_contract"] == body_contract()


def test_manifest_only_native_frame_relabel_cannot_override_checkpoint_stats(tmp_path, monkeypatch):
    manifest = write_tiny_bundle(tmp_path, legacy_native=True)
    weights_before = (tmp_path / "model.safetensors").read_bytes()
    with pytest.raises(ValueError, match="H1 pose_frame"):
        model.load_policy(tmp_path, "cpu")

    # A new label/evidence alone does not convert the model's original training frame.
    manifest["normalization_stats"]["command_contract"]["pose_frame"] = "body_yaw_link"
    manifest["normalization_stats"]["command_contract"]["waist_units"] = ["m", "rad", "rad"]
    manifest["normalization_stats"]["command_contract"]["frame_evidence"] = "changed label only"
    write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(model, "move_policy", refuse)
    with pytest.raises(ValueError, match="Normalization/checkpoint mismatch: stats_sha256"):
        model.load_policy(tmp_path, "cpu")
    assert (tmp_path / "model.safetensors").read_bytes() == weights_before
