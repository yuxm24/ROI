import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from zwam import doctor
from zwam.assets import sha256
from zwam.h1.adapter import validate_planner
from zwam.source import source_sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def server_files(root):
    for directory in (
        "policy",
        "wan",
        "vggt",
        "text",
        "wan_source/wan/modules",
        "vggt_source/vggt/models",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for name in (
        "wan_source/wan/__init__.py",
        "wan_source/wan/modules/__init__.py",
        "wan_source/wan/modules/vae2_2.py",
        "vggt_source/vggt/models/vggt.py",
    ):
        (root / name).write_text(
            "raise AssertionError('Doctor must not import external sources')\n"
        )
    for name in ("policy/model.safetensors", "wan/Wan2.2_VAE.pth", "vggt/model.safetensors"):
        (root / name).write_bytes(b"synthetic asset; file checks only")
    write_json(root / "vggt/config.json", {})
    write_json(
        root / "policy/manifest.json",
        dict(
            format="zwam-h1-inference-v1",
            config={},
            normalization_stats={
                "command_contract": {
                    "pose_frame": "body_yaw_link",
                    "position_unit": "m",
                    "waist_units": ["m", "rad", "rad"],
                    "waist_joint_order": [
                        "waist_joint1.position",
                        "waist_joint2.position",
                        "waist_joint3.position",
                    ],
                    "action_row_offset": 0,
                    "time_alignment_evidence": "synthetic fixture: paired state and action rows",
                    "frame_evidence": "synthetic fixture: matching H1 state and command frame",
                    "physical_semantics_verified": False,
                    "conversion": "none",
                }
            },
            weights_sha256=sha256(root / "policy/model.safetensors"),
            vae_sha256=sha256(root / "wan/Wan2.2_VAE.pth"),
        ),
    )
    write_json(root / "text/tasks.json", ["synthetic task"])
    np.save(root / "text/text_embeddings.npy", np.zeros((1, 64, 4096), np.float16))
    write_json(
        root / "text/text_manifest.json",
        dict(
            tasks_sha256=sha256(root / "text/tasks.json"),
            embeddings_sha256=sha256(root / "text/text_embeddings.npy"),
        ),
    )
    write_json(
        root / "visual.json",
        dict(
            wan_source=str(root / "wan_source"),
            wan_source_sha256=source_sha256(root / "wan_source/wan"),
            wan_weights=str(root / "wan"),
            vae_sha256=sha256(root / "wan/Wan2.2_VAE.pth"),
            vggt_source=str(root / "vggt_source"),
            vggt_source_sha256=source_sha256(root / "vggt_source/vggt"),
            vggt_weights=str(root / "vggt"),
            vggt_files={
                name: sha256(root / "vggt" / name) for name in ("config.json", "model.safetensors")
            },
        ),
    )
    config = dict(
        bind="127.0.0.1",
        port=55558,
        device="cuda:0",
        artifact=str(root / "policy"),
        visual_assets=str(root / "visual.json"),
        text_assets=str(root / "text"),
    )
    write_json(root / "server.json", config)
    return root / "server.json"


def test_doctor_verified_assets_without_model_or_external_imports(tmp_path, monkeypatch):
    path = server_files(tmp_path)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _: object())
    records = doctor.run_checks("server", path, verify_assets=True)
    assert not any(r["status"] == "FAIL" for r in records), records
    assert any(r["check"] == "GPU validation" and r["status"] == "WARN" for r in records)


def test_doctor_reports_bad_checksum(tmp_path, monkeypatch):
    path = server_files(tmp_path)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _: object())
    (tmp_path / "policy/model.safetensors").write_bytes(b"changed")
    records = doctor.run_checks("server", path, verify_assets=True)
    assert any(r["check"] == "policy bundle" and r["status"] == "FAIL" for r in records)


@pytest.mark.parametrize("value", [[], "invalid", None])
def test_doctor_rejects_nonobject_config(tmp_path, value):
    path = tmp_path / "config.json"
    write_json(path, value)
    assert doctor.run_checks("h1", path)[0]["status"] == "FAIL"


def test_doctor_missing_resources_reports_all_failures(tmp_path, capsys):
    path = tmp_path / "config.json"
    write_json(path, {})
    with pytest.raises(SystemExit) as result:
        doctor.main(["--role", "h1", "--config", str(path), "--json"])
    assert result.value.code == 1
    records = json.loads(capsys.readouterr().out)
    assert len([r for r in records if r["status"] == "FAIL"]) == 3


def test_doctor_does_not_construct_sdk_or_planner(tmp_path, monkeypatch):
    sdk = tmp_path / "sdk"
    (sdk / "lib").mkdir(parents=True)
    (sdk / "lib/lib_h1_sdk_python.py").write_text("raise AssertionError('hardware')")
    (sdk / "lib/camera_client.py").write_text("raise AssertionError('camera')")
    planner = tmp_path / "planner"
    (planner / "processors").mkdir(parents=True)
    (planner / "provider.py").write_text("raise AssertionError('planner')")
    (planner / "processors/fast_ik_backend.py").write_text("raise AssertionError('native')")
    path = tmp_path / "h1.json"
    write_json(
        path,
        dict(
            sdk_root=str(sdk),
            planner_source=str(planner),
            planner_factory="provider:build_planner",
            server_url="http://127.0.0.1:55558",
            prompt="synthetic",
            commissioned=False,
            frame="body_yaw_link",
            eef_links=["left_end_effector_link", "right_end_effector_link"],
            controller_lock="/run/lock/zwam-h1-controller.lock",
            zcm_region="/dev/shm/example",
            camera_address="localhost:50051",
        ),
    )
    monkeypatch.setitem(sys.modules, "lib_h1_sdk_python", None)
    records = doctor.run_checks("h1", path)
    assert not any(r["status"] == "FAIL" for r in records), records


def test_doctor_accepts_public_sdk_split_layout(tmp_path):
    sdk = tmp_path / "h1_sdk"
    camera = tmp_path / "camera_sdk_python"
    (sdk / "lib").mkdir(parents=True)
    camera.mkdir()
    (sdk / "lib/lib_h1_sdk_python.so").write_bytes(b"test fixture")
    (camera / "camera_client.cpython-310-x86_64-linux-gnu.so").write_bytes(b"test fixture")
    planner = tmp_path / "planner"
    (planner / "processors").mkdir(parents=True)
    (planner / "provider.py").write_text("", encoding="utf-8")
    (planner / "processors/fast_ik_backend.py").write_text("", encoding="utf-8")
    config = tmp_path / "h1.json"
    write_json(
        config,
        dict(
            sdk_root=str(sdk),
            camera_sdk_root=str(camera),
            planner_source=str(planner),
            planner_factory="provider:build_planner",
            server_url="http://127.0.0.1:55558",
            prompt="synthetic",
            commissioned=False,
            frame="body_yaw_link",
            eef_links=["left_end_effector_link", "right_end_effector_link"],
            controller_lock="/run/lock/zwam-h1-controller.lock",
            zcm_region="/dev/shm/example",
            camera_address="localhost:50051",
        ),
    )
    records = doctor.run_checks("h1", config)
    assert not any(r["status"] == "FAIL" for r in records), records


def valid_planner():
    return SimpleNamespace(
        lower=np.full(19, -2.0),
        upper=np.full(19, 2.0),
        arm=object(),
        arm_control_indices=np.arange(3, 17),
        root_from_body=np.eye(4),
        body_eef_poses=lambda _: None,
        eef_names=("left_end_effector_link", "right_end_effector_link"),
    )


def test_planner_contract_accepts_arms_and_optional_head():
    p = valid_planner()
    validate_planner(p)
    p.arm_control_indices = np.arange(3, 19)
    validate_planner(p)


@pytest.mark.parametrize(
    "indices", [np.arange(19), np.arange(3, 16), [3] * 14, np.arange(3, 17, dtype=float)]
)
def test_planner_contract_rejects_bad_mapping(indices):
    p = valid_planner()
    p.arm_control_indices = indices
    with pytest.raises(ValueError, match="IK indices"):
        validate_planner(p)


@pytest.mark.parametrize(
    "transform", [np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan), np.diag([-1, 1, 1, 1])]
)
def test_planner_contract_rejects_bad_transform(transform):
    p = valid_planner()
    p.root_from_body = transform
    with pytest.raises(ValueError, match="rigid transform"):
        validate_planner(p)
