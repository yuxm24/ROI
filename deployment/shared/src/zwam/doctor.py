"""Check local deployment configuration without opening a GPU, network or robot."""

import argparse
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from .config import read_json
from .contract import H1_POSE_FRAME, validate_h1_command_contract
from .encoder_contract import validate_encoder_contract


def json_object(path):
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def required_text(config, key):
    value = config.get(key)
    if not isinstance(value, str) or not value.strip() or "REPLACE_WITH" in value:
        raise ValueError(f"Set {key} in the local configuration")
    return value


def existing_path(value, *, directory=False):
    path = Path(value)
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Missing {'directory' if directory else 'file'}: {path}")
    return path


def digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("Replace checksum placeholders with provider-supplied SHA-256 values")
    return value.lower()


def check_asset(root, relative, expected, verify):
    from .assets import verified_file

    expected = digest(expected)
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Asset path escapes its configured directory")
    existing_path(path)
    if verify:
        verified_file(root, relative, expected)


def run_checks(role, config_path, *, verify_assets=False):
    """Return PASS/WARN/FAIL records; never import vendor or visual source code.

    Relative paths have the same working-directory semantics as the runtime.
    File checks do not establish GPU compatibility, SDK ABI or robot readiness.
    """
    if role not in ("server", "h1"):
        raise ValueError("role must be server or h1")
    records = []

    def check(name, function):
        try:
            result = function()
            records.append(dict(check=name, status="PASS", detail="OK"))
            return result
        except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
            records.append(dict(check=name, status="FAIL", detail=str(error)))
            return None

    def warn(name, detail):
        records.append(dict(check=name, status="WARN", detail=detail))

    config = check("configuration", lambda: json_object(config_path))
    if config is None:
        return records

    if role == "server":
        from .assets import check_visual_assets

        def endpoint():
            host = config.get("bind", "127.0.0.1")
            port = config.get("port", 55558)
            if (
                not isinstance(host, str)
                or not host
                or type(port) is not int
                or not 1 <= port <= 65535
            ):
                raise ValueError("Set a bind address and integer port in 1..65535")
            if (
                host not in ("127.0.0.1", "localhost")
                and len(os.environ.get("ZWAM_API_KEY", "")) < 32
            ):
                raise ValueError(
                    "Non-loopback binding requires ZWAM_API_KEY of at least 32 characters"
                )
            if re.fullmatch(r"cuda(?::\d+)?", required_text(config, "device")) is None:
                raise ValueError("Set a CUDA device, for example cuda:0")

        check("server settings", endpoint)

        def policy():
            root = required_text(config, "artifact")
            manifest = json_object(Path(root) / "manifest.json")
            if manifest.get("format") != "zwam-h1-inference-v1":
                raise ValueError("Prepare a zwam-h1-inference-v1 bundle; see 'zwam prepare --help'")
            for field in ("config", "normalization_stats"):
                if not isinstance(manifest.get(field), dict):
                    raise ValueError(f"Policy manifest is missing {field}")
            validate_h1_command_contract(manifest["normalization_stats"].get("command_contract"))
            validate_encoder_contract(manifest["config"])
            check_asset(root, "model.safetensors", manifest["weights_sha256"], verify_assets)
            digest(manifest["vae_sha256"])
            return manifest

        manifest = check("policy bundle", policy)

        def visual():
            assets = json_object(required_text(config, "visual_assets"))
            check_visual_assets(assets, verify=verify_assets)
            if manifest is not None and digest(manifest["vae_sha256"]) != digest(
                assets["vae_sha256"]
            ):
                raise ValueError("Policy and visual configuration name different VAE checksums")

        check("visual assets", visual)

        def text_bank():
            from .assets import load_text_bank

            root = required_text(config, "text_assets")
            if verify_assets:
                load_text_bank(root)
                return
            text_manifest = json_object(Path(root) / "text_manifest.json")
            for name, key in (
                ("tasks.json", "tasks_sha256"),
                ("text_embeddings.npy", "embeddings_sha256"),
            ):
                check_asset(root, name, text_manifest[key], False)
            tasks = read_json(Path(root) / "tasks.json")
            if (
                not isinstance(tasks, list)
                or not tasks
                or any(not isinstance(t, str) or not t.strip() for t in tasks)
            ):
                raise ValueError("Text bank needs a nonempty list of instruction strings")
            if len(set(tasks)) != len(tasks):
                raise ValueError("Text bank instructions must be unique")

        check("text bank", text_bank)

        def dependencies():
            missing = [
                name
                for name in ("torch", "torchvision", "safetensors", "cv2", "einops", "timm")
                if importlib.util.find_spec(name) is None
            ]
            if missing:
                raise ValueError("Install the inference package; missing: " + ", ".join(missing))

        check("Python package availability", dependencies)
        if not verify_assets:
            warn(
                "checksums",
                "File presence checked; add --verify-assets to hash assets and validate the text bank",
            )
        warn(
            "GPU validation",
            "Not performed; start the service and run offline replay with matching observations",
        )
    else:

        def controller():
            url = urlsplit(required_text(config, "server_url"))
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username
                or url.password
            ):
                raise ValueError("server_url must be an HTTP(S) URL without embedded credentials")
            timeout = config.get("timeout_s", 60)
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                raise ValueError("timeout_s must be finite and positive")
            required_text(config, "prompt")
            if type(config.get("commissioned")) is not bool:
                raise ValueError("commissioned must be true or false")
            if config.get("frame") != H1_POSE_FRAME or config.get("eef_links") != [
                "left_end_effector_link",
                "right_end_effector_link",
            ]:
                raise ValueError("Configure the H1 bodyyaw frame and left/right EEF links")
            for name in ("controller_lock", "zcm_region", "camera_address"):
                required_text(config, name)

        check("H1 settings", controller)

        def sdk():
            root = existing_path(required_text(config, "sdk_root"), directory=True)
            camera_root = existing_path(config.get("camera_sdk_root", root), directory=True)
            candidates = [
                p
                for base in (root, root / "lib")
                for p in base.glob("lib_h1_sdk_python*")
                if p.is_file() and p.suffix in (".so", ".pyd", ".py")
            ]
            if not candidates:
                raise ValueError(
                    "Place the SDK's lib_h1_sdk_python module under sdk_root or sdk_root/lib"
                )
            if not any(
                p.is_file()
                for p in (
                    camera_root / "lib/camera_client.py",
                    camera_root / "lib/camera_client/__init__.py",
                    camera_root / "camera_client.py",
                )
            ) and not any(p.is_file() for p in camera_root.glob("camera_client*.so")):
                raise ValueError(
                    "camera_sdk_root must provide camera_client (or legacy lib.camera_client)"
                )

        check("SDK files", sdk)

        def planner():
            root = existing_path(required_text(config, "planner_source"), directory=True)
            factory = required_text(config, "planner_factory")
            parts = factory.split(":")
            if (
                len(parts) != 2
                or not all(p.isidentifier() for p in parts[0].split("."))
                or not parts[1].isidentifier()
            ):
                raise ValueError("planner_factory must use module:function syntax")
            path = root.joinpath(*parts[0].split("."))
            if not path.with_suffix(".py").is_file() and not (path / "__init__.py").is_file():
                raise ValueError(
                    "Planner factory module missing; see the Zerith-VA README, H1 setup"
                )
            existing_path(root / "processors/fast_ik_backend.py")

        check("IK provider files", planner)
        if not config.get("commissioned"):
            warn(
                "commissioning",
                "Hardware startup remains disabled until the H1 commissioning checklist is complete",
            )
        warn(
            "hardware validation",
            "SDK ABI, native IK, URDF, services and cameras are not loaded or contacted",
        )
    return records


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--role", choices=("server", "h1"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--verify-assets",
        action="store_true",
        help="Hash server model assets; may take several minutes",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable check records")
    args = parser.parse_args(argv)
    if args.verify_assets and args.role != "server":
        parser.error("--verify-assets applies to the server role")
    records = run_checks(args.role, args.config, verify_assets=args.verify_assets)
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        for record in records:
            print(f"{record['status']:4s} {record['check']}: {record['detail']}")
    if any(record["status"] == "FAIL" for record in records):
        raise SystemExit(1)
