"""Prepare a Zerith VA policy bundle from an exported deployment artifact."""

import argparse
import json
import shutil
from dataclasses import fields
from pathlib import Path

from .assets import read_json, sha256, verified_file
from .contract import validate_h1_command_contract
from .encoder_contract import validate_encoder_contract

ZERITH_VA_EXPORT_FORMAT = "zerith-va-policy-action20-linear-g4d-teacher"
LEGACY_EXPORT_FORMAT = "fam-wam-policy-v5.2-action20-linear-g4d-teacher"


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    old = read_json(source / "manifest.json")
    if not isinstance(old, dict) or old.get("format") not in (
        ZERITH_VA_EXPORT_FORMAT,
        LEGACY_EXPORT_FORMAT,
    ):
        raise ValueError(
            "Expected export format zerith-va-policy-action20-linear-g4d-teacher; "
            "provide the exported weights and manifest described in the Zerith VA README"
        )
    stats = old.get("normalization_stats")
    validate_h1_command_contract(stats.get("command_contract") if isinstance(stats, dict) else None)
    validate_encoder_contract(old.get("config"))
    from zwam_core.models.zerith_va.config import ZerithVAConfig

    allowed = {f.name for f in fields(ZerithVAConfig)}
    config = {k: v for k, v in old["config"].items() if k in allowed}
    if any("rtc" in k.lower() and bool(v) for k, v in old["config"].items()):
        raise ValueError("RTC-enabled checkpoints are incompatible with this policy loader")
    parsed = ZerithVAConfig.from_dict(config)
    if not parsed.h1_command23:
        raise ValueError("H1 Command23 weights required")
    schema = old["action_schema"]
    if (
        schema.get("width") != 23
        or schema.get("execute_rows") != [0, 1, 2, 3]
        or schema.get("primitive_dt") != 1 / 30
        or schema.get("video_dt") != 4 / 30
    ):
        raise ValueError("Checkpoint time/action schema mismatch")
    profile = old["camera_schema"]["profiles"]["fam_wam_raw_3canonical_view_v1"]
    from .contract import CAMERAS

    if tuple(profile["source_camera_names"]) != CAMERAS or profile["source_to_canonical"] != [
        0,
        1,
        2,
    ]:
        raise ValueError("Checkpoint camera order mismatch")
    model = verified_file(source, "model.safetensors", old["weights"]["sha256"])
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(model, destination / "model.safetensors")
    result = dict(
        format="zwam-h1-inference-v1",
        config=parsed.as_dict(),
        normalization_stats=old["normalization_stats"],
        weights_sha256=sha256(destination / "model.safetensors"),
        vae_sha256=old["vae_encoder"]["sha256"],
    )
    (destination / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Export directory with weights and manifest.json"
    )
    parser.add_argument("--output", required=True, help="New directory for the inference bundle")
    args = parser.parse_args(argv)
    prepare(args.source, args.output)
    print(f"Policy bundle prepared: {args.output}")


if __name__ == "__main__":
    main()
