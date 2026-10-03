"""Audit installed dependencies with expiring, version-scoped exceptions."""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import date
from importlib.metadata import distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def findings(report, exceptions, today=None):
    today = today or date.today()
    errors, accepted = [], []
    for exception in exceptions:
        if date.fromisoformat(exception["expires"]) <= today:
            errors.append(f"Expired dependency exception: {exception['id']}")
    for package in report["dependencies"]:
        if package["name"] in {"zwam-client", "zwam-inference", "zwam-h1"}:
            continue
        if "skip_reason" in package:
            errors.append(f"Unaudited dependency: {package['name']}")
        for vulnerability in package.get("vulns", []):
            ids = {vulnerability["id"], *vulnerability.get("aliases", [])}
            match = next(
                (
                    e
                    for e in exceptions
                    if e["id"] in ids
                    and e["package"] == package["name"]
                    and e["version"] == package["version"].split("+")[0]
                    and date.fromisoformat(e["expires"]) > today
                ),
                None,
            )
            label = f"{package['name']} {package.get('version', '?')}: {vulnerability['id']}"
            (accepted if match else errors).append(label)
    return errors, accepted


def installed_requirements(installed):
    """Audit official Torch CPU/CUDA wheels against their upstream version."""
    versions = {}
    for package in installed:
        name = re.sub(r"[-_.]+", "-", package.metadata["Name"]).lower()
        if name in {"zwam-client", "zwam-inference", "zwam-h1"}:
            continue
        version = package.version
        if name in {"torch", "torchvision"}:
            version = re.sub(r"\+(?:cpu|cu[0-9]+)$", "", version)
        if name in versions and versions[name] != version:
            raise ValueError(f"Conflicting installed versions for {name}")
        versions[name] = version
    return "".join(f"{name}=={version}\n" for name, version in sorted(versions.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Read a pre-generated pip-audit JSON report")
    args = parser.parse_args()
    exceptions = json.loads((ROOT / "docs/maintainers/dependency_exceptions.json").read_text())
    with tempfile.TemporaryDirectory() as directory:
        report_path = args.report or Path(directory) / "audit.json"
        if args.report is None:
            requirements = Path(directory) / "requirements.txt"
            requirements.write_text(installed_requirements(distributions()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "--cache-dir",
                    str(Path(directory) / "cache"),
                    "--disable-pip",
                    "--no-deps",
                    "--requirement",
                    str(requirements),
                    "--format",
                    "json",
                    "--output",
                    str(report_path),
                ],
                check=False,
            )
            if result.returncode not in (0, 1) or not report_path.is_file():
                raise SystemExit("Dependency audit failed to produce a report")
        errors, accepted = findings(json.loads(report_path.read_text()), exceptions)
    for finding in accepted:
        print(f"TEMPORARY EXCEPTION: {finding}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Dependency audit passed with the exceptions listed above")


if __name__ == "__main__":
    main()
