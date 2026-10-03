import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest


def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_dependencies.py"
    spec = importlib.util.spec_from_file_location("dependency_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "version,expires,expected",
    [
        ("2.9.1+cpu", "2026-11-01", False),
        ("2.9.0", "2026-11-01", True),
        ("2.9.1", "2026-09-29", True),
    ],
)
def test_exceptions_match_alias_version_and_expiry(version, expires, expected):
    report = {
        "dependencies": [
            {
                "name": "torch",
                "version": version,
                "vulns": [
                    {"id": "CVE-example", "aliases": ["GHSA-example"]},
                ],
            }
        ]
    }
    exceptions = [
        {"id": "GHSA-example", "package": "torch", "version": "2.9.1", "expires": expires}
    ]
    errors, accepted = checker().findings(report, exceptions, date(2026, 9, 29))
    assert bool(errors) is expected
    assert bool(accepted) is not expected


def test_new_vulnerability_or_skipped_dependency_fails():
    report = {
        "dependencies": [
            {"name": "some-library", "version": "1", "vulns": [{"id": "CVE-new"}]},
            {"name": "unknown-library", "skip_reason": "No advisory data"},
            {"name": "zwam-client", "skip_reason": "Local project"},
        ]
    }
    errors, accepted = checker().findings(report, [])
    assert len(errors) == 2 and not accepted


@pytest.mark.parametrize(
    "name, version, expected",
    [
        ("torch", "2.10.0+cpu", "2.10.0"),
        ("torchvision", "0.25.0+cu128", "0.25.0"),
        ("torch", "2.10.0+custom", "2.10.0+custom"),
        ("another-library", "1.0+cpu", "1.0+cpu"),
    ],
)
def test_official_wheel_versions_are_audited(name, version, expected):
    package = SimpleNamespace(metadata={"Name": name}, version=version)
    assert checker().installed_requirements([package]) == f"{name}=={expected}\n"


def test_conflicting_installed_versions_fail_audit():
    packages = [SimpleNamespace(metadata={"Name": "torch"}, version=v) for v in ("2.9.1", "2.10.0")]
    with pytest.raises(ValueError, match="Conflicting"):
        checker().installed_requirements(packages)
