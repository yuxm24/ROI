"""Audit the publication tree without echoing matched private values."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
RESTRICTED_DIRS = {
    "assets",
    "external",
    "weights",
    "data",
    "checkpoints",
    "outputs",
    "logs",
    "wandb",
    ".run",
    "runs",
}
RESTRICTED_SUFFIXES = {
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".npy",
    ".npz",
    ".so",
    ".dylib",
    ".dll",
    ".h5",
    ".hdf5",
    ".mp4",
    ".mov",
    ".pem",
    ".key",
    ".pkl",
    ".pickle",
    ".parquet",
    ".arrow",
    ".db",
}
APPROVED_MEDIA = {
    "media/zerith-logo.png": ("174aa4e86630e042deb8d145681648246691e60d7bd70469ab2b13712916beaf"),
}
SENSITIVE = {
    "private key": re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    "private network address": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b"
    ),
    "private home path": re.compile(
        r"/(?:home|Users)/(?:robot|user|[A-Za-z0-9._-]+)/(?:[^\s'\"]+)"
    ),
    "access token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|(?:AKIA|ASIA)[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,})\b"
    ),
    "credential URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def files_to_publish(root: Path) -> list[Path]:
    if (root / ".git").is_dir():
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            check=True,
        )
        return sorted(root / Path(name.decode()) for name in result.stdout.split(b"\0") if name)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
        if not any(
            part in SKIP or part.endswith(".egg-info") for part in path.relative_to(root).parts
        )
    )


def check(root: Path = ROOT) -> tuple[int, list[str]]:
    root = Path(root).resolve()
    errors: list[str] = []
    files = files_to_publish(root)
    for path in files:
        relative = path.relative_to(root)
        parts = relative.parts
        if path.is_symlink() or any(
            (root / Path(*parts[:i])).is_symlink() for i in range(1, len(parts))
        ):
            errors.append(f"{relative}: symlink in publication tree")
            continue
        if (
            not path.is_file()
            or any(part.lower() in RESTRICTED_DIRS for part in parts)
            or relative.suffix.lower() in RESTRICTED_SUFFIXES
            or relative.name.endswith(".local.json")
            or relative.name.startswith(".env")
            or path.stat().st_size > 5 * 1024 * 1024
        ):
            errors.append(f"{relative}: restricted local asset or oversized file")
            continue
        expected_media_hash = APPROVED_MEDIA.get(relative.as_posix())
        if expected_media_hash is not None:
            data = path.read_bytes()
            if (
                not data.startswith(b"\x89PNG\r\n\x1a\n")
                or hashlib.sha256(data).hexdigest() != expected_media_hash
            ):
                errors.append(f"{relative}: approved media hash or format mismatch")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{relative}: binary or non-UTF-8 file")
            continue
        for label, pattern in SENSITIVE.items():
            if pattern.search(content):
                errors.append(f"{relative}: {label} pattern (value redacted)")
        if "vendor" not in parts and EMAIL.search(content):
            errors.append(f"{relative}: personal email pattern (value redacted)")
    expected = [
        "README.md",
        "LICENSE",
        "LICENSE_POLICY.md",
        "MODEL_LICENSE_POLICY.md",
        "media/zerith-logo.png",
        "NOTICE",
        "simulation/setup.py",
        "deployment/shared/pyproject.toml",
        "deployment/inference/pyproject.toml",
        "deployment/robot/pyproject.toml",
    ]
    for name in expected:
        if not (root / name).is_file():
            errors.append(f"{name}: required release file missing")
    apache_text = root / "simulation/LICENSES/Apache-2.0.txt"
    if (root / "LICENSE").is_file() and apache_text.is_file():
        if (root / "LICENSE").read_bytes() != apache_text.read_bytes():
            errors.append("LICENSE: root Apache-2.0 text differs from bundled copy")
    manifest = root / "deployment/docs/model_source_manifest.json"
    if manifest.is_file():
        for row in json.loads(manifest.read_text())["files"]:
            path = root / "deployment" / row["file"]
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["distributed_sha256"]
            ):
                errors.append(f"deployment/{row['file']}: provenance hash mismatch")
    return len(files), errors


if __name__ == "__main__":
    count, problems = check()
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Release tree audit PASS: {count} files")
