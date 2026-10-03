"""Check the publishable source set; report paths, never matched secret values."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".DS_Store",
}
RESTRICTED = EXCLUDED - {".git"} | {
    ".run",
    "assets",
    "external",
    "weights",
    "data",
    "datasets",
    "checkpoints",
    "outputs",
    "logs",
    "wandb",
    "training",
}
BINARY = {
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".so",
    ".dylib",
    ".dll",
    ".mp4",
    ".mov",
    ".zip",
    ".gz",
    ".tar",
    ".pem",
    ".key",
    ".parquet",
    ".arrow",
    ".csv",
    ".jsonl",
    ".log",
    ".bin",
    ".pkl",
    ".pickle",
    ".db",
    ".sqlite",
}
patterns = [
    re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"/Users/" + r"[^\s/]+/"),
    re.compile(r"/home/" + r"(?:robot|user)/"),
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
]


def source_files(root):
    """Include tracked files even when ignored, plus non-ignored new source."""
    root = Path(root).resolve()
    if (root / ".git").exists():
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
            check=True,
            capture_output=True,
        )
        paths = {Path(p.decode("utf-8")) for p in result.stdout.split(b"\0") if p}
        return sorted(p for p in paths if (root / p).exists() or (root / p).is_symlink())
    return sorted(
        p.relative_to(root)
        for p in root.rglob("*")
        if not any(
            part in EXCLUDED or part.endswith(".egg-info") for part in p.relative_to(root).parts
        )
        and (p.is_file() or p.is_symlink())
    )


def restricted_path(relative, *, distribution=False):
    return (
        any(
            part.lower() in RESTRICTED or (part.endswith(".egg-info") and not distribution)
            for part in relative.parts
        )
        or relative.suffix.lower() in BINARY
        or relative.name.lower().endswith(".local.json")
        or relative.name.lower().startswith(".env")
        or relative.name in {"normalization_stats.json", "text_manifest.json", "tasks.json"}
        or re.match(
            r"(?:train(?:ing)?|finetun(?:e|ing)|distill)(?:[_\-.]|$)", relative.name.lower()
        )
        is not None
    )


def check(root=ROOT):
    root = Path(root).resolve()
    errors = []
    files = source_files(root)
    for relative in files:
        path = root / relative
        if any(
            (root / Path(*relative.parts[:i])).is_symlink()
            for i in range(1, len(relative.parts) + 1)
        ):
            errors.append(f"{relative}: symlink excluded from release")
            continue
        if restricted_path(relative) or not path.is_file() or path.stat().st_size > 1024 * 1024:
            errors.append(f"{relative}: local/generated/binary/large or missing asset")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{relative}: not UTF-8 source")
            continue
        if any(pattern.search(text) for pattern in patterns):
            errors.append(f"{relative}: sensitive pattern (value redacted)")
    return len(files), errors


if __name__ == "__main__":
    count, errors = check()
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Release hygiene PASS: {count} source files")
