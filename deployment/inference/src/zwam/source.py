"""Fingerprint an external Python package without importing it."""

import argparse
import hashlib
from pathlib import Path

IGNORED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store"}


def source_sha256(root):
    """Hash sorted relative names and file bytes; reject symlinks and large assets.

    Package roots exclude model weights and installed third-party dependencies.
    Cache files do not participate, so importing a package leaves its hash stable.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Source package must be a directory")
    result = hashlib.sha256(b"zwam-source-tree-v1\0")
    count = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED for part in relative.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError("External source packages must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("External source packages must contain source files only")
        result.update(relative.as_posix().encode("utf-8") + b"\0")
        result.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    if not count:
        raise ValueError("External source package is empty")
    return result.hexdigest()


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("package", help="Reviewed package directory, such as external/Wan2.2/wan")
    args = parser.parse_args(argv)
    print(source_sha256(args.package))


if __name__ == "__main__":
    main()
