"""Check that the three deploy wheels are separate and contain no local assets."""

from __future__ import annotations

import argparse
import stat
import zipfile
from pathlib import Path, PurePosixPath

from check_release import RESTRICTED_DIRS, RESTRICTED_SUFFIXES, SENSITIVE

PACKAGES = {
    "zwam_client": {
        "required": "zwam/client.py",
        "forbidden": ("zwam_core/", "zwam/h1/", "zwam/server.py"),
    },
    "zwam_inference": {"required": "zwam/server.py", "forbidden": ("zwam/h1/", "zwam/client.py")},
    "zwam_h1": {
        "required": "zwam/h1/runtime.py",
        "forbidden": ("zwam_core/", "zwam/server.py", "zwam/model.py"),
    },
}


def check_wheel(path: Path) -> tuple[str, int]:
    name = next((name for name in PACKAGES if path.name.startswith(name + "-")), None)
    if name is None:
        raise ValueError(f"Unrecognized wheel: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = set()
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = PurePosixPath(member.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or member.filename in names
                or stat.S_ISLNK(member.external_attr >> 16)
                or member.file_size > 5 * 1024 * 1024
                or any(part in RESTRICTED_DIRS for part in relative.parts)
                or relative.suffix in RESTRICTED_SUFFIXES
            ):
                raise ValueError(f"Unsafe wheel member: {path.name}:{member.filename}")
            names.add(member.filename)
            content = archive.read(member)
            try:
                decoded = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"Non-text wheel member: {path.name}:{member.filename}") from error
            if any(pattern.search(decoded) for pattern in SENSITIVE.values()):
                raise ValueError(
                    f"Sensitive pattern in {path.name}:{member.filename}; value redacted"
                )
        metadata_files = [member for member in names if member.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError(f"Package metadata missing or duplicated: {path.name}")
        metadata_file = metadata_files[0]
        dist_info = metadata_file.rsplit("/", 1)[0]
        required_license_files = {
            f"{dist_info}/licenses/LICENSE",
            f"{dist_info}/licenses/NOTICE",
        }
        if not required_license_files.issubset(names):
            raise ValueError(f"License files missing: {path.name}")
        if (
            "License-Expression: Apache-2.0"
            not in archive.read(metadata_file).decode("utf-8").splitlines()
        ):
            raise ValueError(f"License metadata mismatch: {path.name}")
    expected = PACKAGES[name]
    if expected["required"] not in names or any(
        path.startswith(prefix) for prefix in expected["forbidden"] for path in names
    ):
        raise ValueError(f"Package boundary violated: {path.name}")
    return name, len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    seen = set()
    for path in sorted(args.directory.glob("*.whl")):
        name, count = check_wheel(path)
        if name in seen:
            raise ValueError(f"Duplicate wheel for {name}")
        seen.add(name)
        print(f"PASS {path.name}: {count} files")
    if seen != set(PACKAGES):
        raise SystemExit(f"Expected three deploy wheels; found {sorted(seen)}")


if __name__ == "__main__":
    main()
