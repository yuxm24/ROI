"""Read-only Linux service/ZCM checks, plus a cooperative controller lock."""

import os
import stat
import subprocess
from pathlib import Path

from ..lock import acquire_lock

__all__ = ["acquire_lock", "check_services"]


def check_services(config):
    status = subprocess.run(
        ["systemctl", "is-active", "robotd.service"], capture_output=True, timeout=5
    )
    if status.returncode:
        raise RuntimeError("robotd is not active; inspect locally, no automatic restart")
    path = config.get("zcm_region", "/dev/shm/zcm/ipcshm/default")
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        region = os.fstat(fd)
        if not stat.S_ISREG(region.st_mode) or region.st_size <= 0 or region.st_nlink != 1:
            raise RuntimeError("Invalid shared memory")
    finally:
        os.close(fd)
    services = {"Motion_Control": [], "SDKService": []}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            parts = (proc / "cmdline").read_bytes().split(b"\0")
            name = Path(parts[0].decode(errors="replace")).name
        except (OSError, IndexError):
            continue
        if name not in services:
            continue
        services[name].append(proc.name)
        # Permission failures must fail closed, not silently skip verification.
        rows = (proc / "maps").read_text().splitlines()
        if any("zcm" in row.lower() and "(deleted)" in row for row in rows):
            raise RuntimeError(f"{name} maps deleted shared memory; repair services locally")
        mappings = [row.split() for row in rows if path in row]
        if not mappings or any(
            int(row[4]) != region.st_ino
            or tuple(int(v, 16) for v in row[3].split(":"))
            != (os.major(region.st_dev), os.minor(region.st_dev))
            for row in mappings
        ):
            raise RuntimeError(f"{name} is not attached to the current shared memory")
    if any(len(pids) != 1 for pids in services.values()):
        raise RuntimeError("Missing or duplicate robot services")
    return services
