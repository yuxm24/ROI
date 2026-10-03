"""Cooperative local process lock with symlink and replacement checks."""

import fcntl
import os
import stat
from pathlib import Path


def acquire_lock(path):
    path = Path(path)
    flags = os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    for _ in range(3):
        try:
            fd = os.open(path, os.O_RDONLY | flags)
        except FileNotFoundError:
            try:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | flags, 0o600)
            except FileExistsError:
                continue
        try:
            info, named = os.fstat(fd), path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise RuntimeError("Unsafe/replaced controller lock")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise
    raise RuntimeError("Controller lock changed repeatedly")
