"""Linux foreground start/identity-checked stop. No broad pkill or service restart."""

import argparse
import json
import os
import signal
from pathlib import Path

from .lock import acquire_lock


def identity(pid):
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("operation", choices=("start", "stop"))
    p.add_argument("component", choices=("server", "h1"))
    args, rest = p.parse_known_args(argv)
    if args.operation == "stop" and rest:
        p.error("stop takes no additional options")
    root = Path(os.environ.get("ZWAM_RUN_DIR", ".run")).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_path = root / (args.component + ".json")
    lock_path = root / (args.component + ".lock")
    if args.operation == "stop":
        state = json.loads(state_path.read_text())
        if state["component"] != args.component:
            raise RuntimeError("Wrong process record")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("Safe pidfd stop unavailable; use Ctrl+C in the owning terminal")
        fd = os.pidfd_open(state["pid"])
        try:
            if identity(state["pid"]) != state["start"]:
                raise RuntimeError("Stale process record; refusing to signal")
            signal.pidfd_send_signal(fd, signal.SIGINT)
        finally:
            os.close(fd)
        print(
            "Normal exit requested. This is NOT an emergency stop; confirm completion in the terminal."
        )
        return
    with acquire_lock(lock_path):
        if state_path.is_symlink():
            raise RuntimeError("Refusing symlink process record")
        state_path.write_text(
            json.dumps(dict(pid=os.getpid(), start=identity(os.getpid()), component=args.component))
        )
        try:
            if args.component == "server":
                from .server import main as run
            else:
                from .h1.runtime import main as run
            run(rest)
        finally:
            state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
