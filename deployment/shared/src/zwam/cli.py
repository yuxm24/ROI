"""Command-line tools for Zerith VA inference and robot control."""

import argparse
from importlib import import_module

from . import __version__

COMMANDS = {
    "serve": ("zwam.server", "Run the GPU inference server"),
    "robot": ("zwam.h1.runtime", "Check or run the H1 controller; offline by default"),
    "replay": ("zwam.replay", "Run inference on recorded observations"),
    "prepare": ("zwam.artifact", "Prepare a policy bundle from an exported artifact"),
    "doctor": ("zwam.doctor", "Check local configuration and dependency files without hardware"),
    "source-hash": ("zwam.source", "Fingerprint a reviewed external source package"),
}


def main(argv=None):
    """Dispatch to a component without importing its optional dependencies early."""
    parser = argparse.ArgumentParser(
        prog="zwam",
        description=__doc__,
        epilog="Run 'zwam COMMAND --help' for command options.",
    )
    parser.add_argument(
        "--version", action="version", version=f"Zerith VA Deployment {__version__}"
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, help=description, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.command is None:
        if remaining:
            parser.error("unrecognized arguments: " + " ".join(remaining))
        parser.print_help()
        return
    module, _ = COMMANDS[args.command]
    return import_module(module).main(remaining, prog=f"{parser.prog} {args.command}")
