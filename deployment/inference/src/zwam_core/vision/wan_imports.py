# SPDX-License-Identifier: Apache-2.0
# Adapted from the ZWAM/ZVLA model source. Inference-only distribution.
# See NOTICE and the repository LICENSE_POLICY.md.
from __future__ import annotations

import importlib
import importlib.machinery
import sys
import threading
import types
from pathlib import Path

_NAMESPACE = "_zvla_pinned_wan"
_IMPORT_LOCK = threading.Lock()


def _install_namespace(name: str, path: Path) -> None:
    existing = sys.modules.get(name)
    if existing is not None:
        existing_paths = tuple(
            (Path(value).resolve() for value in getattr(existing, "__path__", ()))
        )
        if existing_paths != (path,):
            raise RuntimeError(
                f"pinned Wan namespace {name!r} already points to {existing_paths}, expected {(path,)}"
            )
        return
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    package = types.ModuleType(name)
    package.__file__ = str(path)
    package.__package__ = name
    package.__path__ = [str(path)]
    package.__spec__ = spec
    sys.modules[name] = package


def import_pinned_wan_module(source_repo: str | Path, relative: str):
    """Import one module from the sealed Wan checkout under a private name."""
    source_repo = Path(source_repo).resolve()
    wan_root = source_repo / "wan"
    modules_root = wan_root / "modules"
    if not (wan_root / "__init__.py").is_file():
        raise FileNotFoundError(wan_root / "__init__.py")
    if not (modules_root / "__init__.py").is_file():
        raise FileNotFoundError(modules_root / "__init__.py")
    if relative not in {"modules.t5", "modules.vae2_2", "modules.tokenizers"}:
        raise ValueError(f"unapproved pinned Wan submodule: {relative!r}")
    with _IMPORT_LOCK:
        _install_namespace(_NAMESPACE, wan_root)
        _install_namespace(f"{_NAMESPACE}.modules", modules_root)
        return importlib.import_module(f"{_NAMESPACE}.{relative}")


__all__ = ["import_pinned_wan_module"]
