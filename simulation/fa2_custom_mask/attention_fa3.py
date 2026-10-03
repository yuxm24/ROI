"""Reproducible FP16/BF16 FlashAttention-3 for Hopper bool masks.

The CUDA kernel is the repository's mask-aware Hopper/CuTe FA3 variant from
commit ``c5ddf2a``.  Its build name and directory include every compiled source
and vendored header, plus the PyTorch/CUDA/Python ABI.  A stale extension from
another checkout or environment therefore cannot be imported accidentally.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import weakref
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import torch

_ROOT = Path(__file__).resolve().parent
_HOPPER_INCLUDE = _ROOT / "vendor" / "flash_attn_hopper"
_CUTLASS_COMMIT = "7127592069c2fe01b041e174ba4345ef9b279671"
_CUTLASS_ROOT = Path(
    os.environ.get(
        "ZVLA_FA3_CUTLASS_ROOT",
        str(_ROOT / "vendor" / "flash_attn_cutlass"),
    )
).expanduser()
_CUTLASS_INCLUDE = _CUTLASS_ROOT / "include"
_COMPILE_SOURCES = (
    _ROOT / "fa3_bindings.cpp",
    _ROOT / "fa3_fwd.cu",
    _ROOT / "fa3_bwd.cu",
    _ROOT / "fa3_mask.cu",
)
_LOCAL_HEADERS = (_ROOT / "fa3_config.h", _ROOT / "fa3_launch.h")
_BASE_SOURCE_COMMIT = "c5ddf2a"
_BASE_SOURCE_TREE = "a0a131ad8e00a2cb8bbcd2550b79b603f4525831"
_VENDOR_TREE = "8008a0b39cb741e0c342ed726ad1a430b334ff74"
_PINNED_SOURCE_FINGERPRINT = (
    "d236711637d0415da1519ed70edd18eb8305e1a6972d31502326ce33ff3b65a2")
_PREBUILT_FORMAT = "zvla-fa3-prebuilt-sm90-v1"
_CUDA_FLAGS = (
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "--ftemplate-backtrace-limit=0",
    "-lineinfo",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
)
_MASK_CACHE_MAX_SIZE = max(
    0,
    int(os.environ.get("ZVLA_FA3_MASK_CACHE_SIZE",
                       os.environ.get("FA2_CUTE_MASK_CACHE_SIZE", "8"))),
)
_MASK_CACHE_MIN_ELEMENTS = max(
    0,
    int(os.environ.get(
        "ZVLA_FA3_MASK_CACHE_MIN_ELEMENTS",
        os.environ.get("FA2_CUTE_MASK_CACHE_MIN_ELEMENTS", "0"),
    )),
)
_MASK_CACHE: OrderedDict[tuple[Any, ...], _MaskCacheEntry] = OrderedDict()


class _MaskCacheEntry:
    __slots__ = ("owner_ref", "version", "prepared", "event", "stream")

    def __init__(self, owner_ref, version, prepared, event, stream):
        self.owner_ref = owner_ref
        self.version = version
        self.prepared = prepared
        self.event = event
        self.stream = stream


def _mask_owner(mask: torch.Tensor) -> torch.Tensor:
    owner = mask
    while isinstance(getattr(owner, "_base", None), torch.Tensor):
        owner = owner._base
    return owner


def _mask_cache_key(mask: torch.Tensor,
                    logical_heads: int) -> tuple[Any, ...]:
    owner = _mask_owner(mask)
    return (
        id(owner),
        logical_heads,
        int(mask.storage_offset()),
        tuple(mask.shape),
        tuple(mask.stride()),
    )


def _mask_cache_eligible(mask: torch.Tensor, logical_heads: int) -> bool:
    if _MASK_CACHE_MAX_SIZE == 0:
        return False
    if _MASK_CACHE_MIN_ELEMENTS == 0:
        return True
    packed_heads = 1 if mask.stride(1) == 0 else logical_heads
    prepared_elements = mask.size(0) * packed_heads * mask.size(2) * mask.size(3)
    return prepared_elements >= _MASK_CACHE_MIN_ELEMENTS


def _lookup_prepared_mask(mask: torch.Tensor, logical_heads: int):
    if not _mask_cache_eligible(mask, logical_heads):
        return None
    owner = _mask_owner(mask)
    key = _mask_cache_key(mask, logical_heads)
    entry = _MASK_CACHE.get(key)
    if (entry is None or entry.owner_ref() is not owner
            or entry.version != owner._version):
        if entry is not None:
            _MASK_CACHE.pop(key, None)
        return None
    _MASK_CACHE.move_to_end(key)
    current_stream_raw = torch._C._cuda_getCurrentRawStream(mask.device.index)
    if current_stream_raw != entry.stream:
        torch.cuda.current_stream(mask.device).wait_event(entry.event)
    return entry.prepared


def _store_prepared_mask(mask: torch.Tensor, logical_heads: int, prepared) -> None:
    if not _mask_cache_eligible(mask, logical_heads):
        return
    owner = _mask_owner(mask)
    key = _mask_cache_key(mask, logical_heads)
    ready = torch.cuda.Event()
    current_stream = torch.cuda.current_stream(mask.device)
    ready.record(current_stream)

    def remove(dead_ref) -> None:
        entry = _MASK_CACHE.get(key)
        if entry is not None and entry.owner_ref is dead_ref:
            _MASK_CACHE.pop(key, None)

    owner_ref = weakref.ref(owner, remove)
    _MASK_CACHE[key] = _MaskCacheEntry(
        owner_ref,
        owner._version,
        prepared,
        ready,
        current_stream.cuda_stream,
    )
    _MASK_CACHE.move_to_end(key)
    while len(_MASK_CACHE) > _MASK_CACHE_MAX_SIZE:
        _MASK_CACHE.popitem(last=False)


def _fingerprinted_files() -> tuple[tuple[str, Path], ...]:
    # Preserve the original virtual paths for a matching external CUTLASS
    # checkout. Attribution documents added to the package are not compiler
    # inputs; the original vendor licenses retain their historical identity.
    header_suffixes = {".h", ".hpp", ".inl", ".cuh"}
    vendor_files = [
        (
            path.relative_to(_ROOT).as_posix(),
            path,
        )
        for path in _HOPPER_INCLUDE.rglob("*")
        if path.is_file() and path.suffix in header_suffixes
    ]
    vendor_files.append((
        "vendor/flash_attn_hopper/LICENSE",
        _HOPPER_INCLUDE / "LICENSE",
    ))
    vendor_files.extend(
        (
            (
                Path("vendor/flash_attn_cutlass/include")
                / path.relative_to(_CUTLASS_INCLUDE)
            ).as_posix(),
            path,
        )
        for path in _CUTLASS_INCLUDE.rglob("*")
        if path.is_file() and path.suffix in header_suffixes
    )
    vendor_files.append((
        "vendor/flash_attn_cutlass/LICENSE.txt",
        _CUTLASS_ROOT / "LICENSE.txt",
    ))
    # ``Path`` ordering compares path components.  Keep that exact ordering
    # because it is part of the historical source fingerprint domain.
    vendor_files.sort(key=lambda item: Path(item[0]))
    files = tuple(
        (path.relative_to(_ROOT).as_posix(), path)
        for path in _COMPILE_SOURCES + _LOCAL_HEADERS
    ) + tuple(vendor_files)
    missing = [str(path) for path in (_HOPPER_INCLUDE, _CUTLASS_INCLUDE)
               if not path.is_dir()]
    missing.extend(str(path) for _, path in files if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "FlashAttention-3 build resources are incomplete. Reinstall the "
            "complete zvla package or restore its fa2_custom_mask resources. "
            "A ZVLA_FA3_CUTLASS_ROOT override must contain the matching "
            "include/ tree and LICENSE.txt. Missing: "
            + ", ".join(missing[:8]))
    return files


@lru_cache(maxsize=1)
def flash_attention3_source_fingerprint() -> str:
    # Source checkouts and installed packages validate the same complete
    # resources before either compiling or accepting a prebuilt receipt.
    digest = hashlib.sha256()
    # Keep the fingerprint domain stable while reporting these as the base
    # snapshot (the local binding changes are hashed below in full).
    digest.update(f"source-commit={_BASE_SOURCE_COMMIT}\n".encode())
    digest.update(f"source-tree={_BASE_SOURCE_TREE}\n".encode())
    digest.update(f"vendor-tree={_VENDOR_TREE}\n".encode())
    for virtual_path, path in _fingerprinted_files():
        digest.update(virtual_path.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    actual = digest.hexdigest()
    if actual != _PINNED_SOURCE_FINGERPRINT:
        raise RuntimeError(
            "FlashAttention-3 source tree differs from the pinned SM90 "
            f"package: {actual} != {_PINNED_SOURCE_FINGERPRINT}")
    return actual


@lru_cache(maxsize=1)
def flash_attention3_build_fingerprint() -> str:
    payload = {
        "source": flash_attention3_source_fingerprint(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python_abi": getattr(sys.implementation, "cache_tag", None),
        "platform": platform.machine(),
        "arch": "9.0a",
        "cflags": ["-O3", "-std=c++17"],
        "cuda_flags": list(_CUDA_FLAGS),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def flash_attention3_build_directory() -> Path:
    default = _ROOT / ".build_fa3"
    root = Path(
        os.environ.get(
            "ZVLA_FA3_BUILD_DIR",
            os.environ.get("FA2_CUTE_BUILD_DIR", str(default)),
        )).expanduser()
    if not root.is_absolute():
        raise ValueError("ZVLA_FA3_BUILD_DIR must be an absolute path")
    return root / flash_attention3_build_fingerprint()[:20]


def flash_attention3_prebuilt_directory(
    root: str | Path | None = None,
) -> Path | None:
    """Return the ABI-fingerprinted prebuilt directory, when configured."""
    if root is None:
        raw = os.environ.get("ZVLA_FA3_PREBUILT_ROOT")
        packaged = _ROOT / "prebuilt"
        if raw is None and not packaged.is_dir():
            return None
        root = raw if raw is not None else packaged
    value = Path(root).expanduser()
    if not value.is_absolute() or value == Path("/"):
        raise ValueError(
            "ZVLA_FA3_PREBUILT_ROOT must be a specific absolute path")
    return value / flash_attention3_build_fingerprint()[:20]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flash_attention3_prebuilt_manifest(
    root: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Validate and return an immutable prebuilt FA3 artifact receipt.

    The shared artifact is accepted only when its compiled source, PyTorch,
    CUDA, Python extension ABI, libstdc++ ABI, platform and SM target match
    this process exactly.  The library byte hash prevents a partial CFS copy
    from being mistaken for a reusable build.
    """
    directory = flash_attention3_prebuilt_directory(root)
    if directory is None:
        raise RuntimeError("ZVLA_FA3_PREBUILT_ROOT is not configured")
    marker = directory / ".complete"
    receipt_path = directory / "zvla-fa3-prebuilt.json"
    if not marker.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(
            f"prebuilt FlashAttention-3 artifact is incomplete: {directory}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "format": _PREBUILT_FORMAT,
        "complete": True,
        "source_fingerprint": flash_attention3_source_fingerprint(),
        "build_fingerprint": flash_attention3_build_fingerprint(),
        "cflags": ["-O3", "-std=c++17"],
        "cuda_flags": list(_CUDA_FLAGS),
        "module_name": (
            f"zvla_fa3_{flash_attention3_build_fingerprint()[:16]}"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python_abi": getattr(sys.implementation, "cache_tag", None),
        "platform": platform.machine(),
        "cxx11_abi": bool(torch.compiled_with_cxx11_abi()),
        "arch": "9.0a",
    }
    if {key: receipt.get(key) for key in expected} != expected:
        raise RuntimeError(
            "prebuilt FlashAttention-3 ABI/source receipt does not match "
            "this runtime")
    build_environment = receipt.get("build_environment")
    expected_environment = {
        "python": platform.python_version(),
        "glibc": list(platform.libc_ver()),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    actual_environment = {
        key: build_environment.get(key)
        for key in expected_environment
    } if isinstance(build_environment, dict) else None
    if actual_environment != expected_environment:
        raise RuntimeError(
            "prebuilt FlashAttention-3 host/GPU receipt does not match this "
            "runtime")
    library = receipt.get("library")
    if not isinstance(library, dict):
        raise TypeError("prebuilt FlashAttention-3 library receipt is invalid")
    name = str(library.get("name", ""))
    if not name or Path(name).name != name or not name.endswith(".so"):
        raise ValueError(f"unsafe prebuilt FlashAttention-3 library: {name!r}")
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(
            f"prebuilt FlashAttention-3 library missing: {path}")
    if path.stat().st_size != int(library.get("bytes", -1)):
        raise RuntimeError(
            f"prebuilt FlashAttention-3 library size differs: {path}")
    if _sha256_file(path) != library.get("sha256"):
        raise RuntimeError(
            f"prebuilt FlashAttention-3 library checksum differs: {path}")
    return path, receipt


def flash_attention3_supported(
    tensor: Optional[torch.Tensor] = None,
) -> bool:
    if not torch.cuda.is_available():
        return False
    device = tensor.device if tensor is not None else torch.cuda.current_device()
    if tensor is not None:
        if (not tensor.is_cuda
                or tensor.dtype not in (torch.float16, torch.bfloat16)
                or tensor.ndim != 4 or tensor.shape[-1] != 128):
            return False
    return torch.cuda.get_device_capability(device) == (9, 0)


def flash_attention3_build_info(*, load_extension: bool = False) -> dict[str, Any]:
    extension = _extension() if load_extension else None
    directory = flash_attention3_build_directory()
    prebuilt = flash_attention3_prebuilt_directory()
    return {
        "backend": "custom-mask-flashattention-3-hopper-cute",
        "base_source_commit": _BASE_SOURCE_COMMIT,
        "base_source_tree": _BASE_SOURCE_TREE,
        "vendor_tree": _VENDOR_TREE,
        "cutlass_commit": _CUTLASS_COMMIT,
        "cutlass_root": str(_CUTLASS_ROOT),
        "local_kernel_changes": [
            "accept Wan dense stride-V without a materializing copy",
            "source-and-ABI-fingerprinted build isolation",
        ],
        "source_fingerprint": flash_attention3_source_fingerprint(),
        "build_fingerprint": flash_attention3_build_fingerprint(),
        "module_name": f"zvla_fa3_{flash_attention3_build_fingerprint()[:16]}",
        "build_directory": str(directory),
        "prebuilt_directory": None if prebuilt is None else str(prebuilt),
        "prebuilt_only": os.environ.get(
            "ZVLA_FA3_PREBUILT_ONLY", "0") == "1",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "python_abi": getattr(sys.implementation, "cache_tag", None),
        "platform": platform.machine(),
        "cxx11_abi": bool(torch.compiled_with_cxx11_abi()),
        "arch": "9.0a",
        "cflags": ["-O3", "-std=c++17"],
        "cuda_flags": list(_CUDA_FLAGS),
        "loaded": extension is not None,
    }


def _write_build_metadata(directory: Path) -> None:
    metadata = flash_attention3_build_info(load_extension=False)
    target = directory / "zvla-fa3-build.json"
    temporary = directory / f".{target.name}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _load_prebuilt_extension(root: str | Path | None = None):
    library, receipt = flash_attention3_prebuilt_manifest(root)
    module_name = str(receipt["module_name"])
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, library)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot create import spec for prebuilt FA3 library {library}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@lru_cache(maxsize=1)
def _extension():
    if not flash_attention3_supported():
        raise RuntimeError(
            "custom FlashAttention-3 requires an NVIDIA SM90 GPU")
    prebuilt_root = os.environ.get("ZVLA_FA3_PREBUILT_ROOT")
    if prebuilt_root is None and (_ROOT / "prebuilt").is_dir():
        prebuilt_root = str(_ROOT / "prebuilt")
    prebuilt_only = os.environ.get("ZVLA_FA3_PREBUILT_ONLY", "0")
    if prebuilt_only not in {"0", "1"}:
        raise ValueError("ZVLA_FA3_PREBUILT_ONLY must be 0 or 1")
    if prebuilt_root is not None:
        return _load_prebuilt_extension(prebuilt_root)
    if prebuilt_only == "1":
        raise RuntimeError(
            "production requires a prebuilt FlashAttention-3 artifact; set "
            "ZVLA_FA3_PREBUILT_ROOT")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    os.environ.setdefault("MAX_JOBS", "2")
    from torch.utils.cpp_extension import load

    build_directory = flash_attention3_build_directory()
    build_directory.mkdir(parents=True, exist_ok=True)
    module_name = f"zvla_fa3_{flash_attention3_build_fingerprint()[:16]}"
    extension = load(
        name=module_name,
        sources=[str(path) for path in _COMPILE_SOURCES],
        extra_include_paths=[
            str(_ROOT),
            str(_HOPPER_INCLUDE),
            str(_CUTLASS_INCLUDE),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=list(_CUDA_FLAGS),
        build_directory=str(build_directory),
        verbose=os.environ.get("ZVLA_FA3_VERBOSE_BUILD",
                               os.environ.get("FA2_CUTE_VERBOSE_BUILD", "0"))
        == "1",
    )
    _write_build_metadata(build_directory)
    return extension


def prebuild_flash_attention3() -> dict[str, Any]:
    """Compile/load the exact FA3 extension and return auditable metadata."""
    return flash_attention3_build_info(load_extension=True)


class _FA3Attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, mask, sm_scale):
        has_mask = mask is not None
        if mask is None:
            mask = torch.empty(0, device=q.device, dtype=torch.bool)
        prepared = _lookup_prepared_mask(mask, q.size(1)) if has_mask else None
        if prepared is None:
            (
                out,
                lse,
                packed_mask,
                bwd_tile_state,
                fwd_tile_state,
                fwd_row_state,
            ) = _extension().forward(q, k, v, mask, float(sm_scale), has_mask)
            if has_mask:
                prepared = (
                    packed_mask,
                    bwd_tile_state,
                    fwd_tile_state,
                    fwd_row_state,
                )
                _store_prepared_mask(mask, q.size(1), prepared)
        else:
            packed_mask, bwd_tile_state, fwd_tile_state, fwd_row_state = prepared
            out, lse = _extension().forward_prepared(
                q,
                k,
                v,
                packed_mask,
                fwd_tile_state,
                fwd_row_state,
                float(sm_scale),
            )
        ctx.save_for_backward(q, k, v, out, packed_mask, bwd_tile_state, lse)
        ctx.sm_scale = float(sm_scale)
        ctx.has_mask = has_mask
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, packed_mask, bwd_tile_state, lse = ctx.saved_tensors
        dq, dk, dv = _extension().backward(
            q,
            k,
            v,
            out,
            dout.contiguous(),
            packed_mask,
            bwd_tile_state,
            lse,
            ctx.sm_scale,
            ctx.has_mask,
        )
        return dq, dk, dv, None, None


def flash_attention_custom_mask_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run mask-aware FA3 with PyTorch SDPA's default scaling semantics."""
    if sm_scale is None:
        sm_scale = q.shape[-1]**-0.5
    return _FA3Attention.apply(q, k, v, mask, float(sm_scale))
