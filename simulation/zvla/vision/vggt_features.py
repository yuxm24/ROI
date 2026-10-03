"""Pinned FP16 VGGT patch feature extraction for canonical camera views."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .wan_assets import sha256_file, validate_relocated_asset_bundle

VGGT_GIT_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_MODEL_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
VGGT_TOKENS_PER_CAMERA = 49
VGGT_FEATURE_DIM = 2048


def validate_vggt_assets(
    receipt_path: str | Path,
    source_repo: str | Path,
    model_root: str | Path,
) -> str:
    import json

    receipt_path = Path(receipt_path).resolve()
    source_repo = Path(source_repo).resolve()
    model_root = Path(model_root).resolve()
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if value.get("format") != "fam-wam-v5-cfs-asset-receipt-v1":
        raise ValueError(f"unsupported asset receipt: {receipt_path}")
    source = value.get("source", {}).get("vggt", {})
    relocated = (Path(str(source.get("path", ""))).resolve() != source_repo
                 or Path(str(value.get("vggt", {}).get("root", ""))).resolve()
                 != model_root)
    if relocated:
        validate_relocated_asset_bundle(
            receipt_path, source_repo, model_root, kind="vggt")
    if source.get("revision") != VGGT_GIT_REVISION:
        raise ValueError("VGGT source revision differs from pinned revision")
    head = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True).stdout.strip()
    if head != VGGT_GIT_REVISION:
        raise ValueError(f"VGGT checkout HEAD differs: {head}")
    model = value.get("vggt", {})
    if model.get("revision") != VGGT_MODEL_REVISION:
        raise ValueError("VGGT model revision differs from pinned revision")
    expected = model.get("files", {}).get("model.safetensors")
    if not isinstance(expected, Mapping):
        raise ValueError("asset receipt misses VGGT model.safetensors")
    checkpoint = model_root / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.stat().st_size != int(expected.get("bytes", -1)):
        raise ValueError("VGGT checkpoint byte size differs")
    if (os.environ.get("ZVLA_FAST_ASSET_VALIDATION") != "1"
            and sha256_file(checkpoint) != expected.get("sha256")):
        raise ValueError("VGGT checkpoint hash differs")
    return sha256_file(receipt_path)


def load_vggt_aggregator(
    source_repo: str | Path,
    model_root: str | Path,
    device: torch.device,
):
    source_repo = Path(source_repo).resolve()
    if str(source_repo) not in sys.path:
        sys.path.insert(0, str(source_repo))
    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained(str(Path(model_root).resolve()))
    return model.aggregator.to(device=device, dtype=torch.bfloat16).eval()


@torch.inference_mode()
def encode_vggt_views(
    aggregator,
    images: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Encode ``(B,V,518,518,3)`` RGB into FP16 ``(B,V,49,2048)``."""
    value = np.asarray(images, dtype=np.uint8)
    if value.ndim != 5 or value.shape[2:] != (518, 518, 3):
        raise ValueError(f"unexpected VGGT RGB shape {value.shape}")
    tensor = torch.from_numpy(value).permute(0, 1, 4, 2, 3).to(
        device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        layers, patch_start = aggregator(tensor)
    patches = layers[-1][:, :, patch_start:]
    side = int(patches.shape[2]**0.5)
    if side * side != patches.shape[2] or patches.shape[-1] != 2048:
        raise RuntimeError(f"unexpected VGGT patches {tuple(patches.shape)}")
    pooled = F.adaptive_avg_pool2d(
        patches.permute(0, 1, 3, 2).reshape(value.shape[0] * value.shape[1],
                                            2048, side, side),
        (7, 7),
    )
    stored = pooled.flatten(2).transpose(1, 2).reshape(
        value.shape[0],
        value.shape[1],
        49,
        2048,
    ).to(dtype=torch.float16)
    if not bool(torch.isfinite(stored).all().item()):
        raise FloatingPointError("VGGT aggregator produced NaN/Inf")
    return stored.cpu().numpy()


__all__ = [
    "VGGT_FEATURE_DIM",
    "VGGT_GIT_REVISION",
    "VGGT_MODEL_REVISION",
    "VGGT_TOKENS_PER_CAMERA",
    "encode_vggt_views",
    "load_vggt_aggregator",
    "validate_vggt_assets",
]
