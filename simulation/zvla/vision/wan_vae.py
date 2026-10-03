"""Pinned Wan2.2 causal VAE preprocessing for canonical robot videos.

This module contains only model math.  It has no knowledge of MCAP, legacy
action schemas, source adapters, or the raw archive layout.  Callers provide
one already-letterboxed 224-pixel RGB stream per canonical camera.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .wan_imports import import_pinned_wan_module

WAN_VAE_LATENT_CHANNELS = 48
WAN_VAE_LATENT_SIZE = 14


def load_wan22_vae(
    checkpoint: str | Path,
    source_repo: str | Path,
    device: torch.device,
):
    """Load the pinned Wan2.2 TI2V VAE without a generic fallback."""
    source_repo = Path(source_repo).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    vae_module = import_pinned_wan_module(source_repo, "modules.vae2_2")
    Wan2_2_VAE = vae_module.Wan2_2_VAE

    vae = Wan2_2_VAE(
        z_dim=48,
        c_dim=160,
        vae_pth=str(checkpoint),
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        device=str(device),
    )
    if not hasattr(vae, "model") or not hasattr(vae, "scale"):
        raise TypeError("pinned Wan2.2 VAE object does not expose model/scale")
    # Keep every runtime reference inside the private, receipt-pinned Wan
    # namespace.  Importing ``wan.modules.vae2_2`` here would either execute
    # Wan's dependency-heavy umbrella package or fail in the sealed image,
    # where the source checkout is deliberately absent from PYTHONPATH.
    vae._zvla_patchify = vae_module.patchify
    return vae


def wan_vae_rgb_window_bounds(
    latent_frames: int,
    *,
    window_latents: int | None = None,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield exact causal latent/RGB half-open windows.

    The first Wan latent consumes one RGB frame and every later latent
    consumes four.  Adjacent windows therefore partition the source stream
    without overlap or omission; changing the window size cannot change the
    model invocation order or its causal feature cache.
    """
    total = int(latent_frames)
    width = (
        int(os.environ.get("ZVLA_VAE_STREAM_WINDOW_LATENTS", "256"))
        if window_latents is None else int(window_latents))
    if total <= 0:
        raise ValueError("latent_frames must be positive")
    if width <= 0:
        raise ValueError("ZVLA_VAE_STREAM_WINDOW_LATENTS must be positive")
    for latent_start in range(0, total, width):
        latent_stop = min(total, latent_start + width)
        rgb_start = (0 if latent_start == 0 else 1 + 4 * (latent_start - 1))
        rgb_stop = 1 + 4 * (latent_stop - 1)
        yield latent_start, latent_stop, rgb_start, rgb_stop


def encode_wan22_vae_camera_windows(
    vae,
    image_windows: Iterable[np.ndarray],
    *,
    device: torch.device,
    latent_frames: int,
    window_latents: int | None = None,
) -> Iterator[np.ndarray]:
    """Encode a bounded RGB-window iterator with the exact causal cache.

    Each yielded array has ``(window_T, 48, 14, 14)`` FP16 layout.  The
    caller may write it directly into a memmap, so neither a full episode's
    RGB frames nor a full episode's latent tensor must exist in host memory.
    """
    model = vae.model
    if tuple(model.conv1.kernel_size) != (1, 1, 1):
        raise RuntimeError(
            "streaming Wan VAE requires the pinned temporal-kernel-1 conv1")
    patchify = getattr(vae, "_zvla_patchify", None)
    if not callable(patchify):
        raise TypeError("pinned Wan2.2 VAE does not expose private patchify")
    windows = iter(image_windows)
    bounds = tuple(
        wan_vae_rgb_window_bounds(
            latent_frames, window_latents=window_latents))
    model.clear_cache()
    try:
        autocast = (
            torch.amp.autocast("cuda", dtype=vae.dtype)
            if vae.dtype in (torch.float16, torch.bfloat16) else nullcontext())
        with torch.inference_mode(), autocast:
            for latent_start, latent_stop, rgb_start, rgb_stop in bounds:
                try:
                    value = np.asarray(next(windows), dtype=np.uint8)
                except StopIteration as exc:
                    raise ValueError(
                        "Wan RGB window iterator ended before latent stream") \
                        from exc
                expected_shape = (rgb_stop - rgb_start, 224, 224, 3)
                if tuple(value.shape) != expected_shape:
                    raise ValueError("Wan RGB window shape differs: "
                                     f"{value.shape} != {expected_shape}")
                tensor = torch.from_numpy(np.array(value,
                                                   copy=True), ).permute(
                                                       0, 3, 1, 2)
                tensor = tensor.to(device=device, dtype=torch.float32)
                tensor = tensor.div_(127.5).sub_(1.0).permute(
                    1, 0, 2, 3).contiguous().unsqueeze(0)
                tensor = patchify(tensor, patch_size=2)
                window_chunks: list[torch.Tensor] = []
                local_start = 0
                for latent_index in range(latent_start, latent_stop):
                    local_stop = local_start + (1 if latent_index == 0 else 4)
                    model._enc_conv_idx = [0]
                    encoded = model.encoder(
                        tensor[:, :, local_start:local_stop],
                        feat_cache=model._enc_feat_map,
                        feat_idx=model._enc_conv_idx,
                    )
                    mu, _ = model.conv1(encoded).chunk(2, dim=1)
                    scale_mean, scale_inverse_std = vae.scale
                    mu = (mu - scale_mean.view(1, model.z_dim, 1, 1, 1)
                          ) * scale_inverse_std.view(1, model.z_dim, 1, 1, 1)
                    if tuple(mu.shape) != (1, 48, 1, 14, 14):
                        raise ValueError(
                            f"unexpected Wan VAE output {tuple(mu.shape)}")
                    window_chunks.append(mu.float().to(dtype=torch.float16))
                    del encoded, mu
                    local_start = local_stop
                if local_start != tensor.shape[2]:
                    raise RuntimeError("Wan RGB window cursor drifted")
                stored = torch.cat(window_chunks, dim=2).squeeze(0)
                if not bool(torch.isfinite(stored).all().item()):
                    raise FloatingPointError("Wan VAE produced NaN/Inf")
                output = stored.permute(1, 0, 2, 3).contiguous().cpu().numpy()
                del tensor, stored, window_chunks
                yield output
            try:
                next(windows)
            except StopIteration:
                pass
            else:
                raise ValueError(
                    "Wan RGB window iterator contains trailing windows")
    finally:
        model.clear_cache()


def _encode_camera_streaming(
    vae,
    camera: np.ndarray,
    *,
    device: torch.device,
    expected_latents: int,
) -> torch.Tensor:
    """Compatibility wrapper for an already-materialized RGB stream."""
    value = np.asarray(camera, dtype=np.uint8)
    if value.ndim != 4 or value.shape[1:] != (224, 224, 3):
        raise ValueError(f"unexpected Wan RGB shape {value.shape}")
    expected_rgb = 1 + 4 * (int(expected_latents) - 1)
    if expected_latents <= 0 or len(value) != expected_rgb:
        raise ValueError(
            f"Wan RGB/latent contract differs: {len(value)} != {expected_rgb}")
    bounds = tuple(wan_vae_rgb_window_bounds(expected_latents))
    encoded = list(
        encode_wan22_vae_camera_windows(
            vae,
            (value[rgb_start:rgb_stop]
             for _, _, rgb_start, rgb_stop in bounds),
            device=device,
            latent_frames=expected_latents,
        ))
    joined = np.concatenate(encoded, axis=0)
    if tuple(joined.shape) != (expected_latents, 48, 14, 14):
        raise RuntimeError(f"Wan camera latent shape drifted: {joined.shape}")
    return torch.from_numpy(joined).permute(1, 0, 2, 3).contiguous()


def encode_wan22_vae_cameras(
    vae,
    images: Sequence[np.ndarray | None],
    *,
    latent_frames: int,
    device: torch.device,
) -> np.ndarray:
    """Encode three canonical views into vertical ``(T,48,42,14)`` latents.

    A missing canonical camera is represented by exact zero latent rows and is
    skipped on the GPU.  Its validity is stored separately by the caller.
    """
    if len(images) != 3:
        raise ValueError(
            "Wan preprocessing requires exactly three camera slots")
    if latent_frames <= 0:
        raise ValueError("latent_frames must be positive")
    expected_rgb = 1 + 4 * (latent_frames - 1)
    encoded_views: list[torch.Tensor] = []
    for slot, values in enumerate(images):
        if values is None:
            encoded_views.append(
                torch.zeros(
                    WAN_VAE_LATENT_CHANNELS,
                    latent_frames,
                    WAN_VAE_LATENT_SIZE,
                    WAN_VAE_LATENT_SIZE,
                    dtype=torch.float16,
                ))
            continue
        value = np.asarray(values, dtype=np.uint8)
        if tuple(value.shape) != (expected_rgb, 224, 224, 3):
            raise ValueError(
                f"camera slot {slot} RGB shape differs: {value.shape}")
        encoded_views.append(
            _encode_camera_streaming(
                vae, value, device=device, expected_latents=latent_frames))
    joined = torch.cat(encoded_views, dim=2).permute(1, 0, 2, 3)
    if tuple(joined.shape) != (latent_frames, 48, 42, 14):
        raise RuntimeError(f"joined Wan latent shape drifted: {joined.shape}")
    return joined.numpy()


__all__ = [
    "WAN_VAE_LATENT_CHANNELS",
    "WAN_VAE_LATENT_SIZE",
    "encode_wan22_vae_camera_windows",
    "encode_wan22_vae_cameras",
    "load_wan22_vae",
    "wan_vae_rgb_window_bounds",
]
