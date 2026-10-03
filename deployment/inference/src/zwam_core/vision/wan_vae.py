# SPDX-License-Identifier: Apache-2.0
# Adapted from the ZWAM/ZVLA model source. Inference-only distribution.
# See NOTICE and the repository LICENSE_POLICY.md.
from __future__ import annotations

from pathlib import Path

import torch

from .wan_imports import import_pinned_wan_module

WAN_VAE_LATENT_CHANNELS = 48
WAN_VAE_LATENT_SIZE = 14


def load_wan22_vae(checkpoint: str | Path, source_repo: str | Path, device: torch.device):
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
    vae._zvla_patchify = vae_module.patchify
    return vae


__all__ = ["load_wan22_vae"]
