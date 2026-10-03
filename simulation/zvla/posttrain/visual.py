"""Causal visual encoders shared by benchmark inference."""

from pathlib import Path
import sys
from typing import Sequence
import torch
import numpy as np
from zvla.vision.letterbox import resize_canonical_canvas, resize_with_pad_rgb

ClientInputError = ValueError

class StreamingWanVAEEncoder:
    """Current-observation-only Wan VAE encoder with per-camera causal caches."""

    def __init__(self, vae, camera_profile, device: torch.device) -> None:
        self.vae = vae
        self.model = vae.model
        self.encoder = self.model.encoder
        self.quant_conv = self.model.conv1
        self.scale = vae.scale
        self.camera_profile = camera_profile
        self.device = device
        self.window = 4
        self.conv_count = sum("CausalConv3d" in module.__class__.__name__
                              for module in self.encoder.modules())
        self.reset()

    def reset(self) -> None:
        self.caches = [[None] * self.conv_count
                       for _ in range(self.camera_profile.source_count)]
        self.calls = 0

    @staticmethod
    def _patchify(value: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = value.shape
        value = value.view(batch, channels, frames, height // 2, 2, width // 2,
                           2)
        value = value.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
        return value.view(batch, channels * 4, frames, height // 2, width // 2)

    def _encode_camera(self, frames: torch.Tensor,
                       camera_index: int) -> torch.Tensor:
        value = frames[None].permute(0, 2, 1, 3, 4).contiguous()
        output = self.encoder(
            self._patchify(value),
            feat_cache=self.caches[camera_index],
            feat_idx=[0],
        )
        mean, _ = self.quant_conv(output).chunk(2, dim=1)
        if isinstance(self.scale[0], torch.Tensor):
            mean = (mean - self.scale[0].view(
                1, -1, 1, 1, 1)) * self.scale[1].view(1, -1, 1, 1, 1)
        else:
            mean = (mean - self.scale[0]) * self.scale[1]
        return mean[0, :, -1]

    @torch.no_grad()
    def encode(
            self, images: np.ndarray,
            camera_names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        self.camera_profile.validate_source_camera_names(
            camera_names, label="inference payload")
        if images.ndim == 4:
            images = images[None]
        if images.ndim != 5 or images.shape[
                1] != self.camera_profile.source_count:
            raise ClientInputError("camera array must have shape (T,C,H,W,3)")
        expected_frames = 1 if self.calls == 0 else None
        if self.calls == 0:
            allowed = (1,)
        else:
            # A deployment may explicitly batch two causal four-frame VAE
            # updates into one RPC.  The model still consumes the final latent
            # and predicts sixteen actions; this only removes one HTTP/visual
            # synchronization point from the outer evaluator.
            allowed = (self.window, self.window * 2)
        if images.shape[0] not in allowed:
            raise ClientInputError(
                "streaming VAE requires exactly one frame on the first "
                f"request and {self.window} or {self.window * 2} newly executed "
                f"frames thereafter; got {images.shape[0]}")
        # Process a possible eight-frame request as two causal four-frame
        # updates.  Returning the final latent is equivalent to two ordinary
        # requests, while keeping the cache state and numerical path intact.
        source = []
        for camera_index in range(images.shape[1]):
            result = None
            for begin in range(0, images.shape[0], self.window):
                frames = np.stack([
                    resize_canonical_canvas(
                        resize_with_pad_rgb(frame)[0], 224)
                    for frame in images[begin:begin + self.window, camera_index]
                ])
                value = torch.from_numpy(frames).permute(0, 3, 1, 2)
                value = value.to(device=self.device, dtype=torch.float32)
                result = self._encode_camera(value.div(127.5).sub(1.0), camera_index)
            source.append(result)
        self.calls += 1
        stacked = torch.cat(source, dim=-2)
        canonical = self.camera_profile.pack_vertical_tensor(
            stacked, height_axis=-2)
        per_camera_height = canonical.shape[
            -2] // self.camera_profile.canonical_count
        spatial = self.camera_profile.spatial_value_mask_numpy(
            self.camera_profile.presence_mask_numpy(),
            frames=1,
            per_camera_h=per_camera_height,
            width=canonical.shape[-1],
            dtype=np.bool_,
        )[0, 0]
        return canonical[None], torch.from_numpy(spatial).to(self.device)[None]

class StreamingVGGTGeometryEncoder:
    """Encode the two current raw views into canonical 7x7 VGGT tokens."""

    def __init__(self, repo: Path, model_path: Path, camera_profile,
                 device: torch.device) -> None:
        sys.path.insert(0, str(repo.resolve()))
        from vggt.models.vggt import VGGT
        model = VGGT.from_pretrained(str(model_path.resolve()))
        self.aggregator = model.aggregator.to(
            device=device, dtype=torch.bfloat16).eval()
        self.camera_profile = camera_profile
        self.device = device

    def reset(self) -> None:
        return None

    @torch.no_grad()
    def encode(
            self, images: np.ndarray,
            camera_names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        self.camera_profile.validate_source_camera_names(
            camera_names, label="VGGT inference payload")
        if images.ndim == 4:
            images = images[None]
        if images.ndim != 5:
            raise ClientInputError("VGGT camera array must be (T,C,H,W,3)")
        current_images = np.stack([
            resize_canonical_canvas(resize_with_pad_rgb(frame)[0], 518)
            for frame in images[-1]
        ])
        current = torch.from_numpy(current_images).permute(0, 3, 1, 2).float()
        current = current.div(255.0)[None].to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            layers, patch_start = self.aggregator(current)
        patches = layers[-1][:, :, patch_start:]
        side = int(patches.shape[2]**0.5)
        if side * side != patches.shape[2] or patches.shape[-1] != 2048:
            raise RuntimeError("unexpected VGGT patch feature shape")
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            patches.permute(0, 1, 3, 2).reshape(-1, patches.shape[-1], side,
                                                side), (7, 7))
        pooled = pooled.flatten(2).transpose(1, 2).reshape(
            1, images.shape[1], 49, 2048).to(torch.float16)
        canonical = pooled.new_zeros(1, self.camera_profile.canonical_count,
                                     49, 2048)
        valid = torch.zeros(
            1,
            self.camera_profile.canonical_count,
            49,
            device=self.device,
            dtype=torch.bool,
        )
        for source, target in enumerate(
                self.camera_profile.source_to_canonical):
            if target >= 0:
                canonical[:, target] = pooled[:, source]
                valid[:, target] = True
        return canonical.flatten(1, 2), valid.flatten(1, 2)
