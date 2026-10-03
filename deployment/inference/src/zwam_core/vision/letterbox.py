# SPDX-License-Identifier: Apache-2.0
# Adapted from the ZWAM/ZVLA model source. Inference-only distribution.
# See NOTICE and the repository LICENSE_POLICY.md.
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

CANONICAL_IMAGE_SIZE = 512
CANONICAL_PAD_RGB = (0, 0, 0)
CANONICAL_LETTERBOX_CONTRACT = "aspect_preserving_bicubic_resize_black_pad_512_once_v1"
VAE_IMAGE_CONTRACT = "canonical_letterbox_512_then_bicubic_square_resize_224_v1"
VGGT_IMAGE_CONTRACT = "canonical_letterbox_512_then_bicubic_square_resize_518_v1"
RAW_IMAGE_PROVENANCE_CONTRACT = "fam_wam_raw_exactly_one_black_letterbox_512_before_model_resize_v1"


@dataclass(frozen=True)
class LetterboxGeometry:
    source_height: int
    source_width: int
    canvas_height: int
    canvas_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    scale: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resize_with_pad_rgb(
    image: np.ndarray,
    *,
    size: int = CANONICAL_IMAGE_SIZE,
    pad_rgb: tuple[int, int, int] = CANONICAL_PAD_RGB,
) -> tuple[np.ndarray, np.ndarray, LetterboxGeometry]:
    """Letterbox one uint8 RGB image and return its real-pixel mask."""
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError("image must be uint8 HWC RGB")
    if size <= 0:
        raise ValueError("letterbox size must be positive")
    (source_h, source_w) = value.shape[:2]
    if source_h <= 0 or source_w <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(size / source_w, size / source_h)
    resized_w = min(size, max(1, int(round(source_w * scale))))
    resized_h = min(size, max(1, int(round(source_h * scale))))
    resized = cv2.resize(value, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    pad_left = (size - resized_w) // 2
    pad_right = size - resized_w - pad_left
    pad_top = (size - resized_h) // 2
    pad_bottom = size - resized_h - pad_top
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[...] = np.asarray(pad_rgb, dtype=np.uint8)
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
    content = np.zeros((size, size), dtype=np.bool_)
    content[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = True
    geometry = LetterboxGeometry(
        source_height=source_h,
        source_width=source_w,
        canvas_height=size,
        canvas_width=size,
        resized_height=resized_h,
        resized_width=resized_w,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        scale=float(scale),
    )
    return (canvas, content, geometry)


def resize_canonical_canvas(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an already-letterboxed square; never alter its geometry again."""
    value = np.asarray(image)
    if (
        value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[2] != 3
        or (value.shape[0] != value.shape[1])
    ):
        raise ValueError("canonical image must be square uint8 HWC RGB")
    if size <= 0:
        raise ValueError("target size must be positive")
    if value.shape[0] == size:
        return np.array(value, copy=True)
    return cv2.resize(value, (size, size), interpolation=cv2.INTER_CUBIC)


__all__ = ["LetterboxGeometry", "resize_canonical_canvas", "resize_with_pad_rgb"]
