"""One canonical aspect-preserving image transform for VAE and VGGT.

Raw cameras are letterboxed exactly once onto a square RGB canvas.  All model
branches then resize that same canvas without another crop, stretch, or pad.
The current ``fam-wam-raw`` inventory contains both source-resolution videos
and videos already padded to 512 by the sealed converter.  Provenance decides
which operation is legal: source-resolution frames are padded here, while
upstream-padded frames pass through unchanged.  Double padding is forbidden.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import cv2
import numpy as np

CANONICAL_IMAGE_SIZE = 512
CANONICAL_PAD_RGB = (0, 0, 0)
CANONICAL_LETTERBOX_CONTRACT = (
    "aspect_preserving_bicubic_resize_black_pad_512_once_v1")
VAE_IMAGE_CONTRACT = (
    "canonical_letterbox_512_then_bicubic_square_resize_224_v1")
VGGT_IMAGE_CONTRACT = (
    "canonical_letterbox_512_then_bicubic_square_resize_518_v1")
RAW_IMAGE_PROVENANCE_CONTRACT = (
    "fam_wam_raw_exactly_one_black_letterbox_512_before_model_resize_v1")


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
    source_h, source_w = value.shape[:2]
    if source_h <= 0 or source_w <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(size / source_w, size / source_h)
    resized_w = min(size, max(1, int(round(source_w * scale))))
    resized_h = min(size, max(1, int(round(source_h * scale))))
    resized = cv2.resize(
        value, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    pad_left = (size - resized_w) // 2
    pad_right = size - resized_w - pad_left
    pad_top = (size - resized_h) // 2
    pad_bottom = size - resized_h - pad_top
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[...] = np.asarray(pad_rgb, dtype=np.uint8)
    canvas[pad_top:pad_top + resized_h,
           pad_left:pad_left + resized_w] = resized
    content = np.zeros((size, size), dtype=np.bool_)
    content[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = True
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
    return canvas, content, geometry


def resize_canonical_canvas(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an already-letterboxed square; never alter its geometry again."""
    value = np.asarray(image)
    if (value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3
            or value.shape[0] != value.shape[1]):
        raise ValueError("canonical image must be square uint8 HWC RGB")
    if size <= 0:
        raise ValueError("target size must be positive")
    if value.shape[0] == size:
        return np.array(value, copy=True)
    return cv2.resize(value, (size, size), interpolation=cv2.INTER_CUBIC)


def fam_wam_raw_is_preletterboxed(metadata: Mapping[str, Any]) -> bool:
    """Validate image provenance and report whether padding is upstream."""
    if metadata.get("format") != "fam-wam-raw":
        raise ValueError("expected fam-wam-raw metadata")
    if metadata.get("format_version") != "1.0":
        raise ValueError("unsupported fam-wam-raw format_version")
    cameras = metadata.get("camera_specs")
    if not isinstance(cameras, Mapping) or not cameras:
        raise ValueError("fam-wam-raw camera_specs are missing")
    invalid_cameras = {
        str(name): value
        for name, value in cameras.items()
        if not isinstance(value, Mapping) or int(value.get("width", -1)) <= 0
        or int(value.get("height", -1)) <= 0
    }
    if invalid_cameras:
        raise ValueError(
            f"fam-wam-raw camera source dimensions are invalid: {invalid_cameras}"
        )
    provenance = metadata.get("algo_operators_provenance")
    operators = (
        provenance.get("operator_chain", []) if isinstance(
            provenance, Mapping) else [])
    conversions = [
        value for value in operators if isinstance(value, Mapping)
        and value.get("operator") == "mcap_to_fam_wam"
    ]
    if len(conversions) != 1:
        raise ValueError("expected exactly one mcap_to_fam_wam provenance row")
    params = conversions[0].get("params") or {}
    actual = {
        "resize_mode": params.get("resize_mode"),
        "resize_width": params.get("resize_width"),
        "resize_height": params.get("resize_height"),
    }
    upstream_pad = {
        "resize_mode": "pad",
        "resize_width": CANONICAL_IMAGE_SIZE,
        "resize_height": CANONICAL_IMAGE_SIZE,
    }
    no_resize = {
        "resize_mode": None,
        "resize_width": None,
        "resize_height": None,
    }
    if actual == upstream_pad:
        return True
    if actual == no_resize:
        return False
    raise ValueError(
        "fam-wam-raw image conversion must be either no-resize or one "
        f"black-pad 512 transform, got {actual}")


def validate_fam_wam_raw_letterbox(metadata: Mapping[str, Any]) -> None:
    """Compatibility spelling for callers needing provenance validation."""
    fam_wam_raw_is_preletterboxed(metadata)


__all__ = [
    "CANONICAL_IMAGE_SIZE",
    "CANONICAL_LETTERBOX_CONTRACT",
    "CANONICAL_PAD_RGB",
    "LetterboxGeometry",
    "RAW_IMAGE_PROVENANCE_CONTRACT",
    "VAE_IMAGE_CONTRACT",
    "VGGT_IMAGE_CONTRACT",
    "fam_wam_raw_is_preletterboxed",
    "resize_canonical_canvas",
    "resize_with_pad_rgb",
    "validate_fam_wam_raw_letterbox",
]
