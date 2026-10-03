# SPDX-License-Identifier: Apache-2.0
# Adapted from the ZWAM/ZVLA model source. Inference-only distribution.
# See NOTICE and the repository LICENSE_POLICY.md.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class CameraProfile:
    """Resolved source-camera mapping for one dataset or runtime."""

    name: str
    source_camera_names: Tuple[str, ...]
    source_to_canonical: Tuple[int, ...]
    canonical_slots: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera profile name must be non-empty")
        if not self.source_camera_names:
            raise ValueError("source_camera_names must be non-empty")
        if any((not name for name in self.source_camera_names)):
            raise ValueError("source camera names must be non-empty")
        if len(set(self.source_camera_names)) != len(self.source_camera_names):
            raise ValueError("source camera names must be unique")
        if len(self.source_to_canonical) != len(self.source_camera_names):
            raise ValueError(
                f"source_to_canonical must contain one entry per source camera: {len(self.source_to_canonical)} != {len(self.source_camera_names)}"
            )
        mapped = [index for index in self.source_to_canonical if index >= 0]
        invalid_ignored = [index for index in self.source_to_canonical if index < -1]
        if invalid_ignored:
            raise ValueError(
                f"ignored source cameras must use source_to_canonical=-1, got {invalid_ignored}"
            )
        if not mapped:
            raise ValueError("camera profile must map at least one source camera")
        if len(set(mapped)) != len(mapped):
            raise ValueError("non-negative source_to_canonical indices must be unique")
        invalid = [index for index in mapped if index >= len(self.canonical_slots)]
        if invalid:
            raise ValueError(
                f"source_to_canonical indices out of range [0, {len(self.canonical_slots)}): {invalid}"
            )

    @property
    def source_count(self) -> int:
        return len(self.source_camera_names)

    @property
    def canonical_count(self) -> int:
        return len(self.canonical_slots)

    @property
    def mapped_canonical_count(self) -> int:
        """Maximum canonical-camera count this profile can activate."""
        return len({index for index in self.source_to_canonical if index >= 0})

    def validate_source_camera_names(self, names: Optional[Sequence[str]], *, label: str) -> None:
        """Require source order to match the profile when metadata exists."""
        if names is None:
            return
        actual = tuple((str(name) for name in names))
        if actual != self.source_camera_names:
            raise ValueError(
                f"{label} camera names/order {list(actual)!r} do not match profile {self.name!r}: {list(self.source_camera_names)!r}"
            )

    def presence_mask_numpy(self, dtype=np.float32) -> np.ndarray:
        mask = np.zeros(self.canonical_count, dtype=dtype)
        for index in self.source_to_canonical:
            if index >= 0:
                mask[index] = 1
        return mask

    def presence_mask_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(self.presence_mask_numpy(), device=device, dtype=dtype)

    def _validate_vertical_shape(
        self, shape: Sequence[int], height_axis: int, label: str
    ) -> tuple[int, int]:
        ndim = len(shape)
        axis = height_axis if height_axis >= 0 else ndim + height_axis
        if axis < 0 or axis >= ndim:
            raise ValueError(f"{label} height_axis={height_axis} is invalid for rank-{ndim} input")
        height = int(shape[axis])
        if height % self.source_count != 0:
            raise ValueError(
                f"{label} height {height} is not divisible by source camera count {self.source_count} for profile {self.name!r}"
            )
        return (axis, height // self.source_count)

    def pack_vertical_numpy(self, values: np.ndarray, *, height_axis: int = -2) -> np.ndarray:
        """Map a vertically stacked source latent into canonical slot order."""
        values = np.asarray(values)
        (axis, per_camera_h) = self._validate_vertical_shape(
            values.shape, height_axis, "source video latent"
        )
        output_shape = list(values.shape)
        output_shape[axis] = per_camera_h * self.canonical_count
        output = np.zeros(output_shape, dtype=values.dtype)
        for source_index, canonical_index in enumerate(self.source_to_canonical):
            if canonical_index < 0:
                continue
            source_slice = [slice(None)] * values.ndim
            source_slice[axis] = slice(
                source_index * per_camera_h, (source_index + 1) * per_camera_h
            )
            target_slice = [slice(None)] * values.ndim
            target_slice[axis] = slice(
                canonical_index * per_camera_h, (canonical_index + 1) * per_camera_h
            )
            output[tuple(target_slice)] = values[tuple(source_slice)]
        return output

    def pack_vertical_tensor(self, values: torch.Tensor, *, height_axis: int = -2) -> torch.Tensor:
        """Torch equivalent of :meth:`pack_vertical_numpy`."""
        (axis, per_camera_h) = self._validate_vertical_shape(
            values.shape, height_axis, "source video latent"
        )
        output_shape = list(values.shape)
        output_shape[axis] = per_camera_h * self.canonical_count
        output = torch.zeros(output_shape, device=values.device, dtype=values.dtype)
        for source_index, canonical_index in enumerate(self.source_to_canonical):
            if canonical_index < 0:
                continue
            source_slice = [slice(None)] * values.ndim
            source_slice[axis] = slice(
                source_index * per_camera_h, (source_index + 1) * per_camera_h
            )
            target_slice = [slice(None)] * values.ndim
            target_slice[axis] = slice(
                canonical_index * per_camera_h, (canonical_index + 1) * per_camera_h
            )
            output[tuple(target_slice)] = values[tuple(source_slice)]
        return output

    def spatial_value_mask_numpy(
        self,
        active_mask: np.ndarray,
        *,
        frames: int,
        per_camera_h: int,
        width: int,
        dtype=np.float32,
    ) -> np.ndarray:
        """Expand a canonical slot mask to ``(1, F, H, W)``."""
        active = np.asarray(active_mask, dtype=dtype)
        if active.shape != (self.canonical_count,):
            raise ValueError(
                f"camera active mask must have shape ({self.canonical_count},), got {active.shape}"
            )
        if frames <= 0 or per_camera_h <= 0 or width <= 0:
            raise ValueError("frames, per_camera_h, and width must all be positive")
        rows = np.repeat(active, per_camera_h)
        return np.broadcast_to(
            rows[None, None, :, None], (1, frames, per_camera_h * self.canonical_count, width)
        ).copy()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_camera_names": list(self.source_camera_names),
            "source_to_canonical": list(self.source_to_canonical),
        }
