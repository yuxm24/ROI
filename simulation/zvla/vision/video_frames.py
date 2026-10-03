"""Bounded sequential MP4 decoding for materialized fam-wam archives."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import av
import numpy as np

from .letterbox import (
    CANONICAL_IMAGE_SIZE,
    resize_canonical_canvas,
    resize_with_pad_rgb,
)


class SequentialVideoFrames:
    """Read nondecreasing global frame indices across ordered MP4 parts.

    Only frames requested by one call are retained.  Callers processing merged
    raw ZIPs must issue bounded, nondecreasing windows; ``read_indices`` does
    not silently split an arbitrarily large request because doing so would
    obscure the causal window contract owned by the downstream model.
    """

    def __init__(
        self,
        parts: Sequence[str | Path],
        *,
        target_size: int,
        source_is_preletterboxed: bool,
        expected_source_size: tuple[int, int] | None = None,
        global_frame_offset: int = 0,
        transform_selected_only: bool = False,
    ) -> None:
        self.parts = tuple(Path(value) for value in parts)
        if not self.parts:
            raise ValueError("video reader needs at least one MP4 part")
        if target_size <= 0:
            raise ValueError("target_size must be positive")
        self.target_size = int(target_size)
        self.source_is_preletterboxed = bool(source_is_preletterboxed)
        self.expected_source_size = (None if expected_source_size is None else
                                     (int(expected_source_size[0]),
                                      int(expected_source_size[1])))
        self.global_frame_offset = int(global_frame_offset)
        # Decode every frame in order (including inter-frame dependencies),
        # but avoid RGB conversion and two resizes for frames not requested.
        # Opt-in: callers outside the VGGT continuation retain their old path.
        self.transform_selected_only = bool(transform_selected_only)
        if self.global_frame_offset < 0:
            raise ValueError("global_frame_offset cannot be negative")
        self._part_index = -1
        self._container = None
        self._iterator = None
        # A packed preprocessing task extracts only the MP4 parts intersecting
        # its episode interval.  Keep requests in the archive-global frame
        # coordinate without decoding every preceding part again.
        self._decoded_index = self.global_frame_offset - 1
        self._cached_index = self.global_frame_offset - 1
        self._cached_frame = None
        self._last_request = self.global_frame_offset - 1

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
        self._container = None
        self._iterator = None

    def __enter__(self) -> "SequentialVideoFrames":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _open_next_part(self) -> bool:
        if self._container is not None:
            self._container.close()
        self._part_index += 1
        if self._part_index >= len(self.parts):
            self._container = None
            self._iterator = None
            return False
        path = self.parts[self._part_index]
        if not path.is_file():
            raise FileNotFoundError(path)
        self._container = av.open(str(path), mode="r")
        self._iterator = self._container.decode(video=0)
        return True

    def _next_frame(self, *, transform: bool = True) -> np.ndarray | None:
        while True:
            if self._iterator is None and not self._open_next_part():
                raise EOFError(
                    f"video ended after global frame {self._decoded_index}")
            try:
                frame = next(self._iterator)
            except StopIteration:
                self._iterator = None
                continue
            self._decoded_index += 1
            source_size = (int(frame.height), int(frame.width))
            if self.source_is_preletterboxed:
                if source_size != (CANONICAL_IMAGE_SIZE, CANONICAL_IMAGE_SIZE):
                    raise ValueError(
                        "provenance declares upstream pad512 but decoded "
                        f"frame has size {source_size}")
            else:
                if (self.expected_source_size is not None
                        and source_size != self.expected_source_size):
                    raise ValueError(
                        "decoded source frame differs from camera metadata: "
                        f"{source_size} != {self.expected_source_size}")
            if not transform:
                return None
            rgb = frame.to_ndarray(format="rgb24")
            if self.source_is_preletterboxed:
                canonical = rgb
            else:
                canonical, _, _ = resize_with_pad_rgb(rgb)
            return resize_canonical_canvas(canonical, self.target_size)

    def read_indices(self, indices: np.ndarray) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1 or not len(requested):
            raise ValueError(
                "requested frame indices must be a non-empty vector")
        if np.any(np.diff(requested) < 0):
            raise ValueError("requested frame indices must be nondecreasing")
        if int(requested[0]) < self._last_request:
            raise ValueError("video reader cannot seek backward")
        output = np.empty(
            (len(requested), self.target_size, self.target_size, 3),
            dtype=np.uint8,
        )
        for output_index, requested_index in enumerate(requested):
            target = int(requested_index)
            if target < 0:
                raise ValueError("video frame index cannot be negative")
            while self._cached_index < target:
                self._cached_frame = self._next_frame(
                    transform=(not self.transform_selected_only
                               or self._decoded_index + 1 == target))
                self._cached_index = self._decoded_index
            if self._cached_index != target or self._cached_frame is None:
                raise RuntimeError(
                    "sequential decoder frame accounting drifted")
            output[output_index] = self._cached_frame
            self._last_request = target
        return output


__all__ = ["SequentialVideoFrames"]
