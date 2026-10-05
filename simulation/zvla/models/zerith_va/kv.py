"""Dense Wan-coordinate K/V containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class LayerKV:
    """One layer of K/V in ``(B,time,heads,spatial,head_dim)`` order."""

    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self) -> None:
        if self.key.ndim != 5 or tuple(self.key.shape) != tuple(
                self.value.shape):
            raise ValueError("LayerKV key/value must share shape (B,T,H,S,D)")


@dataclass(frozen=True)
class DenseKV:
    layers: Tuple[LayerKV, ...]
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("DenseKV must contain at least one layer")
        if self.valid.ndim != 3 or self.valid.dtype != torch.bool:
            raise ValueError("DenseKV.valid must be boolean (B,T,S)")
        expected = (
            self.valid.shape[0],
            self.valid.shape[1],
            self.valid.shape[2],
        )
        for layer in self.layers:
            if (layer.key.shape[0], layer.key.shape[1],
                    layer.key.shape[3]) != expected:
                raise ValueError("DenseKV layer axes do not match valid mask")
