"""Fail-closed collators for Zerith VA batch contracts."""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Mapping, Sequence

import torch

from zvla.core.batch_contracts import (
    FAM_ABSOLUTE_POSE_GEOMETRY_ROBOT_SAMPLE_KEYS,
    FAM_ABSOLUTE_POSE_ROBOT_SAMPLE_KEYS,
    FAM_GEOMETRY_ROBOT_SAMPLE_KEYS,
    FAM_ROBOT_SAMPLE_KEYS,
    FAM_STATE_TRAJECTORY_GEOMETRY_ROBOT_SAMPLE_KEYS,
    FAM_STATE_TRAJECTORY_ROBOT_SAMPLE_KEYS,
    FAM_VIDEO_SAMPLE_KEYS,
    FAM_PAIRED_SAMPLE_KEYS,
)
from zvla.engines import COLLATORS


def _strict_stack(
    batch: Sequence[Mapping[str, torch.Tensor]],
    expected_keys: FrozenSet[str],
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    for index, sample in enumerate(batch):
        actual = frozenset(sample)
        missing = sorted(expected_keys - actual)
        extra = sorted(actual - expected_keys)
        if missing or extra:
            raise KeyError(
                f"sample {index} violates strict Zerith VA contract: "
                f"missing={missing}, extra={extra}"
            )
        invalid = sorted(
            key for key, value in sample.items() if not isinstance(value, torch.Tensor)
        )
        if invalid:
            raise TypeError(f"sample {index} contains non-tensor fields: {invalid}")
    collated = {}
    for key in sorted(expected_keys):
        reference = batch[0][key]
        for index, sample in enumerate(batch[1:], 1):
            value = sample[key]
            if tuple(value.shape) != tuple(reference.shape):
                raise ValueError(
                    f"field {key!r} shape differs in sample {index}: "
                    f"{tuple(value.shape)} != {tuple(reference.shape)}"
                )
            if value.dtype != reference.dtype:
                raise TypeError(
                    f"field {key!r} dtype differs in sample {index}: "
                    f"{value.dtype} != {reference.dtype}"
                )
        collated[key] = torch.stack([sample[key] for sample in batch])
    return collated


@COLLATORS.register_module()
class ZerithVARobotCollator:
    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(batch, FAM_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithVAGeometryRobotCollator:
    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(batch, FAM_GEOMETRY_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithVAAbsolutePoseRobotCollator:
    """Collate only the canonical raw absolute-pose public contract."""

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(batch, FAM_ABSOLUTE_POSE_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithVAAbsolutePoseGeometryRobotCollator:
    """Absolute pose batches with frame-aligned frozen geometry tokens."""

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(
            batch, FAM_ABSOLUTE_POSE_GEOMETRY_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithVAStateTrajectoryRobotCollator:
    """Collate the state-only absolute State34 public contract."""

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(batch, FAM_STATE_TRAJECTORY_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithVAStateTrajectoryGeometryRobotCollator:
    """State34 batches with frame-aligned frozen geometry tokens."""

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(
            batch, FAM_STATE_TRAJECTORY_GEOMETRY_ROBOT_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithVAVideoCollator:
    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return _strict_stack(batch, FAM_VIDEO_SAMPLE_KEYS)


@COLLATORS.register_module()
class ZerithVAPairedCollator:
    """Keep equal robot/video microbatches on every rank and every step."""

    def __call__(self, batch):
        if not batch:
            raise ValueError("cannot collate an empty paired batch")
        for index, sample in enumerate(batch):
            if frozenset(sample) != FAM_PAIRED_SAMPLE_KEYS:
                raise KeyError(f"paired sample {index} must contain robot and video")
        return {
            "robot": _strict_stack(
                [sample["robot"] for sample in batch], FAM_ROBOT_SAMPLE_KEYS
            ),
            "video": _strict_stack(
                [sample["video"] for sample in batch], FAM_VIDEO_SAMPLE_KEYS
            ),
        }
