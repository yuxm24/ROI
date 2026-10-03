"""Versioned semantic camera layouts for multi-dataset video training.

Every model-side video tensor uses one fixed vertical stack of canonical
camera slots.  A :class:`CameraProfile` maps a source dataset's camera order
into that stack.  Unmapped canonical slots are exact zeros and carry false
value masks; ignored source cameras never reach the model.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

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
        if any(not name for name in self.source_camera_names):
            raise ValueError("source camera names must be non-empty")
        if len(set(self.source_camera_names)) != len(self.source_camera_names):
            raise ValueError("source camera names must be unique")
        if len(self.source_to_canonical) != len(self.source_camera_names):
            raise ValueError(
                "source_to_canonical must contain one entry per source "
                f"camera: {len(self.source_to_canonical)} != "
                f"{len(self.source_camera_names)}"
            )
        mapped = [index for index in self.source_to_canonical if index >= 0]
        invalid_ignored = [
            index for index in self.source_to_canonical if index < -1
        ]
        if invalid_ignored:
            raise ValueError(
                "ignored source cameras must use source_to_canonical=-1, "
                f"got {invalid_ignored}"
            )
        if not mapped:
            raise ValueError("camera profile must map at least one source camera")
        if len(set(mapped)) != len(mapped):
            raise ValueError(
                "non-negative source_to_canonical indices must be unique"
            )
        invalid = [
            index
            for index in mapped
            if index >= len(self.canonical_slots)
        ]
        if invalid:
            raise ValueError(
                "source_to_canonical indices out of range "
                f"[0, {len(self.canonical_slots)}): {invalid}"
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

    def validate_source_camera_names(
        self,
        names: Optional[Sequence[str]],
        *,
        label: str,
    ) -> None:
        """Require source order to match the profile when metadata exists."""
        if names is None:
            return
        actual = tuple(str(name) for name in names)
        if actual != self.source_camera_names:
            raise ValueError(
                f"{label} camera names/order {list(actual)!r} do not match "
                f"profile {self.name!r}: "
                f"{list(self.source_camera_names)!r}"
            )

    def presence_mask_numpy(self, dtype=np.float32) -> np.ndarray:
        mask = np.zeros(self.canonical_count, dtype=dtype)
        for index in self.source_to_canonical:
            if index >= 0:
                mask[index] = 1
        return mask

    def presence_mask_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.as_tensor(
            self.presence_mask_numpy(),
            device=device,
            dtype=dtype,
        )

    def sample_active_mask_numpy(
        self,
        *,
        dropout_prob: float,
        seed: int,
        dtype=np.float32,
    ) -> np.ndarray:
        """Drop exactly one present camera with the configured probability.

        Dropout is clip-wide.  It is disabled for profiles with fewer than
        two present cameras so a sample can never lose its only observation.
        """
        if not 0.0 <= float(dropout_prob) <= 1.0:
            raise ValueError(
                f"camera dropout probability must be in [0, 1], got "
                f"{dropout_prob}"
            )
        active = self.presence_mask_numpy(dtype=dtype)
        present = np.flatnonzero(active)
        if len(present) < 2 or float(dropout_prob) == 0.0:
            return active
        rng = random.Random(int(seed))
        if rng.random() < float(dropout_prob):
            active[int(rng.choice(present.tolist()))] = 0
        return active

    def _validate_vertical_shape(
        self,
        shape: Sequence[int],
        height_axis: int,
        label: str,
    ) -> tuple[int, int]:
        ndim = len(shape)
        axis = height_axis if height_axis >= 0 else ndim + height_axis
        if axis < 0 or axis >= ndim:
            raise ValueError(
                f"{label} height_axis={height_axis} is invalid for "
                f"rank-{ndim} input"
            )
        height = int(shape[axis])
        if height % self.source_count != 0:
            raise ValueError(
                f"{label} height {height} is not divisible by source camera "
                f"count {self.source_count} for profile {self.name!r}"
            )
        return axis, height // self.source_count

    def pack_vertical_numpy(
        self,
        values: np.ndarray,
        *,
        height_axis: int = -2,
    ) -> np.ndarray:
        """Map a vertically stacked source latent into canonical slot order."""
        values = np.asarray(values)
        axis, per_camera_h = self._validate_vertical_shape(
            values.shape,
            height_axis,
            "source video latent",
        )
        output_shape = list(values.shape)
        output_shape[axis] = per_camera_h * self.canonical_count
        output = np.zeros(output_shape, dtype=values.dtype)
        for source_index, canonical_index in enumerate(
            self.source_to_canonical
        ):
            if canonical_index < 0:
                continue
            source_slice = [slice(None)] * values.ndim
            source_slice[axis] = slice(
                source_index * per_camera_h,
                (source_index + 1) * per_camera_h,
            )
            target_slice = [slice(None)] * values.ndim
            target_slice[axis] = slice(
                canonical_index * per_camera_h,
                (canonical_index + 1) * per_camera_h,
            )
            output[tuple(target_slice)] = values[tuple(source_slice)]
        return output

    def pack_vertical_tensor(
        self,
        values: torch.Tensor,
        *,
        height_axis: int = -2,
    ) -> torch.Tensor:
        """Torch equivalent of :meth:`pack_vertical_numpy`."""
        axis, per_camera_h = self._validate_vertical_shape(
            values.shape,
            height_axis,
            "source video latent",
        )
        output_shape = list(values.shape)
        output_shape[axis] = per_camera_h * self.canonical_count
        output = torch.zeros(
            output_shape,
            device=values.device,
            dtype=values.dtype,
        )
        for source_index, canonical_index in enumerate(
            self.source_to_canonical
        ):
            if canonical_index < 0:
                continue
            source_slice = [slice(None)] * values.ndim
            source_slice[axis] = slice(
                source_index * per_camera_h,
                (source_index + 1) * per_camera_h,
            )
            target_slice = [slice(None)] * values.ndim
            target_slice[axis] = slice(
                canonical_index * per_camera_h,
                (canonical_index + 1) * per_camera_h,
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
                f"camera active mask must have shape "
                f"({self.canonical_count},), got {active.shape}"
            )
        if frames <= 0 or per_camera_h <= 0 or width <= 0:
            raise ValueError(
                "frames, per_camera_h, and width must all be positive"
            )
        rows = np.repeat(active, per_camera_h)
        return np.broadcast_to(
            rows[None, None, :, None],
            (1, frames, per_camera_h * self.canonical_count, width),
        ).copy()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_camera_names": list(self.source_camera_names),
            # -1 is the portable config representation for an ignored input.
            "source_to_canonical": list(self.source_to_canonical),
        }


@dataclass(frozen=True)
class CameraSchema:
    """Versioned model-wide semantic camera layout."""

    name: str
    version: int
    canonical_slots: Tuple[str, ...]
    rope_bases: Tuple[int, ...]
    profiles: Mapping[str, CameraProfile]
    default_profile: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera schema name must be non-empty")
        if self.version <= 0:
            raise ValueError(
                f"camera schema version must be positive, got {self.version}"
            )
        if not self.canonical_slots:
            raise ValueError("canonical camera slots must be non-empty")
        if any(not slot for slot in self.canonical_slots):
            raise ValueError("canonical camera slot names must be non-empty")
        if len(set(self.canonical_slots)) != len(self.canonical_slots):
            raise ValueError("canonical camera slot names must be unique")
        if len(self.rope_bases) != len(self.canonical_slots):
            raise ValueError(
                "rope_bases must contain one entry per canonical camera slot"
            )
        if any(base < 0 for base in self.rope_bases):
            raise ValueError("camera RoPE bases must be non-negative")
        if tuple(sorted(self.rope_bases)) != self.rope_bases:
            raise ValueError("camera RoPE bases must be strictly increasing")
        if len(set(self.rope_bases)) != len(self.rope_bases):
            raise ValueError("camera RoPE bases must be unique")
        if not self.profiles:
            raise ValueError("camera schema must define at least one profile")
        for profile_name, profile in self.profiles.items():
            if profile_name != profile.name:
                raise ValueError(
                    f"profile key {profile_name!r} does not match "
                    f"profile.name={profile.name!r}"
                )
            if profile.canonical_slots != self.canonical_slots:
                raise ValueError(
                    f"camera profile {profile_name!r} canonical slots do "
                    "not match its schema"
                )
        if (
            self.default_profile is not None
            and self.default_profile not in self.profiles
        ):
            raise ValueError(
                f"default_profile={self.default_profile!r} is not present "
                "in camera schema profiles"
            )

    @property
    def canonical_count(self) -> int:
        return len(self.canonical_slots)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CameraSchema":
        required = {
            "name",
            "version",
            "canonical_slots",
            "rope_bases",
            "profiles",
        }
        missing = sorted(required.difference(config.keys()))
        if missing:
            raise ValueError(f"camera_schema is missing required keys: {missing}")
        canonical_slots = tuple(
            str(value) for value in config["canonical_slots"]
        )
        raw_profiles = config["profiles"]
        if not isinstance(raw_profiles, Mapping):
            raise TypeError("camera_schema.profiles must be a mapping")
        profiles: dict[str, CameraProfile] = {}
        for name, raw_profile in raw_profiles.items():
            if not isinstance(raw_profile, Mapping):
                raise TypeError(f"camera profile {name!r} must be a mapping")
            required_profile = {
                "source_camera_names",
                "source_to_canonical",
            }
            missing_profile = sorted(
                required_profile.difference(raw_profile.keys())
            )
            if missing_profile:
                raise ValueError(
                    f"camera profile {name!r} is missing keys: "
                    f"{missing_profile}"
                )
            profiles[str(name)] = CameraProfile(
                name=str(name),
                source_camera_names=tuple(
                    str(value)
                    for value in raw_profile["source_camera_names"]
                ),
                source_to_canonical=tuple(
                    int(value)
                    for value in raw_profile["source_to_canonical"]
                ),
                canonical_slots=canonical_slots,
            )
        return cls(
            name=str(config["name"]),
            version=int(config["version"]),
            canonical_slots=canonical_slots,
            rope_bases=tuple(int(value) for value in config["rope_bases"]),
            profiles=profiles,
            default_profile=(
                str(config["default_profile"])
                if config.get("default_profile") is not None
                else None
            ),
        )

    def resolve_profile(self, profile: Optional[str] = None) -> CameraProfile:
        selected = profile or self.default_profile
        if selected is None:
            if len(self.profiles) == 1:
                selected = next(iter(self.profiles))
            else:
                raise ValueError(
                    f"camera schema {self.name!r} has multiple profiles; "
                    "select camera_profile explicitly"
                )
        try:
            return self.profiles[selected]
        except KeyError as exc:
            raise ValueError(
                f"unknown camera profile {selected!r}; available profiles: "
                f"{sorted(self.profiles)}"
            ) from exc

    def as_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "version": self.version,
            "canonical_slots": list(self.canonical_slots),
            "rope_bases": list(self.rope_bases),
            "profiles": {
                name: profile.as_dict()
                for name, profile in sorted(self.profiles.items())
            },
        }
        if self.default_profile is not None:
            result["default_profile"] = self.default_profile
        return result

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def build_camera_schema(
    config: Optional[Mapping[str, Any] | CameraSchema],
) -> Optional[CameraSchema]:
    if config is None:
        return None
    if isinstance(config, CameraSchema):
        return config
    if not isinstance(config, Mapping):
        raise TypeError(
            "camera_schema must be a mapping or None, got "
            f"{type(config).__name__}"
        )
    return CameraSchema.from_config(config)


def resolve_camera_profile(
    camera_schema: Mapping[str, Any] | CameraSchema,
    profile: Optional[str] = None,
) -> CameraProfile:
    schema = build_camera_schema(camera_schema)
    if schema is None:
        raise ValueError("a camera_schema is required to resolve a camera profile")
    return schema.resolve_profile(profile)
