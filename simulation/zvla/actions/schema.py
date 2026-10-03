"""Semantic action schemas shared by preprocessing, training, and inference.

The model consumes one fixed-width canonical action vector.  Each dataset
selects an :class:`ActionProfile` that maps its source dimensions into named
canonical slots.  Source-space normalization happens before packing, and
canonical-space decoding happens before de-normalization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch


def expand_action_feature_mask(
    canonical_mask: torch.Tensor,
    *,
    batch_size: int,
    frames: int,
    spatial: Tuple[int, int],
) -> torch.Tensor:
    """Materialize a canonical ``(D,)`` mask as ``(B, D, F, Ha, Wa)``.

    Inference starts from one static semantic feature mask, while the action
    expert requires the same full five-dimensional layout used during
    training.  Materializing the expanded tensor keeps that contract exact
    instead of relying on implicit broadcasting.
    """
    if canonical_mask.ndim != 1:
        raise ValueError(
            "canonical action feature mask must be one-dimensional, got "
            f"shape {tuple(canonical_mask.shape)}"
        )
    if batch_size <= 0 or frames <= 0:
        raise ValueError(
            f"batch_size and frames must be positive, got "
            f"{batch_size} and {frames}"
        )
    if len(spatial) != 2 or any(int(value) <= 0 for value in spatial):
        raise ValueError(
            f"action spatial shape must contain two positive values, got {spatial}"
        )
    height, width = (int(value) for value in spatial)
    return (
        canonical_mask.view(1, canonical_mask.numel(), 1, 1, 1)
        .expand(batch_size, canonical_mask.numel(), frames, height, width)
        .contiguous()
    )


@dataclass(frozen=True)
class ActionProfile:
    """Resolved source-to-canonical semantic mapping for one dataset."""

    name: str
    source_dim: int
    canonical_dim: int
    source_to_canonical: Tuple[int, ...]
    source_semantics: Tuple[str, ...]
    source_units: Tuple[str, ...]
    normalization_id: str
    source_control_hz: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action profile name must be non-empty")
        if self.source_dim <= 0:
            raise ValueError(f"source_dim must be positive, got {self.source_dim}")
        if self.canonical_dim < self.source_dim:
            raise ValueError(
                f"canonical_dim={self.canonical_dim} must be >= "
                f"source_dim={self.source_dim}"
            )
        if len(self.source_to_canonical) != self.source_dim:
            raise ValueError(
                "source_to_canonical must contain exactly source_dim entries: "
                f"{len(self.source_to_canonical)} != {self.source_dim}"
            )
        if len(set(self.source_to_canonical)) != self.source_dim:
            raise ValueError("source_to_canonical indices must be unique")
        invalid = [
            index for index in self.source_to_canonical
            if index < 0 or index >= self.canonical_dim
        ]
        if invalid:
            raise ValueError(
                f"source_to_canonical indices out of range "
                f"[0, {self.canonical_dim}): {invalid}"
            )
        if len(self.source_semantics) != self.source_dim:
            raise ValueError(
                "source_semantics must contain exactly source_dim entries: "
                f"{len(self.source_semantics)} != {self.source_dim}"
            )
        if any(not semantic for semantic in self.source_semantics):
            raise ValueError("source_semantics entries must be non-empty")
        if len(self.source_units) != self.source_dim:
            raise ValueError(
                "source_units must contain exactly source_dim entries: "
                f"{len(self.source_units)} != {self.source_dim}"
            )
        if any(not unit for unit in self.source_units):
            raise ValueError("source_units entries must be non-empty")
        if not self.normalization_id:
            raise ValueError("normalization_id must be non-empty")
        if self.source_control_hz <= 0:
            raise ValueError(
                f"source_control_hz must be positive, got {self.source_control_hz}"
            )

    def _validate_last_dim(self, values: Any, expected: int, label: str) -> None:
        if not hasattr(values, "shape") or len(values.shape) == 0:
            raise ValueError(f"{label} must have at least one dimension")
        if int(values.shape[-1]) != expected:
            raise ValueError(
                f"{label} last dimension must be {expected} for profile "
                f"{self.name!r}, got shape {tuple(values.shape)}"
            )

    def pack_numpy(
        self,
        values: np.ndarray,
        feature_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pack normalized source values and feature validity canonically."""
        values = np.asarray(values)
        self._validate_last_dim(values, self.source_dim, "source actions")
        if feature_mask is None:
            source_mask = np.ones(values.shape, dtype=np.float32)
        else:
            source_mask = np.asarray(feature_mask, dtype=np.float32)
            if source_mask.shape != values.shape:
                raise ValueError(
                    "source feature mask shape must match source actions: "
                    f"{source_mask.shape} != {values.shape}"
                )
            source_mask = (source_mask != 0).astype(np.float32)

        shape = (*values.shape[:-1], self.canonical_dim)
        canonical = np.zeros(shape, dtype=values.dtype)
        canonical_mask = np.zeros(shape, dtype=np.float32)
        indices = np.asarray(self.source_to_canonical, dtype=np.int64)
        canonical[..., indices] = np.where(source_mask.astype(bool), values, 0)
        canonical_mask[..., indices] = source_mask
        return canonical, canonical_mask

    def unpack_numpy(self, values: np.ndarray) -> np.ndarray:
        """Decode canonical model values into source dimension order."""
        values = np.asarray(values)
        self._validate_last_dim(values, self.canonical_dim, "canonical actions")
        return values[..., list(self.source_to_canonical)]

    def unpack_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Torch equivalent of :meth:`unpack_numpy`."""
        self._validate_last_dim(values, self.canonical_dim, "canonical actions")
        indices = torch.tensor(
            self.source_to_canonical,
            dtype=torch.long,
            device=values.device,
        )
        return values.index_select(-1, indices)

    def pack_tensor(
        self,
        values: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack normalized source tensors and their validity canonically."""
        self._validate_last_dim(values, self.source_dim, "source actions")
        if feature_mask is None:
            source_mask = torch.ones_like(values, dtype=torch.bool)
        else:
            if feature_mask.shape != values.shape:
                raise ValueError(
                    "source feature mask shape must match source actions: "
                    f"{tuple(feature_mask.shape)} != {tuple(values.shape)}"
                )
            source_mask = feature_mask

        shape = (*values.shape[:-1], self.canonical_dim)
        canonical = values.new_zeros(shape)
        canonical_mask = source_mask.new_zeros(shape)
        indices = list(self.source_to_canonical)
        canonical[..., indices] = torch.where(
            source_mask.bool(), values, torch.zeros_like(values)
        )
        canonical_mask[..., indices] = source_mask
        return canonical, canonical_mask

    def presence_mask_numpy(self, dtype=np.float32) -> np.ndarray:
        """Return the static canonical feature-presence mask."""
        mask = np.zeros(self.canonical_dim, dtype=dtype)
        mask[list(self.source_to_canonical)] = 1
        return mask

    def presence_mask_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.zeros(self.canonical_dim, device=device, dtype=dtype)
        mask[list(self.source_to_canonical)] = 1
        return mask

    def pack_mask_tensor(self, source_mask: torch.Tensor) -> torch.Tensor:
        """Pack a source-space feature mask into canonical order."""
        self._validate_last_dim(
            source_mask,
            self.source_dim,
            "source action feature mask",
        )
        output = torch.zeros(
            (*source_mask.shape[:-1], self.canonical_dim),
            device=source_mask.device,
            dtype=source_mask.dtype,
        )
        output[..., list(self.source_to_canonical)] = source_mask
        return output

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_dim": self.source_dim,
            "source_to_canonical": list(self.source_to_canonical),
            "source_semantics": list(self.source_semantics),
            "source_units": list(self.source_units),
            "normalization_id": self.normalization_id,
            "source_control_hz": self.source_control_hz,
        }


@dataclass(frozen=True)
class ActionSchema:
    """Versioned model-wide canonical action layout."""

    name: str
    version: int
    canonical_slots: Tuple[str, ...]
    canonical_units: Tuple[str, ...]
    profiles: Mapping[str, ActionProfile]
    default_profile: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action schema name must be non-empty")
        if self.version <= 0:
            raise ValueError(f"action schema version must be positive, got {self.version}")
        if not self.canonical_slots:
            raise ValueError("canonical_slots must be non-empty")
        if any(not slot for slot in self.canonical_slots):
            raise ValueError("canonical slot names must be non-empty")
        if len(set(self.canonical_slots)) != len(self.canonical_slots):
            raise ValueError("canonical slot names must be unique")
        if len(self.canonical_units) != len(self.canonical_slots):
            raise ValueError(
                "canonical_units must contain one entry per canonical slot"
            )
        if any(not unit for unit in self.canonical_units):
            raise ValueError("canonical units must be non-empty")
        if not self.profiles:
            raise ValueError("action schema must define at least one profile")
        canonical_dim = len(self.canonical_slots)
        for profile_name, profile in self.profiles.items():
            if profile_name != profile.name:
                raise ValueError(
                    f"profile key {profile_name!r} does not match "
                    f"profile.name={profile.name!r}"
                )
            if profile.canonical_dim != canonical_dim:
                raise ValueError(
                    f"profile {profile_name!r} canonical_dim="
                    f"{profile.canonical_dim} does not match schema width "
                    f"{canonical_dim}"
                )
            expected_semantics = tuple(
                self.canonical_slots[index]
                for index in profile.source_to_canonical
            )
            if profile.source_semantics != expected_semantics:
                raise ValueError(
                    f"profile {profile_name!r} semantics do not match its "
                    "canonical slots: "
                    f"{profile.source_semantics!r} != {expected_semantics!r}"
                )
        if (
            self.default_profile is not None
            and self.default_profile not in self.profiles
        ):
            raise ValueError(
                f"default_profile={self.default_profile!r} is not present "
                "in action schema profiles"
            )

    @property
    def canonical_dim(self) -> int:
        return len(self.canonical_slots)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ActionSchema":
        required = {
            "name",
            "version",
            "canonical_slots",
            "canonical_units",
            "profiles",
        }
        missing = sorted(required.difference(config.keys()))
        if missing:
            raise ValueError(f"action_schema is missing required keys: {missing}")
        canonical_slots = tuple(str(value) for value in config["canonical_slots"])
        raw_profiles = config["profiles"]
        if not isinstance(raw_profiles, Mapping):
            raise TypeError("action_schema.profiles must be a mapping")
        profiles: dict[str, ActionProfile] = {}
        for name, raw_profile in raw_profiles.items():
            if not isinstance(raw_profile, Mapping):
                raise TypeError(f"action profile {name!r} must be a mapping")
            required_profile = {
                "source_dim",
                "source_to_canonical",
                "source_semantics",
                "source_units",
                "normalization_id",
                "source_control_hz",
            }
            missing_profile = sorted(
                required_profile.difference(raw_profile.keys())
            )
            if missing_profile:
                raise ValueError(
                    f"action profile {name!r} is missing keys: "
                    f"{missing_profile}"
                )
            profiles[str(name)] = ActionProfile(
                name=str(name),
                source_dim=int(raw_profile["source_dim"]),
                canonical_dim=len(canonical_slots),
                source_to_canonical=tuple(
                    int(value)
                    for value in raw_profile["source_to_canonical"]
                ),
                source_semantics=tuple(
                    str(value)
                    for value in raw_profile["source_semantics"]
                ),
                source_units=tuple(
                    str(value)
                    for value in raw_profile["source_units"]
                ),
                normalization_id=str(raw_profile["normalization_id"]),
                source_control_hz=float(raw_profile["source_control_hz"]),
            )
        return cls(
            name=str(config["name"]),
            version=int(config["version"]),
            canonical_slots=canonical_slots,
            canonical_units=tuple(
                str(value) for value in config["canonical_units"]
            ),
            profiles=profiles,
            default_profile=(
                str(config["default_profile"])
                if config.get("default_profile") is not None else None
            ),
        )

    def resolve_profile(self, profile: Optional[str] = None) -> ActionProfile:
        selected = profile or self.default_profile
        if selected is None:
            if len(self.profiles) == 1:
                selected = next(iter(self.profiles))
            else:
                raise ValueError(
                    f"action schema {self.name!r} has multiple profiles; "
                    "select action_profile explicitly"
                )
        try:
            return self.profiles[selected]
        except KeyError as exc:
            raise ValueError(
                f"unknown action profile {selected!r}; available profiles: "
                f"{sorted(self.profiles)}"
            ) from exc

    def as_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "version": self.version,
            "canonical_slots": list(self.canonical_slots),
            "canonical_units": list(self.canonical_units),
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


def build_action_schema(
    config: Optional[Mapping[str, Any] | ActionSchema],
) -> Optional[ActionSchema]:
    if config is None:
        return None
    if isinstance(config, ActionSchema):
        return config
    if not isinstance(config, Mapping):
        raise TypeError(
            "action_schema must be a mapping or None, got "
            f"{type(config).__name__}"
        )
    return ActionSchema.from_config(config)


def resolve_action_profile(
    action_schema: Mapping[str, Any] | ActionSchema,
    profile: Optional[str] = None,
) -> ActionProfile:
    schema = build_action_schema(action_schema)
    if schema is None:
        raise ValueError("an action_schema is required to resolve an action profile")
    return schema.resolve_profile(profile)


def validate_action_stats(
    stats: Mapping[str, Any],
    profile: ActionProfile,
) -> None:
    """Validate the normalization vectors selected by an action profile."""
    datasets = stats.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("normalization stats must contain non-empty 'datasets'")
    if profile.normalization_id not in datasets:
        raise ValueError(
            f"normalization_id={profile.normalization_id!r} for profile "
            f"{profile.name!r} is not present in stats datasets "
            f"{sorted(datasets)}"
        )
    action_entry = datasets[profile.normalization_id].get("actions")
    if not isinstance(action_entry, Mapping):
        raise ValueError(
            f"normalization stats for {profile.normalization_id!r} have no "
            "actions entry"
        )
    for key in ("q01", "q99", "valid_mask"):
        if key not in action_entry:
            raise ValueError(
                f"normalization stats for {profile.normalization_id!r} "
                f"actions have no {key!r}"
            )
        actual_dim = len(action_entry[key])
        if actual_dim != profile.source_dim:
            raise ValueError(
                f"normalization stats for {profile.normalization_id!r} "
                f"actions.{key} have dim {actual_dim}, expected "
                f"source_dim={profile.source_dim}"
            )
