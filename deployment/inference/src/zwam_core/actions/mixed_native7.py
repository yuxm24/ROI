# SPDX-License-Identifier: Apache-2.0
# Inference-only codec; see NOTICE.
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import torch

from .state_trajectory import (
    ACTION20_COORDINATE_CONVENTION,
    INTERNAL_STATE_ACTION_LAYOUT,
    STATE_TRAJECTORY_LAYOUT,
    CanonicalStateTrajectoryCodec,
    _fp32_robot_boundary,
    _strict_matrix,
)

MINMAX_FORMAT = "zvla-mixed-action20-native7-state20-minmax-v1"
MINMAX_NORMALIZATION_ID = "model_side_robot_minmax_libero_native7_identity_v1"
MINMAX_NORMALIZATION = "per_robot_minmax_minus1_plus1_libero_native7_identity"
MINMAX_STATS_FILE = "mixed_action_minmax.json"
MINMAX_MIN_SPAN = 1e-06
NATIVE7_LAYOUT = ("osc.dx", "osc.dy", "osc.dz", "osc.rx", "osc.ry", "osc.rz", "osc.gripper")
NATIVE7_CONTRACT_ID = "libero-native-osc7-offset0-v1"
NATIVE7_SEMANTICS = "original_libero_osc7_action_j_conditioned_on_obs_j_minus_1"


def native7_rows(valid: torch.Tensor) -> torch.Tensor:
    """Identify rows with seven leading valid action columns; exclude padding."""
    return valid[..., :7].all(-1, keepdim=True) & ~valid[..., 7:].any(-1, keepdim=True)


class MinMaxMixedActionCodec(CanonicalStateTrajectoryCodec):
    def __init__(self, stats: Mapping | None = None, *, embodiments=None):
        selected = tuple(embodiments or (stats or {}).get("embodiments", ("default",)))
        super().__init__(embodiments=selected)
        for modality in ("state", "action"):
            lo = self._buffers.pop(f"{modality}_q01")
            hi = self._buffers.pop(f"{modality}_q99")
            self.register_buffer(f"{modality}_min", lo)
            self.register_buffer(f"{modality}_max", hi)
        native = torch.tensor([name == "libero_franka" for name in selected])
        self.register_buffer("native7_embodiments", native)
        self.contract_version.fill_(4)
        self.action_stat_valid[native, 7:] = False
        if stats is not None:
            expected = {
                "format": MINMAX_FORMAT,
                "normalization": MINMAX_NORMALIZATION,
                "state_layout": list(STATE_TRAJECTORY_LAYOUT),
                "action_layout": list(INTERNAL_STATE_ACTION_LAYOUT),
                "native7_layout": list(NATIVE7_LAYOUT),
                "coordinate_convention": ACTION20_COORDINATE_CONVENTION,
                "embodiments": list(selected),
            }
            for key, value in expected.items():
                if stats.get(key) != value:
                    raise ValueError(f"mixed minmax codec contract mismatch: {key}")
            for modality in ("state", "action"):
                low = _strict_matrix(
                    stats, f"{modality}_min", rows=len(selected), dtype=torch.float64
                )
                high = _strict_matrix(
                    stats, f"{modality}_max", rows=len(selected), dtype=torch.float64
                )
                valid = _strict_matrix(
                    stats, f"{modality}_stat_valid", rows=len(selected), dtype=torch.bool
                )
                constant = _strict_matrix(
                    stats, f"{modality}_degenerate", rows=len(selected), dtype=torch.bool
                )
                if torch.any(valid & ((high < low) | (constant != (high - low < MINMAX_MIN_SPAN)))):
                    raise ValueError("minmax bounds/constant mask are inconsistent")
                if torch.any(valid & ~constant & (high.float() <= low.float())):
                    raise ValueError("nonconstant minmax range collapses at FP32")
                getattr(self, f"{modality}_min").copy_(low)
                getattr(self, f"{modality}_max").copy_(high)
                getattr(self, f"{modality}_stat_valid").copy_(valid)
                getattr(self, f"{modality}_degenerate").copy_(constant)
            if (
                not self.action_stat_valid[native, :7].all()
                or self.action_stat_valid[native, 7:].any()
                or (not (self.action_min[native, :7] == -1).all())
                or (not (self.action_max[native, :7] == 1).all())
            ):
                raise ValueError(
                    "LIBERO native7 requires identity [-1,1] controller bounds and exactly 7 valid columns"
                )
            digest = hashlib.sha256(
                json.dumps(dict(stats), sort_keys=True, separators=(",", ":")).encode()
            ).digest()
            self.stats_sha256.copy_(torch.tensor(list(digest), dtype=torch.uint8))
        semantic = (INTERNAL_STATE_ACTION_LAYOUT, NATIVE7_LAYOUT, MINMAX_NORMALIZATION)
        self.action_layout_sha256.copy_(
            torch.tensor(
                list(hashlib.sha256(json.dumps(semantic, separators=(",", ":")).encode()).digest()),
                dtype=torch.uint8,
            )
        )

    @property
    def state_q01(self):
        return self.state_min

    @property
    def state_q99(self):
        return self.state_max

    @property
    def action_q01(self):
        return self.action_min

    @property
    def action_q99(self):
        return self.action_max

    @staticmethod
    def _normalize(value, valid, q01, q99, stats_valid, degenerate):
        mask = valid & stats_valid
        span = torch.where(degenerate | ~stats_valid, torch.ones_like(q99), q99 - q01)
        result = 2 * (value - q01) / span - 1
        return (torch.where(mask & ~degenerate, result, 0), mask)

    @staticmethod
    def _denormalize(value, q01, q99, degenerate):
        result = q01 + (value + 1) * 0.5 * (q99 - q01)
        return torch.where(degenerate, q01, result)

    def _native(self, embodiment_id):
        return self.native7_embodiments.index_select(0, embodiment_id.long())[:, None, None]

    def _decode_native_actions(self, value, valid, embodiment_id):
        return torch.where(valid, value.float(), 0)

    @_fp32_robot_boundary
    def inference_action_valid(
        self, current_state_abs, current_state_abs_valid, embodiment_id, horizon
    ):
        robot_mask = super().inference_action_valid(
            current_state_abs, current_state_abs_valid, embodiment_id, horizon
        )
        native_mask = self.action_stat_valid[embodiment_id.long(), None].expand_as(robot_mask)
        return torch.where(self._native(embodiment_id), native_mask, robot_mask)

    @_fp32_robot_boundary
    def decode_predictions(
        self,
        normalized_action,
        current_state_abs,
        current_state_abs_valid,
        embodiment_id,
        action_valid=None,
    ):
        if action_valid is None:
            action_valid = self.inference_action_valid(
                current_state_abs,
                current_state_abs_valid,
                embodiment_id,
                normalized_action.shape[1],
            )
        native = self._native(embodiment_id)
        (robot, robot_valid) = super().decode_predictions(
            torch.where(native, 0, normalized_action),
            current_state_abs,
            current_state_abs_valid,
            embodiment_id,
            action_valid & ~native,
        )
        native_valid = action_valid & self.action_stat_valid[embodiment_id.long(), None]
        return (
            torch.where(
                native,
                self._decode_native_actions(normalized_action, native_valid, embodiment_id),
                robot,
            ),
            torch.where(native, native_valid, robot_valid),
        )


def codec_class(config):
    if (
        not config.h1_command23
        or not config.native7_minmax
        or config.action_normalization != "legacy"
    ):
        raise ValueError("This distribution supports the standard H1 Command23 codec only")
    from .h1 import H1Command23Codec

    return H1Command23Codec
