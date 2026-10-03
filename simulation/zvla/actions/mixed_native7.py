"""Versioned min/max codec: robot relative Action20 and native LIBERO OSC7.

Twenty columns are a *transport envelope*, not a shared physical layout.
LIBERO uses columns 0:7 and a false mask in 7:20, with a separate Reader I/O
interface. Its controller bounds are already [-1, 1], so normalization is
identity: in particular, an expert zero rotation remains exactly zero.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import torch

from .state_trajectory import (
    ACTION20_COORDINATE_CONVENTION, INTERNAL_STATE_ACTION_LAYOUT,
    STATE_TRAJECTORY_LAYOUT, CanonicalStateTrajectoryCodec,
    _fp32_robot_boundary, _strict_matrix,
)

MINMAX_FORMAT = "zvla-mixed-action20-native7-state20-minmax-v1"
MINMAX_NORMALIZATION_ID = "model_side_robot_minmax_libero_native7_identity_v1"
MINMAX_NORMALIZATION = "per_robot_minmax_minus1_plus1_libero_native7_identity"
MINMAX_STATS_FILE = "mixed_action_minmax.json"
MINMAX_MIN_SPAN = 1e-6  # Preserve the existing near-constant-dimension policy.
NATIVE7_LAYOUT = ("osc.dx", "osc.dy", "osc.dz", "osc.rx", "osc.ry", "osc.rz", "osc.gripper")
NATIVE7_CONTRACT_ID = "libero-native-osc7-offset0-v1"
NATIVE7_SEMANTICS = "original_libero_osc7_action_j_conditioned_on_obs_j_minus_1"


def native7_rows(valid: torch.Tensor) -> torch.Tensor:
    """Closed structural selector, (..., 1); no device-host synchronization.

    The producer/codec validates that only LIBERO has this mask. All-invalid
    exact-once sentinels select the robot branch and contribute zero loss.
    """
    return valid[..., :7].all(-1, keepdim=True) & ~valid[..., 7:].any(-1, keepdim=True)


class MinMaxMixedActionCodec(CanonicalStateTrajectoryCodec):
    def __init__(self, stats: Mapping | None = None, *, embodiments=None):
        selected = tuple(embodiments or (stats or {}).get("embodiments", ("default",)))
        # Keep the established coordinate operations, but use distinct named
        # buffers and a new semantic version. No percentile bounds are loaded.
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
                low = _strict_matrix(stats, f"{modality}_min", rows=len(selected), dtype=torch.float64)
                high = _strict_matrix(stats, f"{modality}_max", rows=len(selected), dtype=torch.float64)
                valid = _strict_matrix(stats, f"{modality}_stat_valid", rows=len(selected), dtype=torch.bool)
                constant = _strict_matrix(stats, f"{modality}_degenerate", rows=len(selected), dtype=torch.bool)
                if torch.any(valid & ((high < low) | (constant != ((high - low) < MINMAX_MIN_SPAN)))):
                    raise ValueError("minmax bounds/constant mask are inconsistent")
                # Checkpoint/model boundary is FP32, statistics are calculated
                # in float64. Refuse a nonconstant range collapsing at FP32.
                if torch.any(valid & ~constant & (high.float() <= low.float())):
                    raise ValueError("nonconstant minmax range collapses at FP32")
                getattr(self, f"{modality}_min").copy_(low)
                getattr(self, f"{modality}_max").copy_(high)
                getattr(self, f"{modality}_stat_valid").copy_(valid)
                getattr(self, f"{modality}_degenerate").copy_(constant)
            if (not self.action_stat_valid[native, :7].all()
                    or self.action_stat_valid[native, 7:].any()
                    or not (self.action_min[native, :7] == -1).all()
                    or not (self.action_max[native, :7] == 1).all()):
                raise ValueError("LIBERO native7 requires identity [-1,1] controller bounds and exactly 7 valid columns")
            digest = hashlib.sha256(json.dumps(dict(stats), sort_keys=True, separators=(",", ":")).encode()).digest()
            self.stats_sha256.copy_(torch.tensor(list(digest), dtype=torch.uint8))
        semantic = (INTERNAL_STATE_ACTION_LAYOUT, NATIVE7_LAYOUT, MINMAX_NORMALIZATION)
        self.action_layout_sha256.copy_(torch.tensor(list(hashlib.sha256(
            json.dumps(semantic, separators=(",", ":")).encode()).digest()), dtype=torch.uint8))

    # Coordinate routines share their existing argument names; persistent
    # state_dict contains only *_min and *_max, never percentile aliases.
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
        return torch.where(mask & ~degenerate, result, 0), mask

    @staticmethod
    def _denormalize(value, q01, q99, degenerate):
        result = q01 + (value + 1) * 0.5 * (q99 - q01)
        return torch.where(degenerate, q01, result)

    def _native(self, embodiment_id):
        return self.native7_embodiments.index_select(0, embodiment_id.long())[:, None, None]

    @_fp32_robot_boundary
    def raw_action_targets(self, current, current_valid, target, target_valid, embodiment_id):
        """The actual pre-normalization target, shared with statistics fitting."""
        native = self._native(embodiment_id)
        raw, valid, _, _ = self.encode_raw_training_targets(
            current, current_valid, torch.where(native, 0, target), target_valid & ~native)
        native_valid = target_valid & self.action_stat_valid[embodiment_id.long(), None]
        return (torch.where(native, torch.where(native_valid, target.double(), 0), raw),
                torch.where(native, native_valid, valid))

    def _encode_native_actions(self, value, valid, embodiment_id):
        return torch.where(valid, value, 0).float()

    def _decode_native_actions(self, value, valid, embodiment_id):
        return torch.where(valid, value.float(), 0)

    @_fp32_robot_boundary
    def encode_training_targets(self, current_state_abs, current_state_abs_valid,
                                target_state_abs, target_state_abs_valid, embodiment_id):
        native = self._native(embodiment_id)
        # Do not interpret controller numbers as position/rotation6d, even
        # transiently. The robot coordinate path only receives robot targets.
        action, valid, state, state_valid = super().encode_training_targets(
            current_state_abs, current_state_abs_valid,
            torch.where(native, 0, target_state_abs),
            target_state_abs_valid & ~native, embodiment_id)
        native_valid = target_state_abs_valid & self.action_stat_valid[embodiment_id.long(), None]
        native_action = self._encode_native_actions(target_state_abs, native_valid, embodiment_id)
        return (torch.where(native, native_action, action),
                torch.where(native, native_valid, valid), state, state_valid)

    @_fp32_robot_boundary
    def inference_action_valid(self, current_state_abs, current_state_abs_valid, embodiment_id, horizon):
        robot_mask = super().inference_action_valid(
            current_state_abs, current_state_abs_valid, embodiment_id, horizon)
        native_mask = self.action_stat_valid[embodiment_id.long(), None].expand_as(robot_mask)
        return torch.where(self._native(embodiment_id), native_mask, robot_mask)

    @_fp32_robot_boundary
    def decode_predictions(self, normalized_action, current_state_abs, current_state_abs_valid,
                           embodiment_id, action_valid=None):
        if action_valid is None:
            action_valid = self.inference_action_valid(current_state_abs, current_state_abs_valid,
                                                      embodiment_id, normalized_action.shape[1])
        native = self._native(embodiment_id)
        robot, robot_valid = super().decode_predictions(
            torch.where(native, 0, normalized_action), current_state_abs, current_state_abs_valid,
            embodiment_id, action_valid & ~native)
        native_valid = action_valid & self.action_stat_valid[embodiment_id.long(), None]
        # Identity, not affine subtraction: preserve expert zeros and signs.
        return (torch.where(native, self._decode_native_actions(normalized_action, native_valid, embodiment_id), robot),
                torch.where(native, native_valid, robot_valid))


def codec_class(config):
    get = config.get if isinstance(config, Mapping) else lambda name, default: getattr(config, name, default)
    if get('metaworld_native4', False):
        if (not get('native7_minmax', False) or get('robotwin_joint14', False)
                or get('action_normalization', 'legacy') != 'legacy'):
            raise ValueError('MetaWorld Native4 cannot use a different action codec')
        from .metaworld_native4 import MetaWorldNative4Codec
        return MetaWorldNative4Codec
    if get('robotwin_joint14',False):
        if not get('native7_minmax',False) or get('action_normalization','legacy')!='legacy':
            raise ValueError('Joint14 uses only its versioned fixed MinMax codec')
        from .robotwin_joint14 import Joint14MinMaxCodec
        return Joint14MinMaxCodec
    mode = get("action_normalization", "legacy")
    if mode not in ("legacy", "meanstd", "meanstd_horizon"):
        raise ValueError(f"unsupported action_normalization: {mode}")
    if mode in ("meanstd", "meanstd_horizon"):
        if not get("native7_minmax", False):
            raise ValueError("meanstd requires the existing mixed Action20/Native7 interface")
        from .meanstd import MeanStdMixedActionCodec, HorizonMeanStdMixedActionCodec
        return HorizonMeanStdMixedActionCodec if mode == "meanstd_horizon" else MeanStdMixedActionCodec
    return MinMaxMixedActionCodec if get("native7_minmax", False) else CanonicalStateTrajectoryCodec
