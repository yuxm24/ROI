"""Frozen, per-embodiment action MeanStd; State20 remains the original MinMax.

Fit after anchor-relative conversion on the real masked 16-step targets.
The original mode pools time; the explicit horizon variant retains its axis.
Never fit on an individual chunk or clip in normalized space. Statistics are
part of the checkpoint, not a mutable inference-time preprocessing option.
"""
from __future__ import annotations

import hashlib
import copy
import json
from typing import Mapping

import torch

from .mixed_native7 import MinMaxMixedActionCodec, NATIVE7_LAYOUT
from .state_trajectory import (
    ACTION20_COORDINATE_CONVENTION, INTERNAL_STATE_ACTION_LAYOUT,
    STATE_TRAJECTORY_LAYOUT, _fp32_robot_boundary, _strict_matrix,
)

MEANSTD_FORMAT = "zvla-mixed-action20-native7-action-meanstd-state-minmax-v1"
MEANSTD_NORMALIZATION = "per_robot_action_meanstd_all_valid_dims_state_minmax_v1"
MEANSTD_STATS_FILE = "mixed_action_meanstd.json"
HORIZON_MEANSTD_FORMAT = "zvla-mixed-action20-native7-action-horizon-meanstd-state-minmax-v1"
HORIZON_MEANSTD_NORMALIZATION = "per_robot_per_horizon_action_meanstd_all_valid_dims_state_minmax_v1"
MIN_STD = 1e-6


def statistics_digest(stats):
    return hashlib.sha256(json.dumps(dict(stats), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def inherit_action_statistics(source, target_state_minmax, *, source_path, target_path):
    """Select action rows only; benchmark State20 retains its existing scaling."""
    MeanStdMixedActionCodec(source)
    MinMaxMixedActionCodec(target_state_minmax)
    names = target_state_minmax["embodiments"]
    indices = [source["embodiments"].index(name) for name in names]
    result = copy.deepcopy(source)
    for key in ("action_mean", "action_std", "action_count", "action_stat_valid"):
        result[key] = [result[key][i] for i in indices]
    result.update(embodiments=names, state_minmax=copy.deepcopy(target_state_minmax),
                  state_minmax_sha256=statistics_digest(target_state_minmax),
                  dataset_manifest_fingerprint=target_state_minmax["dataset_manifest_fingerprint"],
                  libero_data_revision=target_state_minmax.get("libero_data_revision"))
    from zvla.vision.wan_assets import sha256_file
    result["provenance"] = dict(inherited_action_stats_sha256=statistics_digest(source),
        metadata_sha256={str(source_path): sha256_file(source_path), str(target_path): sha256_file(target_path)},
        source_embodiments=source["embodiments"], source_indices=indices,
        fitting_provenance=source["provenance"], refitted=False)
    MeanStdMixedActionCodec(result)
    return result


class MeanStdMixedActionCodec(MinMaxMixedActionCodec):
    statistics_format = MEANSTD_FORMAT
    normalization = MEANSTD_NORMALIZATION
    per_horizon = False

    def __init__(self, stats: Mapping | None = None, *, embodiments=None):
        selected = tuple(embodiments or (stats or {}).get("embodiments", ("default",)))
        if stats is not None:
            expected = dict(format=self.statistics_format, normalization=self.normalization,
                            embodiments=list(selected), state_layout=list(STATE_TRAJECTORY_LAYOUT),
                            action_layout=list(INTERNAL_STATE_ACTION_LAYOUT), native7_layout=list(NATIVE7_LAYOUT),
                            coordinate_convention=ACTION20_COORDINATE_CONVENTION,
                            horizon=16, ddof=0, includes_gripper=True)
            if self.per_horizon:
                expected["statistics_axes"] = ["robot_type", "prediction_step", "action_dimension"]
            for key, value in expected.items():
                if stats.get(key) != value:
                    raise ValueError(f"action meanstd contract mismatch: {key}")
            if not isinstance(stats.get("state_minmax"), Mapping):
                raise ValueError("action meanstd requires the frozen original State20 MinMax document")
            if stats.get("state_minmax_sha256") != statistics_digest(stats["state_minmax"]):
                raise ValueError("State20 MinMax fingerprint mismatch")
            if not isinstance(stats.get("provenance"), Mapping) or not stats["provenance"]:
                raise ValueError("action meanstd requires target-window provenance")
        super().__init__(None if stats is None else stats["state_minmax"], embodiments=selected)
        self._buffers.pop("action_min")
        self._buffers.pop("action_max")
        shape = (len(selected), 16, 20) if self.per_horizon else (len(selected), 20)
        self.register_buffer("action_mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("action_std", torch.ones(shape, dtype=torch.float32))
        self.action_degenerate.zero_()
        self.contract_version.fill_(6 if self.per_horizon else 5)
        if stats is not None:
            arrays = []
            for key in ("action_mean", "action_std", "action_count"):
                value = torch.as_tensor(stats[key], dtype=torch.float64)
                if tuple(value.shape) != shape or not torch.isfinite(value).all():
                    raise ValueError(f"invalid action MeanStd shape/values: {key}, expected {shape}")
                arrays.append(value)
            mean, std, count = arrays
            valid = _strict_matrix(stats, "action_stat_valid", rows=len(selected), dtype=torch.bool)
            active = valid[:, None] if self.per_horizon else valid
            bad = active & ((count <= 0) | (count != count.floor()) | (std <= MIN_STD)
                           | ~torch.isfinite(mean.float()) | ~torch.isfinite(std.float())
                           | (std.float() <= MIN_STD))
            if bad.any():
                locations = [(selected[row[0]], *row[1:]) for row in bad.nonzero().tolist()]
                raise ValueError(f"invalid/degenerate action MeanStd dimensions: {locations}")
            if not torch.equal(valid, self.action_stat_valid):
                raise ValueError("MeanStd action validity differs from the source layout")
            if torch.any(~active & ((count != 0) | (mean != 0) | (std != 1))):
                raise ValueError("structurally invalid MeanStd dimensions require count=0, mean=0, std=1")
            self.action_mean.copy_(mean)
            self.action_std.copy_(std)
            self.stats_sha256.copy_(torch.tensor(list(bytes.fromhex(statistics_digest(stats))), dtype=torch.uint8))
        semantic = (INTERNAL_STATE_ACTION_LAYOUT, NATIVE7_LAYOUT, self.normalization)
        self.action_layout_sha256.copy_(torch.tensor(list(hashlib.sha256(
            json.dumps(semantic, separators=(",", ":")).encode()).digest()), dtype=torch.uint8))

    @property
    def action_q01(self):
        return self.action_mean

    @property
    def action_q99(self):
        return self.action_std

    @staticmethod
    def _normalize_action(value, valid, mean, std, stats_valid, degenerate):
        mask = valid & stats_valid
        # No epsilon, clipping, per-chunk fitting, or BF16 statistics.
        return torch.where(mask, (value.float() - mean) / std, 0), mask

    @staticmethod
    def _denormalize_action(value, mean, std, degenerate):
        return value.float() * std + mean

    def _encode_native_actions(self, value, valid, embodiment_id):
        mean, std = self._action_statistics(value, embodiment_id)
        return torch.where(valid, (value.float() - mean) / std, 0)

    def _decode_native_actions(self, value, valid, embodiment_id):
        return self.denormalize_raw_actions(value, valid, embodiment_id)

    @_fp32_robot_boundary
    def denormalize_raw_actions(self, value, valid, embodiment_id):
        """Inverse affine only: OSC7, or relative Action20 before pose decode."""
        mean, std = self._action_statistics(value, embodiment_id)
        mask = valid & self.action_stat_valid[embodiment_id.long(), None]
        return torch.where(mask, value.float() * std + mean, 0)

    def _action_statistics(self, value, embodiment_id):
        mean, std = self.action_mean[embodiment_id.long()], self.action_std[embodiment_id.long()]
        if self.per_horizon:
            if value.ndim != 3 or value.shape[1] != 16:
                raise ValueError("per-horizon MeanStd requires the complete 16-step prediction")
            return mean, std
        return mean[:, None], std[:, None]


class HorizonMeanStdMixedActionCodec(MeanStdMixedActionCodec):
    """Dataset-fitted statistics for each future position, never sample-fitted."""
    statistics_format = HORIZON_MEANSTD_FORMAT
    normalization = HORIZON_MEANSTD_NORMALIZATION
    per_horizon = True

    @_fp32_robot_boundary
    def denormalize_raw_actions(self, value, valid, embodiment_id):
        raw = super().denormalize_raw_actions(value, valid, embodiment_id)
        mean, std = self._action_statistics(value, embodiment_id)
        native = self._native(embodiment_id)
        active = native & valid & self.action_stat_valid[embodiment_id.long(), None]
        # Inverse exact FP32 codewords, not a near-zero deadband. Some fitted
        # (mu, sigma) pairs otherwise map encode(0) to a 1-ULP residual and
        # change OSC's exact-zero branch. Other predictions remain affine.
        raw = torch.where(active & (value.float() == (0. - mean) / std), 0., raw)
        for command in (-1., 1.):
            grip = active[..., 6] & (value[..., 6].float() == (command - mean[..., 6]) / std[..., 6])
            raw[..., 6] = torch.where(grip, command, raw[..., 6])
        return raw

    def _select_statistics(self, embodiment_id, *, batch, q01, q99, valid, degenerate, add_time_axis):
        if q01.ndim != 3:
            return super()._select_statistics(
                embodiment_id, batch=batch, q01=q01, q99=q99, valid=valid,
                degenerate=degenerate, add_time_axis=add_time_axis)
        if not add_time_axis:
            raise ValueError("per-horizon action statistics require a prediction axis")
        mean, std, active, constant = super()._select_statistics(
            embodiment_id, batch=batch, q01=q01, q99=q99, valid=valid,
            degenerate=degenerate, add_time_axis=False)
        return mean, std, active[:, None], constant[:, None]


def meanstd_codec_for_stats(stats):
    if stats.get("format") == HORIZON_MEANSTD_FORMAT:
        return HorizonMeanStdMixedActionCodec
    if stats.get("format") == MEANSTD_FORMAT:
        return MeanStdMixedActionCodec
    raise ValueError("unsupported action MeanStd statistics format")
