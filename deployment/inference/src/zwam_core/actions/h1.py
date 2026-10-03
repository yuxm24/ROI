# SPDX-License-Identifier: Apache-2.0
# Inference-only codec; see NOTICE.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from zwam.contract import H1_WAIST_LAYOUT, validate_h1_command_contract

from .mixed_native7 import MINMAX_MIN_SPAN, MinMaxMixedActionCodec
from .state_trajectory import _fp32_robot_boundary

H1_DIM = 23
H1_FORMAT = "zvla-h1-command23-minmax-v1"
WAIST_LAYOUT = list(H1_WAIST_LAYOUT)
# Historical unresolved source declaration, retained for identification only.
# It is rejected for deployment; it does not imply a body_yaw_link transform.
SOURCE_NATIVE_CONTRACT = dict(
    pose_frame="source_native_shared_action_state_frame",
    position_unit="m",
    waist_units=["source_native"] * 3,
    waist_joint_order=WAIST_LAYOUT,
    action_row_offset=0,
    time_alignment_evidence="fam-wam-raw-1.1 fixed-grid paired action/state; user-provided specification",
    frame_evidence="user-approved assumption: use source values and source frame unchanged",
    physical_semantics_verified=False,
    conversion="none",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_command_contract(contract):
    """Require source-backed definitions, not an inferred +1 or guessed units."""
    required = (
        "pose_frame",
        "position_unit",
        "waist_units",
        "waist_joint_order",
        "action_row_offset",
        "time_alignment_evidence",
        "frame_evidence",
    )
    if not isinstance(contract, dict) or any((k not in contract for k in required)):
        raise ValueError("H1 command time/frame/unit contract is incomplete")
    validate_h1_command_contract(contract)
    if type(contract["action_row_offset"]) is not int or contract["action_row_offset"] != 0:
        raise ValueError("H1 fixed-grid command supervision requires action offset 0")
    for key in ("pose_frame", "position_unit", "time_alignment_evidence", "frame_evidence"):
        if not isinstance(contract[key], str) or not contract[key].strip():
            raise ValueError(f"H1 missing {key}")


class H1Command23Codec(nn.Module):
    def __init__(self, stats=None, *, embodiments=None):
        super().__init__()
        selected = tuple(embodiments or (stats or {}).get("embodiments", ("h1",)))
        if selected.count("h1") != 1:
            raise ValueError("H1 codec needs exactly one h1 registry row")
        if stats is not None:
            if (
                stats.get("format") != H1_FORMAT
                or stats.get("embodiments") != list(selected)
                or stats.get("waist_layout") != WAIST_LAYOUT
                or (stats.get("arms_minmax_sha256") != digest(stats["arms_minmax"]))
            ):
                raise ValueError("H1 statistics contract mismatch")
            validate_command_contract(stats.get("command_contract"))
        self.embodiments = selected
        self.arms = MinMaxMixedActionCodec(
            None if stats is None else stats["arms_minmax"], embodiments=selected
        )
        active = torch.zeros(len(selected), 3, dtype=torch.bool)
        active[selected.index("h1")] = True
        self.register_buffer("waist_stat_valid", active)
        for modality in ("state", "action"):
            low = torch.zeros(len(selected), 3)
            high = torch.ones(len(selected), 3)
            degenerate = torch.zeros_like(active)
            if stats is not None:
                low = torch.as_tensor(stats[f"{modality}_waist_min"], dtype=torch.float64)
                high = torch.as_tensor(stats[f"{modality}_waist_max"], dtype=torch.float64)
                if (
                    low.shape != active.shape
                    or high.shape != active.shape
                    or (not torch.isfinite(low).all())
                    or (not torch.isfinite(high).all())
                    or (high < low).any()
                ):
                    raise ValueError("invalid H1 waist bounds")
                if (low[~active] != 0).any() or (high[~active] != 1).any():
                    raise ValueError("old embodiments require neutral invalid waist statistics")
                degenerate = high - low < MINMAX_MIN_SPAN
                if (
                    (~degenerate & active & (high.float() <= low.float())).any()
                    or not torch.isfinite(low.float()).all()
                    or (not torch.isfinite(high.float()).all())
                ):
                    raise ValueError("H1 waist bounds collapse/overflow at FP32")
            self.register_buffer(f"{modality}_waist_min", low.float())
            self.register_buffer(f"{modality}_waist_max", high.float())
            self.register_buffer(f"{modality}_waist_degenerate", degenerate)
        sha = bytes.fromhex(digest(stats)) if stats is not None else bytes(32)
        self.register_buffer("stats_sha256", torch.tensor(list(sha), dtype=torch.uint8))

    @classmethod
    def from_json(
        cls, path, *, expected_embodiments=None, expected_dataset_manifest_fingerprint=None
    ):
        stats = json.loads(Path(path).read_text())
        if (
            expected_dataset_manifest_fingerprint is not None
            and stats.get("provenance", {}).get("dataset_manifest_sha256")
            != expected_dataset_manifest_fingerprint
        ):
            raise ValueError("H1 statistics dataset manifest differs")
        return cls(stats, embodiments=expected_embodiments)

    @property
    def stats_fingerprint(self):
        return bytes(self.stats_sha256.tolist()).hex()

    def _check(self, value, valid, rank):
        if value.ndim != rank or value.shape[-1] != H1_DIM or valid.shape != value.shape:
            raise ValueError("H1 public state/action shape must end in 23")
        if valid.dtype != torch.bool:
            raise ValueError("H1 validity must be boolean")

    def _waist_stats(self, ids, modality, rank):
        low = getattr(self, f"{modality}_waist_min")[ids.long()]
        high = getattr(self, f"{modality}_waist_max")[ids.long()]
        constant = getattr(self, f"{modality}_waist_degenerate")[ids.long()]
        valid = self.waist_stat_valid[ids.long()]
        if rank == 3:
            return (low[:, None], high[:, None], constant[:, None], valid[:, None])
        return (low, high, constant, valid)

    def _encode_waist(self, value, valid, ids, modality):
        (lo, hi, constant, active) = self._waist_stats(ids, modality, value.ndim)
        return self.arms._normalize(value.float(), valid, lo, hi, active, constant)

    @_fp32_robot_boundary
    def encode_inference_state(self, current, valid, ids):
        self._check(current, valid, 2)
        (arm, arm_valid) = self.arms.encode_inference_state(current[..., :20], valid[..., :20], ids)
        (waist, waist_valid) = self._encode_waist(current[..., 20:], valid[..., 20:], ids, "state")
        return (torch.cat((arm, waist), -1), torch.cat((arm_valid, waist_valid), -1))

    @_fp32_robot_boundary
    def inference_action_valid(self, current, valid, ids, horizon):
        self._check(current, valid, 2)
        arm = self.arms.inference_action_valid(current[..., :20], valid[..., :20], ids, horizon)
        waist = (valid[..., 20:] & self.waist_stat_valid[ids.long()])[:, None].expand(
            -1, horizon, -1
        )
        return torch.cat((arm, waist), -1)

    @_fp32_robot_boundary
    def decode_predictions(self, action, current, current_valid, ids, action_valid=None):
        self._check(current, current_valid, 2)
        if action_valid is None:
            action_valid = self.inference_action_valid(current, current_valid, ids, action.shape[1])
        self._check(action, action_valid, 3)
        (arm, arm_valid) = self.arms.decode_predictions(
            action[..., :20],
            current[..., :20],
            current_valid[..., :20],
            ids,
            action_valid[..., :20],
        )
        (lo, hi, constant, active) = self._waist_stats(ids, "action", 3)
        valid = action_valid[..., 20:] & active & current_valid[:, None, 20:]
        waist = self.arms._denormalize(action[..., 20:].float(), lo, hi, constant)
        return (
            torch.cat((arm, torch.where(valid, waist, 0)), -1),
            torch.cat((arm_valid, valid), -1),
        )
