"""Closed target-owner registry for State20 target semantics."""

from __future__ import annotations

from zvla_runtime.libero_action20 import LIBERO_STATE20_TARGET_SEMANTICS

ROBOT_STATE20_TARGET_CONTRACT_ID = "robot-observation-state-offset1-v1"
LIBERO_STATE20_TARGET_CONTRACT_ID = "libero-expert-osc-command-offset0-v1"

ROBOT_STATE20_TARGET_SEMANTICS = (
    "observation_state_rows_t_plus_1_through_t_plus_16_not_source_action")
STATE20_TARGET_CONTRACTS = {
    ROBOT_STATE20_TARGET_CONTRACT_ID: {
        "target_owner": "observation_state",
        "target_semantics": ROBOT_STATE20_TARGET_SEMANTICS,
        "target_offset_start": 1,
        "source_action_arrays_used": False,
        "source_joint_arrays_used": False,
    },
    LIBERO_STATE20_TARGET_CONTRACT_ID: {
        "target_owner": "expert_osc_command",
        "target_semantics": LIBERO_STATE20_TARGET_SEMANTICS,
        "target_offset_start": 0,
        "source_action_arrays_used": True,
        "source_joint_arrays_used": False,
    },
}


def state20_target_contract_registry() -> list[dict[str, object]]:
    """Return the canonical JSON representation in stable ID order."""
    return [
        {"id": identifier, **STATE20_TARGET_CONTRACTS[identifier]}
        for identifier in (
            ROBOT_STATE20_TARGET_CONTRACT_ID,
            LIBERO_STATE20_TARGET_CONTRACT_ID,
        )
    ]


LIBERO_NATIVE7_TARGET_CONTRACT_ID = "libero-native-osc7-offset0-v1"
STATE20_TARGET_CONTRACTS[LIBERO_NATIVE7_TARGET_CONTRACT_ID] = {
    "target_owner": "expert_native_osc7_command",
    "target_semantics": "original_libero_osc7_action_j_conditioned_on_obs_j_minus_1",
    "target_offset_start": 0,
    "source_action_arrays_used": True,
    "source_joint_arrays_used": False,
}


LIBERO_OPENVLA_NATIVE7_TARGET_CONTRACT_ID = "libero-openvla-native-osc7-preaction-tail-v2"
STATE20_TARGET_CONTRACTS[LIBERO_OPENVLA_NATIVE7_TARGET_CONTRACT_ID] = {
    **STATE20_TARGET_CONTRACTS[LIBERO_NATIVE7_TARGET_CONTRACT_ID],
    "target_semantics": "openvla_modified_rlds_action_j_conditioned_on_preaction_obs_j",
}


def mixed_native7_target_contract_registry(*, openvla=False):
    return [{"id": identifier, **STATE20_TARGET_CONTRACTS[identifier]}
            for identifier in (ROBOT_STATE20_TARGET_CONTRACT_ID,
                               LIBERO_OPENVLA_NATIVE7_TARGET_CONTRACT_ID if openvla else LIBERO_NATIVE7_TARGET_CONTRACT_ID)]


__all__ = [
    "LIBERO_STATE20_TARGET_CONTRACT_ID",
    "LIBERO_STATE20_TARGET_SEMANTICS",
    "ROBOT_STATE20_TARGET_CONTRACT_ID",
    "ROBOT_STATE20_TARGET_SEMANTICS",
    "STATE20_TARGET_CONTRACTS",
    "state20_target_contract_registry",
]
