"""Exact field sets shared by Zerith VA datasets, collators, and models."""

FAM_ROBOT_SAMPLE_KEYS = frozenset({
    "observation_latents",
    "observation_valid",
    "observation_spatial_valid",
    "current_state",
    "current_state_valid",
    "target_action",
    "target_action_valid",
    "future_latents",
    "future_spatial_valid",
    "text_emb",
    "primitive_dt",
    "video_dt",
})

FAM_GEOMETRY_ROBOT_SAMPLE_KEYS = FAM_ROBOT_SAMPLE_KEYS | frozenset({
    "geometry_features",
    "geometry_valid",
})

# Production benchmark training accepts exactly one raw action
# representation: metric absolute end-effector targets.  The model-owned pose
# codec converts these fields into its private relative Action32 coordinates.
FAM_ABSOLUTE_POSE_ROBOT_SAMPLE_KEYS = (
    FAM_ROBOT_SAMPLE_KEYS
    - frozenset({
        "current_state",
        "current_state_valid",
        "target_action",
        "target_action_valid",
    })
    | frozenset({
        "current_eef_pose_abs",
        "current_eef_pose_abs_valid",
        "target_eef_pose_abs",
        "target_eef_pose_abs_valid",
    })
)

FAM_ABSOLUTE_POSE_GEOMETRY_ROBOT_SAMPLE_KEYS = (
    FAM_ABSOLUTE_POSE_ROBOT_SAMPLE_KEYS
    | frozenset({"geometry_features", "geometry_valid"})
)

# State20 boundary: the anchor is always an absolute observation state.
# A sealed per-shard target contract selects either robot observation rows
# t+1..t+16 or causally transcoded LIBERO OSC command rows t..t+15;
# source_id remains telemetry only and never changes loss weighting.
FAM_STATE_TRAJECTORY_ROBOT_SAMPLE_KEYS = (
    FAM_ROBOT_SAMPLE_KEYS
    - frozenset({
        "current_state",
        "current_state_valid",
        "target_action",
        "target_action_valid",
    })
    | frozenset({
        "embodiment_id",
        "source_id",
        "current_state_abs",
        "current_state_abs_valid",
        "target_state_abs",
        "target_state_abs_valid",
    })
)

FAM_STATE_TRAJECTORY_GEOMETRY_ROBOT_SAMPLE_KEYS = (
    FAM_STATE_TRAJECTORY_ROBOT_SAMPLE_KEYS
    | frozenset({"geometry_features", "geometry_valid"})
)

FAM_VIDEO_SAMPLE_KEYS = frozenset({
    "observation_latents",
    "observation_valid",
    "observation_spatial_valid",
    "future_latents",
    "future_spatial_valid",
    "text_emb",
    "primitive_dt",
    "video_dt",
})

FAM_PAIRED_SAMPLE_KEYS = frozenset({"robot", "video"})
