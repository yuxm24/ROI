"""Camera, coordinate and state/action contracts shared by server and client."""

import numpy as np

PROTOCOL = "zwam-h1-command23-http-npz-v3"
H1_POSE_FRAME = "body_yaw_link"
H1_POSITION_UNIT = "m"
H1_WAIST_UNITS = ("m", "rad", "rad")
H1_WAIST_LAYOUT = tuple(f"waist_joint{i}.position" for i in (1, 2, 3))
H1_WAIST_SEMANTICS = ("lift", "pitch", "yaw")
CAMERAS = tuple(f"fam_wam_raw/canonical_view_{i}" for i in range(3))
SDK_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def validate_h1_pose_contract(contract):
    """Check the declared training/command frame without inferring a transform.

    Matching declarations do not establish physical validation. A model provider
    must prepare its observations and absolute hand targets in this frame; this
    function never relabels statistics or converts coordinates.
    """
    if not isinstance(contract, dict) or contract.get("pose_frame") != H1_POSE_FRAME:
        raise ValueError(
            f"H1 pose_frame must be {H1_POSE_FRAME}; export matching training data and "
            "checkpoint statistics rather than relabeling an existing manifest"
        )
    evidence = contract.get("frame_evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("H1 pose frame requires nonempty frame_evidence")


def validate_h1_command_contract(contract):
    """Check deployed pose/waist declarations without converting source values.

    Joint1/2/3 mean lift/pitch/yaw in the versioned Command23 schema. Explicit
    semantic names, when supplied, must agree; legacy correct declarations keep
    their original statistics and checkpoint fingerprint without adding fields.
    """
    validate_h1_pose_contract(contract)
    if contract.get("position_unit") != H1_POSITION_UNIT:
        raise ValueError("H1 position_unit must be m")
    units = contract.get("waist_units")
    if not isinstance(units, list) or units != list(H1_WAIST_UNITS):
        raise ValueError("H1 waist_units must be [m, rad, rad] for lift, pitch, yaw")
    order = contract.get("waist_joint_order")
    if not isinstance(order, list) or order != list(H1_WAIST_LAYOUT):
        raise ValueError("H1 waist_joint_order must be joint1, joint2, joint3 (lift, pitch, yaw)")
    if "waist_semantics" in contract:
        semantics = contract["waist_semantics"]
        if not isinstance(semantics, list) or semantics != list(H1_WAIST_SEMANTICS):
            raise ValueError("H1 waist_semantics must be [lift, pitch, yaw]")


def metadata():
    return dict(
        protocol=PROTOCOL,
        ready=True,
        model_type="zwam",
        action_dim=23,
        action_horizon=4,
        prediction_horizon=16,
        rate_hz=30,
        first_image_frames=1,
        subsequent_image_frames=4,
        camera_names=list(CAMERAS),
        frame=H1_POSE_FRAME,
        rotation="column0_xyz_column1_xyz",
        waist_units=list(H1_WAIST_UNITS),
        gripper_observation_range=[0, 1],
        gripper_command_range=[0, 1.5],
        supports_rtc=False,
        smoothing="none",
    )


def validate_metadata(value):
    if not isinstance(value, dict):
        raise ValueError("Incompatible server contract: expected a metadata object")
    for key, expected in metadata().items():
        if value.get(key) != expected:
            raise ValueError(f"Incompatible server contract: {key}")


def state23(state, valid):
    if (
        state.shape != (23,)
        or state.dtype != np.float32
        or not np.isfinite(state).all()
        or valid.shape != (23,)
        or valid.dtype != np.bool_
        or np.any(state[~valid] != 0)
    ):
        raise ValueError("Expected finite FP32 State23, bool validity, zero invalid entries")


def images_rgb(images, frames):
    if (
        images.dtype != np.uint8
        or images.ndim != 5
        or images.shape[0:2] != (frames, 3)
        or images.shape[-1] != 3
        or not all(1 <= n <= 2048 for n in images.shape[2:4])
    ):
        raise ValueError(f"Expected RGB uint8 [{frames},3,H,W,3], H/W <= 2048")


def commands23(commands, valid):
    if commands.shape != (4, 23) or valid.shape != (4, 23):
        raise ValueError("Expected four Command23 rows")
    for c, v in zip(commands, valid, strict=False):
        state23(c, v)
