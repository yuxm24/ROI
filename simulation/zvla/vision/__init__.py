"""Semantic camera contracts shared by training and inference."""

from .schema import (  # noqa: F401
    CameraProfile,
    CameraSchema,
    build_camera_schema,
    resolve_camera_profile,
)
from .letterbox import (  # noqa: F401
    CANONICAL_IMAGE_SIZE,
    CANONICAL_LETTERBOX_CONTRACT,
    CANONICAL_PAD_RGB,
    RAW_IMAGE_PROVENANCE_CONTRACT,
    VAE_IMAGE_CONTRACT,
    VGGT_IMAGE_CONTRACT,
    LetterboxGeometry,
    fam_wam_raw_is_preletterboxed,
    resize_canonical_canvas,
    resize_with_pad_rgb,
    validate_fam_wam_raw_letterbox,
)
from .video_frames import SequentialVideoFrames  # noqa: F401
from .wan_vae import (  # noqa: F401
    WAN_VAE_LATENT_CHANNELS,
    WAN_VAE_LATENT_SIZE,
    encode_wan22_vae_camera_windows,
    encode_wan22_vae_cameras,
    load_wan22_vae,
    wan_vae_rgb_window_bounds,
)
from .vggt_features import (  # noqa: F401
    VGGT_FEATURE_DIM,
    VGGT_GIT_REVISION,
    VGGT_MODEL_REVISION,
    VGGT_TOKENS_PER_CAMERA,
    encode_vggt_views,
    load_vggt_aggregator,
    validate_vggt_assets,
)
