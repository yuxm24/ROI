"""Validate the fixed external encoder interface without model dependencies."""

_FIXED_ENCODER_FIELDS = {
    "text_dim": 4096,
    "text_prefix_length": 64,
    "latent_channels": 48,
    "canonical_latent_height": 42,
    "canonical_latent_width": 14,
    "geometry_feature_dim": 2048,
    "geometry_tokens_per_camera": 49,
    "camera_slots": 3,
    "max_active_camera_slots": 3,
    "patch_size": 2,
}
_DEFAULT_TEXT_LENGTH = 64


def validate_encoder_contract(config):
    """Require the deployed UMT5/Wan2.2/VGGT input and output dimensions.

    ``config`` is the JSON configuration object, or ``ZerithVAConfig.as_dict()``.
    Omitted interface fields use the production ZerithVAConfig defaults above;
    explicit values must be JSON integers. Logical text contexts may contain
    at least 64 rows because the model supports a compact 64-row prefix with
    a logical zero-padded suffix. Other model fields remain the responsibility
    of the structural configuration parser. This function never changes config.
    """
    if not isinstance(config, dict):
        raise ValueError("Encoder interface configuration must be a JSON object")
    for name, expected in _FIXED_ENCODER_FIELDS.items():
        value = config.get(name, expected)
        if type(value) is not int or value != expected:
            raise ValueError(f"Encoder interface requires {name}={expected}; got {value!r}")
    text_length = config.get("text_length", _DEFAULT_TEXT_LENGTH)
    if type(text_length) is not int or text_length < _DEFAULT_TEXT_LENGTH:
        raise ValueError(
            f"Encoder interface requires text_length to be an integer >=64; got {text_length!r}"
        )
