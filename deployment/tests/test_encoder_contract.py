"""Fixed encoder declarations can be checked before importing model code."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest
from zwam import encoder_contract
from zwam.encoder_contract import validate_encoder_contract

SUPPORTED_INTERFACE = {
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


def test_omitted_interface_fields_match_production_config_defaults():
    # The structural config module uses only the standard library, including
    # when installed from a wheel. Keep its defaults aligned with preflight.
    from zwam_core.models.zerith_va.config import ZerithVAConfig

    defaults = ZerithVAConfig().as_dict()
    expected = {**SUPPORTED_INTERFACE, "text_length": 64}
    assert {name: defaults[name] for name in expected} == expected
    validate_encoder_contract({})
    validate_encoder_contract(defaults)


@pytest.mark.parametrize("field,expected", SUPPORTED_INTERFACE.items())
def test_incompatible_encoder_field_is_rejected(field, expected):
    config = {**SUPPORTED_INTERFACE, field: expected + 1}
    original = copy.deepcopy(config)
    with pytest.raises(ValueError, match=field):
        validate_encoder_contract(config)
    assert config == original


@pytest.mark.parametrize("value", [None, True, False, 4096.0, "4096", [4096]])
def test_explicit_encoder_dimension_must_be_a_json_integer(value):
    with pytest.raises(ValueError, match="text_dim"):
        validate_encoder_contract({"text_dim": value})


@pytest.mark.parametrize("length", [64, 65, 512, 4096])
def test_logical_text_context_may_extend_the_fixed_prefix(length):
    config = {
        **SUPPORTED_INTERFACE,
        "text_length": length,
        "student_dim": 32,
        "reader_dim": 32,
        "teacher_layers": 4,
        "reader_layers": 2,
        "camera_rope_bases": [0, 4, 8],
    }
    original = copy.deepcopy(config)
    validate_encoder_contract(config)
    assert config == original


@pytest.mark.parametrize("value", [None, True, False, 0, 63, 64.0, "64", [64]])
def test_logical_text_length_must_be_an_integer_at_least_64(value):
    with pytest.raises(ValueError, match="text_length"):
        validate_encoder_contract({"text_length": value})


@pytest.mark.parametrize("config", [None, [], "config", 0])
def test_config_must_be_a_json_object(config):
    with pytest.raises(ValueError, match="JSON object"):
        validate_encoder_contract(config)


def test_preflight_imports_no_numeric_model_or_gpu_dependencies():
    # Use the imported installation's package root, so this also exercises an
    # extracted wheel rather than reaching back into the source checkout.
    shared_source = Path(encoder_contract.__file__).resolve().parents[1]
    script = """
import builtins
import sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def pure_import(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'torch', 'safetensors', 'zwam_core', 'wan', 'vggt'}:
        raise AssertionError('Encoder preflight imported model dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = pure_import
from zwam.encoder_contract import validate_encoder_contract
validate_encoder_contract({})
try:
    validate_encoder_contract({'text_dim': 12})
except ValueError:
    pass
else:
    raise AssertionError('Unsupported text width was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(shared_source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
