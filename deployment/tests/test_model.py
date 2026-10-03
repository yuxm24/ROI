import pytest

torch = pytest.importorskip("torch")
from zwam_core.actions.state_trajectory import matrix_to_rotation_6d, rotation_6d_to_matrix
from zwam_core.models.fam_wam.config import FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION, FaMWAMConfig
from zwam_core.models.fam_wam.runtime import FaMWAMV5TeacherPolicy


def tiny_config():
    return FaMWAMConfig.tiny(
        architecture_version=FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION,
        action_dim=23,
        state_dim=23,
        state_trajectory_dim=23,
        state_trajectory_contract=True,
        native7_minmax=True,
        h1_command23=True,
        state_codec_embodiments=("h1",),
        action_prediction_steps=16,
        action_chunk_size=4,
        use_proprioception=True,
    )


def test_no_training_recipe_in_config():
    config = tiny_config()
    assert "training_source_names" not in config.as_dict()
    assert "world_ema_decay" not in config.as_dict()
    assert FaMWAMConfig.from_dict(config.as_dict()) == config


def test_tensor_rotation_contract():
    r = torch.eye(3).repeat(2, 1, 1)
    torch.testing.assert_close(rotation_6d_to_matrix(matrix_to_rotation_6d(r)), r)


def test_tiny_full_policy_forward_and_reset():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = tiny_config()
    policy = FaMWAMV5TeacherPolicy(config).eval()
    state = torch.zeros(1, 23)
    state[:, [3, 7, 13, 17]] = 1
    args = [
        torch.randn(1, 6, 12, 4),
        torch.ones(1, 12, 4, dtype=torch.bool),
        torch.randn(1, 3, 12),
        torch.tensor([1 / 30]),
        torch.tensor([4 / 30]),
        state,
        torch.ones(1, 23, dtype=torch.bool),
        torch.tensor([0]),
    ]
    kwargs = dict(
        geometry_features=torch.randn(1, 12, 24), geometry_valid=torch.ones(1, 12, dtype=torch.bool)
    )
    policy.reset_inference(42)
    action, valid = policy.act_state_trajectory(*args, **kwargs)
    assert action.shape == (1, 16, 23)
    assert torch.isfinite(action).all() and valid.all()
    assert policy._runtime.counters["reader_calls"] == config.action_flow_steps
    policy.reset_inference(42)
    repeated, _ = policy.act_state_trajectory(*args, **kwargs)
    torch.testing.assert_close(action, repeated)


def test_state_dict_strict_roundtrip():
    config = tiny_config()
    policy = FaMWAMV5TeacherPolicy(config)
    clone = FaMWAMV5TeacherPolicy(config)
    clone.load_state_dict(policy.state_dict(), strict=True)
    assert policy.state_dict().keys() == clone.state_dict().keys()


def test_meta_network_cpu_codec_materialization():
    config = tiny_config()
    with torch.device("meta"):
        policy = FaMWAMV5TeacherPolicy(config)
    assert next(policy.parameters()).is_meta
    assert not policy.state_codec.waist_stat_valid.is_meta
    original = FaMWAMV5TeacherPolicy(config)
    policy.load_state_dict(original.state_dict(), strict=True, assign=True)
    assert all(not value.is_meta for value in policy.state_dict().values())
