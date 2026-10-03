"""Load ZWAM policy bundles and manage causal visual inference sessions."""

from pathlib import Path

import numpy as np

from .assets import check_visual_assets, load_text_bank, read_json, valid_digest, verified_file
from .contract import CAMERAS, images_rgb, state23, validate_h1_command_contract
from .encoder_contract import validate_encoder_contract


def move_policy(policy, device):
    import torch
    from zwam_core.models.fam_wam.primitives import FP32Linear

    fp32 = {
        id(p)
        for m in policy.modules()
        if isinstance(m, FP32Linear)
        for p in m.parameters(recurse=False)
    }
    for p in policy.parameters():
        dtype = torch.float32 if id(p) in fp32 else torch.bfloat16
        p.data = p.data.to(device=device, dtype=dtype)
    for module in policy.modules():
        for key, value in tuple(module._buffers.items()):
            if value is not None:
                module._buffers[key] = value.to(device=device)
    return policy.eval()


def load_policy(root, device):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("format") != "zwam-h1-inference-v1":
        raise ValueError("Expected zwam-h1-inference-v1 bundle; see 'zwam prepare --help'")
    stats = manifest.get("normalization_stats")
    validate_h1_command_contract(stats.get("command_contract") if isinstance(stats, dict) else None)
    validate_encoder_contract(manifest.get("config"))

    import torch
    from safetensors.torch import load_file
    from zwam_core.actions.h1 import H1Command23Codec
    from zwam_core.models.fam_wam.config import (
        FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION,
        FaMWAMConfig,
    )
    from zwam_core.models.fam_wam.runtime import FaMWAMV5TeacherPolicy

    config = FaMWAMConfig.from_dict(manifest["config"])
    if (
        config.architecture_version != FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION
        or not config.h1_command23
        or config.action_flow_steps != 10
        or config.action_prediction_steps != 16
        or config.action_chunk_size != 4
        or config.text_prefix_length != 64
    ):
        raise ValueError("Policy must use H1 Command23, text64, solver10/predict16/execute4")
    path = verified_file(root, "model.safetensors", manifest["weights_sha256"])
    stats = manifest["normalization_stats"]["arms_minmax"]
    row = stats["embodiments"].index("h1")
    for name, limits in (
        ("state_min", [0, 0]),
        ("state_max", [1, 1]),
        ("action_min", [0, 0]),
        ("action_max", [1.5, 1.5]),
    ):
        if [stats[name][row][i] for i in (9, 19)] != limits:
            raise ValueError("Checkpoint gripper ranges differ from the H1 state/action contract")
    # Meta construction avoids a second ~27GB initialized FP32 parameter copy.
    with torch.device("meta"):
        policy = FaMWAMV5TeacherPolicy(config)
    weights = load_file(str(path), device="cpu")
    expected = policy.state_dict()
    if expected.keys() != weights.keys() or any(
        expected[k].shape != weights[k].shape for k in expected
    ):
        raise ValueError("Checkpoint keys/shapes do not match this inference architecture")
    policy.load_state_dict(weights, strict=True, assign=True)
    codec = H1Command23Codec(
        manifest["normalization_stats"], embodiments=config.state_codec_embodiments
    )
    for key, value in codec.state_dict().items():
        actual = policy.state_codec.state_dict()[key]
        if value.dtype != actual.dtype or not torch.equal(value, actual):
            raise ValueError(f"Normalization/checkpoint mismatch: {key}")
    policy.world.wan.mark_official_weights_loaded()
    return move_policy(policy, device), manifest


class ModelSession:
    def __init__(self, policy, visual, geometry, text_lookup):
        self.policy, self.visual, self.geometry = policy, visual, geometry
        self.text_lookup = text_lookup
        self.active = False
        self.sequence = 0

    def reset(self, *, prompt, seed):
        import torch

        self.active = False
        text = np.asarray(self.text_lookup(prompt))
        if text.shape != (64, 4096) or text.dtype != np.float16 or not np.isfinite(text).all():
            raise ValueError("Exact frozen text64 embedding required")
        self.policy.reset_inference(seed)
        self.visual.reset()
        self.geometry.reset()
        self.device = next(self.policy.parameters()).device
        self.text = torch.from_numpy(text.copy())[None].to(self.device)
        self.sequence, self.active = 0, True

    def infer(self, images, state, valid, *, camera_names):
        import torch

        if not self.active or tuple(camera_names) != CAMERAS:
            raise ValueError("Inactive session or mismatched camera order")
        images_rgb(images, 1 if self.sequence == 0 else 4)
        state23(state, valid)
        try:
            with torch.inference_mode():
                latent, spatial = self.visual.encode(images, camera_names)
                # Preserve the original cached-feature precision boundary.
                latent = latent.to(torch.float16)
                if not torch.isfinite(latent).all():
                    raise ValueError("VAE features overflow FP16")
                geometry, gvalid = self.geometry.encode(images, camera_names)
                device = self.device
                target, mask = self.policy.act_state_trajectory(
                    latent.to(dtype=next(self.policy.parameters()).dtype),
                    spatial,
                    self.text,
                    torch.tensor([1 / 30], device=device),
                    torch.tensor([4 / 30], device=device),
                    torch.from_numpy(state.copy())[None].to(device),
                    torch.from_numpy(valid.copy())[None].to(device),
                    torch.tensor(
                        [self.policy.config.state_codec_embodiments.index("h1")], device=device
                    ),
                    geometry_features=geometry,
                    geometry_valid=gvalid,
                )
                self.sequence += 1
                return target[0, :4].float().cpu().numpy(), mask[0, :4].cpu().numpy()
        except Exception:
            self.active = False
            raise


def load_session(config):
    manifest = read_json(Path(config["artifact"]) / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("format") != "zwam-h1-inference-v1":
        raise ValueError("Expected zwam-h1-inference-v1 bundle; see 'zwam prepare --help'")
    stats = manifest.get("normalization_stats")
    validate_h1_command_contract(stats.get("command_contract") if isinstance(stats, dict) else None)
    validate_encoder_contract(manifest.get("config"))

    import torch
    from zwam_core.vision.schema import CameraProfile
    from zwam_core.vision.streaming import StreamingVGGTGeometryEncoder, StreamingWanVAEEncoder
    from zwam_core.vision.wan_vae import load_wan22_vae

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Full model inference requires a compatible CUDA GPU")
    assets = read_json(config["visual_assets"])
    # Validate every source and model file before importing external code.
    visual_files = check_visual_assets(assets)
    vae_path = visual_files["Wan2.2_VAE.pth"]
    policy, manifest = load_policy(config["artifact"], device)
    if valid_digest(manifest["vae_sha256"]) != valid_digest(assets["vae_sha256"]):
        raise ValueError("Checkpoint/VAE mismatch")
    tasks, lookup = load_text_bank(config["text_assets"])
    profile = CameraProfile("h1", CAMERAS, (0, 1, 2), ("head", "left_wrist", "right_wrist"))
    visual = StreamingWanVAEEncoder(
        load_wan22_vae(vae_path, assets["wan_source"], device), profile, device
    )
    geometry = StreamingVGGTGeometryEncoder(
        Path(assets["vggt_source"]), Path(assets["vggt_weights"]), profile, device
    )
    return ModelSession(policy, visual, geometry, lookup), tasks
