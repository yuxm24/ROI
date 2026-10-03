"""Single-episode HTTP inference with causal video state."""
import base64
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from .recipes import BENCHMARKS
from .sources import libero_state


def serve(args):
    from flask import Flask, request, jsonify
    from zvla.models.fam_wam import load_deployment_artifact
    from zvla.vision import CameraProfile, load_wan22_vae
    from zvla.vision.wan_assets import _load_text_encoder
    from .visual import StreamingWanVAEEncoder, StreamingVGGTGeometryEncoder
    device = torch.device(args.device)
    policy, manifest = load_deployment_artifact(args.artifact, device=device, dtype=torch.bfloat16)
    bench = manifest['action_schema']['benchmark']; spec = BENCHMARKS[bench]
    profile = CameraProfile(bench, spec['cameras'], tuple(range(len(spec['cameras']))), ('front','left','right'))
    vae = load_wan22_vae(Path(args.wan_model)/'Wan2.2_VAE.pth', args.wan_repo, device)
    visual = StreamingWanVAEEncoder(vae, profile, device)
    geometry = StreamingVGGTGeometryEncoder(Path(args.vggt_repo), Path(args.vggt_model), profile, device)
    text_encoder = _load_text_encoder(Path(args.wan_model), Path(args.wan_repo), torch.device('cpu'))
    @lru_cache(maxsize=256)
    def encode_text(prompt):
        text_encoder.model.to(device)
        with torch.inference_mode():
            value = text_encoder([prompt], device)[0]
        text_encoder.model.cpu()
        if len(value) > 64:
            raise ValueError('Instruction exceeds the 64-token model context')
        context = torch.zeros((1,64,4096), device=device, dtype=torch.float16)
        context[0,:len(value)] = value
        return context
    current = {}
    app = Flask(__name__)
    @app.get('/health')
    def health():
        return jsonify(benchmark=bench)
    @app.post('/reset')
    def reset():
        value = request.get_json()
        visual.reset(); policy.reset_inference(int(value.get('seed',0)))
        current['text'] = encode_text(value['prompt'])
        return jsonify(ok=True)
    @app.post('/infer')
    @torch.inference_mode()
    def infer():
        value = request.get_json()
        rgb = np.frombuffer(base64.b64decode(value['images']), np.uint8).reshape(value['shape'])
        state = np.asarray(value['state'], np.float32)
        if bench in ('libero','libero-plus'):
            state = libero_state(state[None])[0]; valid = np.arange(20)<10
        elif bench == 'metaworld':
            from metaworld_benchmark.v52_contract import state20, stored_image
            state = state20(state[:4]); valid = np.isin(np.arange(20), [0,1,2,9])
            rgb = np.stack([stored_image(frame[0]) for frame in rgb])[:,None]
        elif bench == 'robodojo':
            from zvla.actions.robodojo import absolute_ee_row_to_state20
            state = absolute_ee_row_to_state20(state); valid = np.ones(20,bool)
        else:
            valid = np.ones(20,bool)
        latent, spatial = visual.encode(rgb, spec['cameras'])
        geo, gv = geometry.encode(rgb, spec['cameras'])
        pred, _ = policy.act_state_trajectory(latent, spatial, current['text'],
            torch.tensor([spec['dt']],device=device), torch.tensor([spec['dt']*4],device=device),
            torch.from_numpy(state[None]).to(device), torch.from_numpy(valid[None]).to(device),
            torch.zeros(1,dtype=torch.long,device=device), geometry_features=geo,geometry_valid=gv)
        actions = pred[0,:4].float().cpu().numpy()
        if bench in ('libero','libero-plus'): actions = np.clip(actions[:,:7],-1,1)
        elif bench == 'metaworld': actions = np.clip(actions[:,:4],-1,1)
        elif bench == 'robotwin':
            from zvla_runtime.robotwin_action20 import absolute_state_to_official_ee
            actions = absolute_state_to_official_ee(actions)
        else:
            from zvla.actions.robodojo import state20_to_absolute_ee
            actions = state20_to_absolute_ee(actions)
        return jsonify(actions=actions.tolist())
    app.run(host=args.host,port=args.port,threaded=False)
