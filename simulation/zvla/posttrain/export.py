"""Export neural weights and the fitted benchmark codec."""
import json
from pathlib import Path
from zvla.models.zerith_va import ZerithVAConfig, load_policy_from_model_artifact, export_deployment_artifact


def export(args):
    cfg = json.loads((Path(args.run)/'config.json').read_text())
    data = Path(cfg['data']['train']['dataset']['data_root_path'])
    dataset = json.loads((data/'dataset.json').read_text())
    policy = load_policy_from_model_artifact(args.checkpoint, ZerithVAConfig.from_dict(cfg['model']['config']))
    export_deployment_artifact(policy, args.output, action_schema={'benchmark': dataset['benchmark']},
        camera_schema={'camera_count': dataset['camera_count']},
        normalization_stats=json.loads((data/'codec.json').read_text()),
        vae_encoder_provenance={'model': 'Wan2.2-TI2V-5B'})
