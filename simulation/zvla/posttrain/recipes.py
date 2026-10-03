"""Benchmark interfaces and suggested optimizer settings."""
from pathlib import Path

BENCHMARKS = {
    'libero': dict(robot='libero_franka', steps=30000, dt=.05, cameras=('front', 'wrist')),
    'libero-plus': dict(robot='libero_franka', steps=30000, dt=.05, cameras=('front', 'wrist')),
    'metaworld': dict(robot='metaworld_sawyer_applied4', steps=50000, dt=.0125, cameras=('corner2',)),
    'robotwin': dict(robot='robotwin_aloha', steps=50000, dt=.06, cameras=('head_camera', 'left_camera', 'right_camera')),
    'robodojo': dict(robot='robodojo_ee', steps=30000, dt=.04, cameras=('head', 'left_wrist', 'right_wrist')),
}


def recipe(bench, data_root='data/prepared', checkpoint='checkpoints/zvla-base.safetensors', batch_size=10):
    spec = BENCHMARKS[bench]
    cfg = dict(
        model=dict(type='FaMWAMPostTrainModel', pretrained_checkpoint=str(checkpoint),
                   state_codec_stats=str(Path(data_root) / 'codec.json'),
                   config=dict(state_codec_embodiments=(spec['robot'],), training_source_names=(bench,),
                               metaworld_native4=bench == 'metaworld')),
        data=dict(train=dict(dataset=dict(type='BenchmarkDataset', data_root_path=str(data_root),
                    drop_last=False, sampler=dict(type='FaMShardAwareBatchSampler', exact_once=True,
                                                 global_batch_mixing=True)),
                    collator=dict(type='FaMStateTrajectoryGeometryRobotCollator'),
                    per_device_batch_size=batch_size, num_workers=8, prefetch_factor=2, pin_memory=True)),
        distributed=dict(strategy='megatron_fsdp', mixed_precision='bf16'),
        runtime={},
        training=dict(run_name_prefix='zvla-'+bench, run_seed=0, learning_rate=1e-5,
            optimizer=dict(type='adamw', betas=(.9, .95)), weight_decay=.01, grad_clip=1.,
            max_steps=spec['steps'], grad_accumulation_steps=1,
            scheduler=dict(type='cosine', warmup_ratio=.05, min_lr_ratio=.1),
            checkpoint=dict(save_step_interval=1000, max_keep=2, save_full_model=True, format='sharded',
                            export_model_steps=(spec['steps'],), model_artifact_max_keep=2),
            metric=dict(type='VLAMetric', active_trackers=('jsonl',)),
            log_every_n_steps=50, source_metrics_every_n_steps=500, diagnostics_every_n_steps=1000,
            torch_compile=False))

    if bench == 'metaworld':
        cfg['data']['train']['dataset']['sampler'] = dict(type='EpisodeBalancedBatchSampler', steps=spec['steps'])
    return cfg
