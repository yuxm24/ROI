model = {'type': 'ZerithVAPostTrainModel',
 'pretrained_checkpoint': 'checkpoints/zvla-base.safetensors',
 'state_codec_stats': 'data/metaworld/codec.json',
 'config': {'state_codec_embodiments': ('metaworld_sawyer_applied4',),
            'training_source_names': ('metaworld',),
            'metaworld_native4': True}}

data = {'train': {'dataset': {'type': 'BenchmarkDataset',
                       'data_root_path': 'data/metaworld',
                       'drop_last': False,
                       'sampler': {'type': 'EpisodeBalancedBatchSampler',
                                   'steps': 50000}},
           'collator': {'type': 'ZerithVAStateTrajectoryGeometryRobotCollator'},
           'per_device_batch_size': 10,
           'num_workers': 8,
           'prefetch_factor': 2,
           'pin_memory': True}}

distributed = {'strategy': 'megatron_fsdp', 'mixed_precision': 'bf16'}

runtime = {}

training = {'run_name_prefix': 'zvla-metaworld',
 'run_seed': 0,
 'learning_rate': 1e-05,
 'optimizer': {'type': 'adamw', 'betas': (0.9, 0.95)},
 'weight_decay': 0.01,
 'grad_clip': 1.0,
 'max_steps': 50000,
 'grad_accumulation_steps': 1,
 'scheduler': {'type': 'cosine', 'warmup_ratio': 0.05, 'min_lr_ratio': 0.1},
 'checkpoint': {'save_step_interval': 1000,
                'max_keep': 2,
                'save_full_model': True,
                'format': 'sharded',
                'export_model_steps': (50000,),
                'model_artifact_max_keep': 2},
 'metric': {'type': 'VLAMetric', 'active_trackers': ('jsonl',)},
 'log_every_n_steps': 50,
 'source_metrics_every_n_steps': 500,
 'diagnostics_every_n_steps': 1000,
 'torch_compile': False}
