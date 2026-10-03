"""Training batches partitioned across the requested GPU count."""
import torch
from torch.utils.data import DataLoader
from zvla.engines import build_dataset_from_cfg, build_collator_from_cfg
from zvla.engines.utils.torch_utils import worker_init_function
from .sampler_builder import build_sampler_from_cfg


def build_train_dataloader(dataset_cfg, collator_cfg, per_device_batch_size, num_workers=0,
                           prefetch_factor=2, pin_memory=True, strategy=None, seed=0):
    dataset = build_dataset_from_cfg(dataset_cfg)
    sampler = build_sampler_from_cfg(dataset_cfg['sampler'], dataset, per_device_batch_size,
        strategy.world_size() if strategy else 1, strategy.rank() if strategy else 0, seed, dataset_cfg)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=build_collator_from_cfg(collator_cfg),
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_function,
        persistent_workers=num_workers > 0, prefetch_factor=prefetch_factor if num_workers else None,
        generator=torch.Generator().manual_seed(seed))
