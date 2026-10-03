"""Build the resumable shard-aware sampler."""
from .shard_samplers import FaMShardAwareBatchSampler


def build_sampler_from_cfg(sampler_cfg, dataset, batch_size, world_size, rank, seed, dataset_cfg):
    cfg = dict(sampler_cfg)
    kind = cfg.pop('type')
    cfg.setdefault('drop_last', False)
    sampler_class = EpisodeBalancedBatchSampler if kind == 'EpisodeBalancedBatchSampler' else FaMShardAwareBatchSampler
    return sampler_class(dataset=dataset, batch_size=batch_size,
        num_replicas=world_size, rank=rank, seed=seed, **cfg)


class EpisodeBalancedBatchSampler:
    """Uniform episode, then uniform anchor; random access supports resuming."""
    def __init__(self, dataset, batch_size, num_replicas, rank, seed=0, steps=50000, drop_last=False):
        self.dataset = dataset
        self.batch_size, self.num_replicas, self.rank = batch_size, num_replicas, rank
        self.seed, self.steps = seed, steps
        self.epoch, self._next_batch_idx = 0, 0

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch):
        self.epoch, self._next_batch_idx = epoch, 0
        self.dataset.set_epoch(epoch)

    def __iter__(self):
        import numpy as np
        for step in range(self._next_batch_idx, self.steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, step, 413]))
            episode_ids = rng.choice(len(self.dataset.shards), self.batch_size*self.num_replicas)
            values = []
            for episode in episode_ids:
                positions = self.dataset.position_indices_by_shard[episode]
                values.append(positions.start + int(rng.integers(len(positions))))
            values = np.asarray(values)[rng.permutation(len(values))].tolist()
            self._next_batch_idx = step+1
            yield values[self.rank*self.batch_size:(self.rank+1)*self.batch_size]
        self._next_batch_idx = 0

    def state_dict(self):
        return dict(epoch=self.epoch, next_batch_idx=self._next_batch_idx)

    def load_state_dict(self, state):
        self.set_epoch(state['epoch'])
        self._next_batch_idx = state['next_batch_idx']
