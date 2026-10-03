"""Prepared benchmark episodes with native-rate actions and causal video anchors."""
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from zvla.engines import DATASETS


@DATASETS.register_module()
class BenchmarkDataset(Dataset):
    padding_index = -1
    state_trajectory_contract = True
    geometry_production_mode = 'offline'
    camera_dropout_prob = 0.

    def __init__(self, data_root_path, sampler=None, drop_last=False, seed=0, max_open_shards_per_worker=8):
        self.root = Path(data_root_path)
        self.manifest = json.loads((self.root / 'dataset.json').read_text())
        self.shards = self.manifest['episodes']
        self.text = np.load(self.root / 'text_embeddings.npy', mmap_mode='r')
        self.sampler, self.seed = sampler, seed
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._open, self.max_open = OrderedDict(), max_open_shards_per_worker
        self.position_indices_by_shard, self.stops = [], []
        total = 0
        for row in self.shards:
            count = (row['native_rows'] + 3) // 4
            self.position_indices_by_shard.append(range(total, total + count))
            total += count
            self.stops.append(total)
        self.dataset_fingerprint = hashlib.sha256((self.root / 'dataset.json').read_bytes()).hexdigest()

    def __len__(self):
        return self.stops[-1]

    def set_epoch(self, epoch):
        self._epoch.fill_(epoch)

    def prefill(self, shards):
        for i in shards:
            self._arrays(i)

    def update_prefetch(self, current, future):
        pass

    def _arrays(self, i):
        if i in self._open:
            self._open.move_to_end(i)
            return self._open[i]
        root = self.root / self.shards[i]['shard']
        arrays = {name: np.load(root / (name + '.npy'), mmap_mode='r') for name in
                  ('states', 'targets', 'state_valid', 'target_valid', 'video', 'geometry')}
        self._open[i] = arrays
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return arrays

    def __getitem__(self, index):
        sentinel = index == -1
        index = max(index, 0)
        i = bisect_right(self.stops, index)
        anchor = index - (self.stops[i-1] if i else 0)
        row, arrays = self.shards[i], self._arrays(i)
        start = 4 * anchor
        ncam = self.manifest['camera_count']
        spatial = np.zeros((42, 14), bool)
        spatial[:14*ncam] = True
        geom_valid = np.zeros(147, bool)
        geom_valid[:49*ncam] = True
        targets = np.zeros((16, 20), np.float32)
        valid = np.zeros((16, 20), bool)
        count = min(16, row['native_rows'] - start)
        targets[:count] = arrays['targets'][start:start+count]
        valid[:count] = arrays['target_valid'][start:start+count]
        future = np.zeros((4, 48, 42, 14), np.float16)
        fv = np.zeros((4, 42, 14), bool)
        count = min(4, len(arrays['video']) - anchor - 1)
        future[:count] = arrays['video'][anchor+1:anchor+1+count]
        fv[:count] = spatial
        text_ids = row['text_ids']
        key = json.dumps([self.seed, row['shard'], int(self._epoch)], separators=(',', ':')).encode()
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
        text_id = text_ids[draw % len(text_ids)]
        if sentinel:
            targets.fill(0); valid.fill(False); future.fill(0); fv.fill(False)
        return dict(
            observation_latents=torch.from_numpy(np.array(arrays['video'][anchor:anchor+1])),
            observation_valid=torch.ones(1, dtype=torch.bool),
            observation_spatial_valid=torch.from_numpy(spatial[None]),
            future_latents=torch.from_numpy(future), future_spatial_valid=torch.from_numpy(fv),
            text_emb=torch.from_numpy(np.array(self.text[text_id])),
            current_state_abs=torch.from_numpy(np.array(arrays['states'][start])),
            current_state_abs_valid=torch.from_numpy(np.array(arrays['state_valid'][start])),
            target_state_abs=torch.from_numpy(targets), target_state_abs_valid=torch.from_numpy(valid),
            primitive_dt=torch.tensor(self.manifest['dt'], dtype=torch.float32),
            video_dt=torch.tensor(4*self.manifest['dt'], dtype=torch.float32),
            embodiment_id=torch.tensor(0), source_id=torch.tensor(0),
            geometry_features=torch.from_numpy(np.array(arrays['geometry'][anchor]).reshape(147, 2048)),
            geometry_valid=torch.from_numpy(geom_valid))
