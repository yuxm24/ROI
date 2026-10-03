# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0.
"""Shard-aware batch samplers for sharded contiguous latent datasets."""

from __future__ import annotations

import hashlib
import json
import random
from bisect import bisect_right
from collections import OrderedDict
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import BatchSampler

from zvla.engines.utils import initialize_overwatch

overwatch = initialize_overwatch(__name__)


class ShardAwareVariableChunkBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        chunk_size_range: Tuple[int, int] = (1, 4),
        window_size_range: Tuple[int, int] = (25, 50),
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shard_shuffle: bool = True,
        position_shuffle: bool = True,
        shuffle: Optional[bool] = None,
        drop_last: bool = True,
        seed: int = 0,
        lookahead_shards: int = 0,
        prefill_shards: int = 0,
        split_by: str = "balanced_shard",
        **kwargs,
    ) -> None:
        if num_replicas is None:
            num_replicas = overwatch.world_size()
        if rank is None:
            rank = overwatch.rank()
        if shuffle is not None:
            shard_shuffle = bool(shuffle)
            position_shuffle = bool(shuffle)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.chunk_size_range = tuple(chunk_size_range)
        self.window_size_range = tuple(window_size_range)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shard_shuffle = bool(shard_shuffle)
        self.position_shuffle = bool(position_shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.lookahead_shards = int(lookahead_shards)
        self.prefill_shards = int(prefill_shards)
        self.split_by = str(split_by)
        self.epoch = 0
        self._cycle = 0
        self._next_batch_idx = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cycle = 0
        self._next_batch_idx = 0

    def _rank_shards(self, cycle: int) -> list[int]:
        shard_indices = list(range(len(self.dataset.shards)))
        rng = random.Random(self.seed + self.epoch * 10000 + cycle)
        if self.shard_shuffle:
            rng.shuffle(shard_indices)
        if self.split_by in {"round_robin", "round-robin"}:
            return shard_indices[self.rank::self.num_replicas]
        if len(shard_indices) < self.num_replicas:
            return shard_indices

        order = {shard_idx: idx for idx, shard_idx in enumerate(shard_indices)}
        sorted_shards = sorted(
            shard_indices,
            key=lambda shard_idx: (-len(self.dataset.positions_by_shard[shard_idx]), order[shard_idx]),
        )
        buckets: list[list[int]] = [[] for _ in range(self.num_replicas)]
        loads = [0 for _ in range(self.num_replicas)]
        for shard_idx in sorted_shards:
            target = min(range(self.num_replicas), key=lambda idx: (loads[idx], idx))
            buckets[target].append(shard_idx)
            loads[target] += len(self.dataset.positions_by_shard[shard_idx])
        return sorted(buckets[self.rank], key=lambda shard_idx: order[shard_idx])

    def _split_positions_within_shards(self) -> bool:
        return (
            self.split_by not in {"round_robin", "round-robin"}
            and len(self.dataset.shards) < self.num_replicas
        )

    def prefill(self) -> None:
        if self.prefill_shards <= 0 or not hasattr(self.dataset, "prefill"):
            return
        shards = self._rank_shards(self._cycle)[: self.prefill_shards]
        self.dataset.prefill(shards)

    def _build_batches(self, cycle: int) -> list[tuple[int, list[tuple]]]:
        batches: list[tuple[int, list[tuple]]] = []
        batch_rng = random.Random(self.seed + self.epoch * 10000 + cycle * 100003 + self.rank)
        for shard_idx in self._rank_shards(cycle):
            positions = list(self.dataset.positions_by_shard[shard_idx])
            if self.position_shuffle:
                rng = random.Random(self.seed + self.epoch * 10000 + cycle + shard_idx * 7919)
                rng.shuffle(positions)
            if self._split_positions_within_shards():
                positions = positions[self.rank::self.num_replicas]
            for offset in range(0, len(positions), self.batch_size):
                batch_positions = positions[offset: offset + self.batch_size]
                if len(batch_positions) < self.batch_size and self.drop_last:
                    continue
                cs = batch_rng.randint(*self.chunk_size_range)
                ws = batch_rng.randint(*self.window_size_range)
                ws = max((ws // cs) * cs, cs)
                augmentation_rng = random.Random(
                    self.seed
                    + self.epoch * 10_000
                    + cycle * 100_003
                    + self.rank * 104_729
                    + shard_idx * 7_919
                    + offset * 1_000_003
                    + 97
                )
                if getattr(
                    self.dataset,
                    "camera_dropout_prob",
                    0.0,
                ) > 0.0:
                    batch = [
                        (
                            shard_idx,
                            ep,
                            t,
                            cs,
                            ws,
                            augmentation_rng.getrandbits(63),
                        )
                        for ep, t in batch_positions
                    ]
                else:
                    batch = [
                        (shard_idx, ep, t, cs, ws)
                        for ep, t in batch_positions
                    ]
                batches.append((shard_idx, batch))
        return batches

    def _future_shards(self, batches: list[tuple[int, list[tuple]]], batch_idx: int) -> list[int]:
        future: list[int] = []
        seen = {batches[batch_idx][0]}
        for idx in range(batch_idx + 1, len(batches)):
            shard_idx = batches[idx][0]
            if shard_idx in seen:
                continue
            future.append(shard_idx)
            seen.add(shard_idx)
            if len(future) >= self.lookahead_shards:
                break
        return future

    def __iter__(self):
        cycle = self._cycle
        while True:
            batches = self._build_batches(cycle)
            if not batches:
                return
            start_idx = self._next_batch_idx if cycle == self._cycle else 0
            last_prefetch_shard: int | None = None
            for idx in range(start_idx, len(batches)):
                shard_idx, batch = batches[idx]
                if shard_idx != last_prefetch_shard and hasattr(self.dataset, "update_prefetch"):
                    self.dataset.update_prefetch(shard_idx, self._future_shards(batches, idx))
                    last_prefetch_shard = shard_idx
                self._cycle = cycle
                self._next_batch_idx = idx + 1
                yield batch
            self._next_batch_idx = 0
            cycle += 1

    def __len__(self) -> int:
        return len(self._build_batches(self._cycle))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cycle": self._cycle,
            "next_batch_idx": self._next_batch_idx,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "split_by": self.split_by,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for key, current in (
            ("batch_size", self.batch_size),
            ("num_replicas", self.num_replicas),
            ("seed", self.seed),
            ("split_by", self.split_by),
        ):
            if state.get(key, current) != current:
                raise ValueError(f"Cannot resume sampler with a different {key}")
        self.epoch = int(state.get("epoch", 0))
        self._cycle = int(state.get("cycle", 0))
        self._next_batch_idx = int(state.get("next_batch_idx", 0))


class DistributedShardAwareVariableChunkBatchSampler(ShardAwareVariableChunkBatchSampler):
    pass


class _StatelessIndexPermutation:
    """Seeded bijection of [0, size), without an O(size) shuffle array.

    Six Feistel rounds permute the enclosing even-bit domain. Cycle walking
    restricts that permutation to [0, size); it cannot duplicate or drop an
    index. The enclosing domain is less than four times size. SplitMix64 is a
    deterministic integer mixer, not a cryptographic/security primitive.
    Changing rounds, keys, or block layout requires a new sampler partition
    format: otherwise an old logical checkpoint cursor would change meaning.
    """

    def __init__(self, size: int, key: str) -> None:
        self.size = int(size)
        if not 0 <= self.size <= (1 << 64):
            raise ValueError("permutation size must fit an unsigned 64-bit domain")
        self.half_bits = max(1, ((max(1, self.size) - 1).bit_length() + 1) // 2)
        self.mask = (1 << self.half_bits) - 1
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=48).digest()
        self.keys = tuple(int.from_bytes(digest[i:i + 8], "little")
                          for i in range(0, 48, 8))

    def __call__(self, index: int) -> int:
        value = int(index)
        if not 0 <= value < self.size:
            raise IndexError("permutation input out of range")
        if self.size == 1:
            return 0
        mask, bits = self.mask, self.half_bits
        while True:
            left, right = value >> bits, value & mask
            for key in self.keys:
                z = (right + key) & 0xffffffffffffffff
                z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & 0xffffffffffffffff
                z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & 0xffffffffffffffff
                left, right = right, left ^ ((z ^ (z >> 31)) & mask)
            value = (left << bits) | right
            if value < self.size:
                return value


class FaMShardAwareBatchSampler(BatchSampler):
    """Shard-aware integer-anchor batches with an optional exact-once mode.

    The legacy mode first makes shard-local batches and equalizes rank lengths
    by repeating a few leading batches.  Production one-pass training uses
    ``exact_once=True`` instead: all real anchors form one deterministic global
    stream, every global step is sliced across ranks, and only the final step
    receives ``-1`` loss-masked sentinels.  Thus no real anchor is omitted or
    repeated even when the corpus is not divisible by the global batch size.

    ``global_batch_mixing=True`` permutes equal-sized local microbatches across
    the entire corpus *before* assigning them to ranks. A global batch therefore
    mixes tasks even when every storage shard is task-pure. Most local batches
    still touch one shard, retaining I/O locality. The single short microbatch
    remains at the end; padding occurs only in the final global step.

    Mixed mode is lazy and uses O(number of shards) state, including stateless
    within-shard permutations. It never boxes all anchors, constructs all epoch
    batches, or repeatedly shuffles a million-anchor shard on cache misses.
    Legacy modes and their checkpoint formats remain unchanged.
    """

    PARTITION_FORMAT = "contiguous_global_shard_batches_equalized_v1"
    EXACT_ONCE_PARTITION_FORMAT = "global_anchor_stream_exact_once_padded_v2"
    MIXED_EXACT_ONCE_PARTITION_FORMAT = "global_microbatch_permutation_exact_once_v1"

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        lookahead_shards: int = 0,
        prefill_shards: int = 0,
        exact_once: bool = False,
        global_batch_mixing: bool = False,
        **kwargs,
    ) -> None:
        if kwargs:
            raise TypeError(f"unsupported FaM sampler options: {sorted(kwargs)}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.lookahead_shards = int(lookahead_shards)
        self.prefill_shards = int(prefill_shards)
        self.exact_once = bool(exact_once)
        self.global_batch_mixing = bool(global_batch_mixing)
        self.epoch = 0
        self._next_batch_idx = 0
        self._exact_layout_cache = None
        self._exact_shard_index_cache: OrderedDict[int, list[int]] = (
            OrderedDict())
        self._mixed_block_permutation = None
        self._mixed_shard_permutations = {}
        if self.batch_size <= 0 or self.num_replicas <= 0:
            raise ValueError("batch_size and num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if self.global_batch_mixing:
            if not self.exact_once:
                raise ValueError("global_batch_mixing requires exact_once=True")
            if getattr(dataset, "geometry_production_mode", None) == "rolling":
                raise ValueError("global_batch_mixing requires complete offline geometry")
        if self.exact_once:
            if self.drop_last:
                raise ValueError(
                    "exact_once requires drop_last=False; its final global "
                    "step uses loss-masked sentinel samples")
            if int(getattr(self.dataset, "padding_index", -1)) != -1:
                raise ValueError(
                    "exact_once requires dataset.padding_index == -1")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._next_batch_idx = 0
        self._exact_layout_cache = None
        self._exact_shard_index_cache.clear()
        self._mixed_block_permutation = None
        self._mixed_shard_permutations.clear()
        self.dataset.set_epoch(self.epoch)

    def _ordered_shards_for_epoch(self, epoch: int) -> list[int]:
        shards = list(range(len(self.dataset.shards)))
        if self.shuffle:
            random.Random(self.seed + int(epoch) * 10007).shuffle(shards)
        return shards

    def _ordered_shards(self) -> list[int]:
        return self._ordered_shards_for_epoch(self.epoch)

    def _shard_order_sha256(self, epoch: int) -> str:
        """Fingerprint the exact ordered shard/anchor schedule.

        Names catch inventory replacement/reordering and anchor counts catch
        changes to episode boundaries or training horizon.  This same
        canonical representation is emitted by the offline VGGT producer, so
        a checkpoint watermark can never reclaim a differently scheduled
        sidecar by accident.
        """
        values = []
        for shard in self._ordered_shards_for_epoch(epoch):
            entry = self.dataset.shards[shard]
            values.append({
                "shard": str(entry["shard"]),
                "anchors": len(self.dataset.position_indices_by_shard[shard]),
            })
        payload = json.dumps(
            values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _exact_layout(self) -> tuple[list[int], list[int], int]:
        if self._exact_layout_cache is not None:
            return self._exact_layout_cache
        order = self._ordered_shards()
        cumulative_stops = []
        cursor = 0
        for shard in order:
            cursor += len(self.dataset.position_indices_by_shard[shard])
            cumulative_stops.append(cursor)
        self._exact_layout_cache = (order, cumulative_stops, cursor)
        return self._exact_layout_cache

    def _exact_shard_indices(self, shard: int) -> list[int]:
        indices = self._exact_shard_index_cache.pop(shard, None)
        if indices is None:
            indices = list(self.dataset.position_indices_by_shard[shard])
            if self.shuffle:
                random.Random(
                    self.seed + self.epoch * 10007 + shard * 7919
                ).shuffle(indices)
        self._exact_shard_index_cache[shard] = indices
        while len(self._exact_shard_index_cache) > 4:
            self._exact_shard_index_cache.popitem(last=False)
        return indices

    def _exact_steps(self) -> int:
        anchors = self._exact_layout()[2]
        global_batch = self.batch_size * self.num_replicas
        return (anchors + global_batch - 1) // global_batch

    def _exact_local_batch(
        self,
        step: int,
    ) -> tuple[tuple[int, ...], list[int]]:
        order, cumulative_stops, anchors = self._exact_layout()
        global_batch = self.batch_size * self.num_replicas
        start = step * global_batch + self.rank * self.batch_size
        stop = min(start + self.batch_size, anchors)
        batch: list[int] = []
        batch_shards: list[int] = []
        cursor = start
        while cursor < stop:
            ordered_index = bisect_right(cumulative_stops, cursor)
            shard = order[ordered_index]
            shard_start = (
                0 if ordered_index == 0
                else cumulative_stops[ordered_index - 1])
            take = min(
                stop - cursor,
                cumulative_stops[ordered_index] - cursor,
            )
            local_start = cursor - shard_start
            indices = self._exact_shard_indices(shard)
            batch.extend(indices[local_start:local_start + take])
            if not batch_shards or batch_shards[-1] != shard:
                batch_shards.append(shard)
            cursor += take
        batch.extend([-1] * (self.batch_size - len(batch)))
        return tuple(batch_shards), batch

    def _exact_batches(self):
        return [
            self._exact_local_batch(step)
            for step in range(self._exact_steps())
        ]

    def _mixed_stream_span(self, step: int, rank: Optional[int] = None):
        """Map one rank/step to an unpadded microbatch of the global stream."""
        anchors = self._exact_layout()[2]
        slot = step * self.num_replicas + (self.rank if rank is None else rank)
        full_blocks = anchors // self.batch_size
        if slot < full_blocks:
            if self.shuffle:
                if self._mixed_block_permutation is None:
                    self._mixed_block_permutation = _StatelessIndexPermutation(
                        full_blocks, f"fam-taskmix-v1:{self.seed}:{self.epoch}:blocks")
                slot = self._mixed_block_permutation(slot)
            start = slot * self.batch_size
        else:
            # Do not shuffle the one partial block into a full global step.
            start = min(slot * self.batch_size, anchors)
        return start, min(start + self.batch_size, anchors)

    def _mixed_stream_pieces(self, step: int, rank: Optional[int] = None):
        order, stops, _ = self._exact_layout()
        cursor, end = self._mixed_stream_span(step, rank)
        while cursor < end:
            i = bisect_right(stops, cursor)
            shard_start = stops[i - 1] if i else 0
            take = min(stops[i], end) - cursor
            yield order[i], cursor - shard_start, take
            cursor += take

    def _mixed_local_batch(self, step: int):
        batch, shards = [], []
        for shard, offset, take in self._mixed_stream_pieces(step):
            indices = self.dataset.position_indices_by_shard[shard]
            if self.shuffle:
                permutation = self._mixed_shard_permutations.get(shard)
                if permutation is None:
                    permutation = _StatelessIndexPermutation(
                        len(indices), f"fam-taskmix-v1:{self.seed}:{self.epoch}:"
                        f"shard:{self.dataset.shards[shard]['shard']}")
                    self._mixed_shard_permutations[shard] = permutation
                batch.extend(int(indices[permutation(i)])
                             for i in range(offset, offset + take))
            else:
                batch.extend(int(i) for i in indices[offset:offset + take])
            shards.append(shard)
        batch.extend([-1] * (self.batch_size - len(batch)))
        return tuple(shards), batch

    def _mixed_future_shards(self, start: int, limit: int, seen=()):
        """Bounded metadata-only lookahead; never materialize the whole epoch."""
        result, seen = [], set(seen)
        for step in range(start, min(len(self), start + max(8, limit * 4))):
            for shard, _, _ in self._mixed_stream_pieces(step):
                if shard not in seen:
                    result.append(shard)
                    seen.add(shard)
                if len(result) >= limit:
                    return result
        return result

    def _iter_mixed(self):
        for step in range(self._next_batch_idx, len(self)):
            shards, batch = self._mixed_local_batch(step)
            if self.lookahead_shards > 0 and shards:
                future = self._mixed_future_shards(
                    step + 1, self.lookahead_shards, shards)
                self.dataset.update_prefetch(shards[0], future)
            self._next_batch_idx = step + 1
            yield batch
        self._next_batch_idx = 0

    def _shard_batch_count(self, shard: int) -> int:
        positions = len(self.dataset.position_indices_by_shard[shard])
        if self.drop_last:
            return positions // self.batch_size
        return (positions + self.batch_size - 1) // self.batch_size

    def _total_source_batches(self) -> int:
        return sum(
            self._shard_batch_count(shard)
            for shard in self._ordered_shards()
        )

    def _batches_per_rank(self) -> int:
        total = self._total_source_batches()
        return (total + self.num_replicas - 1) // self.num_replicas

    def _rank_spans(self) -> list[tuple[int, int]]:
        """Global half-open batch spans, wrapping only for equalization."""
        total = self._total_source_batches()
        per_rank = self._batches_per_rank()
        if total <= 0 or per_rank <= 0:
            return []
        start = self.rank * per_rank
        # num_replicas can exceed the source batch count in tiny fixtures.
        start %= total
        end = start + per_rank
        if end <= total:
            return [(start, end)]
        return [(start, total), (0, end - total)]

    def _range_batches(self, start: int, stop: int):
        result = []
        cursor = 0
        for shard in self._ordered_shards():
            count = self._shard_batch_count(shard)
            shard_stop = cursor + count
            overlap_start = max(start, cursor)
            overlap_stop = min(stop, shard_stop)
            if overlap_start < overlap_stop:
                indices = list(self.dataset.position_indices_by_shard[shard])
                if self.shuffle:
                    random.Random(
                        self.seed + self.epoch * 10007 + shard * 7919
                    ).shuffle(indices)
                first = overlap_start - cursor
                last = overlap_stop - cursor
                for batch_index in range(first, last):
                    offset = batch_index * self.batch_size
                    batch = indices[offset:offset + self.batch_size]
                    if batch:
                        result.append((shard, batch))
            cursor = shard_stop
            if cursor >= stop:
                break
        return result

    def _shards(self) -> list[int]:
        if self.global_batch_mixing:
            return self._mixed_future_shards(
                self._next_batch_idx, max(1, self.prefill_shards))
        if self.exact_once:
            result = []
            seen = set()
            for shards, _ in self._exact_batches():
                for shard in shards:
                    if shard not in seen:
                        seen.add(shard)
                        result.append(shard)
            return result
        result = []
        seen = set()
        for shard, _ in self._batches():
            if shard not in seen:
                seen.add(shard)
                result.append(shard)
        return result

    def _batches(self):
        if self.exact_once:
            return self._exact_batches()
        batches = []
        for start, stop in self._rank_spans():
            batches.extend(self._range_batches(start, stop))
        return batches

    def prefill(self) -> None:
        if self.prefill_shards > 0:
            self.dataset.prefill(self._shards()[: self.prefill_shards])

    def __iter__(self):
        if self.global_batch_mixing:
            yield from self._iter_mixed()
            return
        batches = self._batches()
        for batch_index in range(self._next_batch_idx, len(batches)):
            current, batch = batches[batch_index]
            current_shards = current if self.exact_once else (current,)
            if self.lookahead_shards > 0:
                future = []
                seen = set(current_shards)
                for future_current, _ in batches[batch_index + 1:]:
                    future_shards = (
                        future_current if self.exact_once
                        else (future_current,))
                    for future_shard in future_shards:
                        if future_shard not in seen:
                            future.append(future_shard)
                            seen.add(future_shard)
                        if len(future) == self.lookahead_shards:
                            break
                    if len(future) == self.lookahead_shards:
                        break
                if current_shards:
                    self.dataset.update_prefetch(current_shards[0], future)
            self._next_batch_idx = batch_index + 1
            yield batch
        self._next_batch_idx = 0

    def __len__(self) -> int:
        if self.exact_once:
            return self._exact_steps()
        return self._batches_per_rank()

    def state_dict(self) -> Dict[str, Any]:
        if self.exact_once:
            anchors = self._exact_layout()[2]
            global_batch = self.batch_size * self.num_replicas
            state = {
                "epoch": self.epoch,
                "next_batch_idx": self._next_batch_idx,
                "seed": self.seed,
                "batch_size": self.batch_size,
                "num_replicas": self.num_replicas,
                "rank": self.rank,
                "partition_format": (
                    self.MIXED_EXACT_ONCE_PARTITION_FORMAT
                    if self.global_batch_mixing else self.EXACT_ONCE_PARTITION_FORMAT),
                "exact_once": True,
                "source_anchors": anchors,
                "global_batch_size": global_batch,
                "steps_per_rank": self._exact_steps(),
                "padding_samples": (
                    self._exact_steps() * global_batch - anchors),
                "shard_order_epoch": self.epoch,
                "shard_order_sha256": self._shard_order_sha256(self.epoch),
            }
            if self.global_batch_mixing:
                state["shuffle"] = self.shuffle
            return state
        return {
            "epoch": self.epoch,
            "next_batch_idx": self._next_batch_idx,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "partition_format": self.PARTITION_FORMAT,
            "source_batches": self._total_source_batches(),
            "batches_per_rank": self._batches_per_rank(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        expected_format = (
            self.MIXED_EXACT_ONCE_PARTITION_FORMAT if self.global_batch_mixing
            else self.EXACT_ONCE_PARTITION_FORMAT if self.exact_once
            else self.PARTITION_FORMAT)
        if state.get("partition_format") != expected_format:
            raise ValueError(
                "cannot resume FaM sampler from a different partition format")
        for key in ("seed", "batch_size", "num_replicas", "rank"):
            if int(state[key]) != int(getattr(self, key)):
                raise ValueError(f"cannot resume FaM sampler with a different {key}")
        if self.global_batch_mixing:
            if state.get("shuffle") is not self.shuffle:
                raise ValueError("cannot resume mixed FaM sampler with a different shuffle")
            if not 0 <= int(state["next_batch_idx"]) <= len(self):
                raise ValueError("mixed FaM sampler cursor is outside the epoch")
        if self.exact_once:
            anchors = self._exact_layout()[2]
            if int(state.get("source_anchors", -1)) != anchors:
                raise ValueError(
                    "cannot resume FaM sampler after the anchor set changed")
            if int(state.get("global_batch_size", -1)) != (
                    self.batch_size * self.num_replicas):
                raise ValueError(
                    "cannot resume FaM sampler after the global batch changed")
            if int(state.get("steps_per_rank", -1)) != self._exact_steps():
                raise ValueError(
                    "cannot resume FaM sampler after rank partitioning changed")
            order_epoch = int(state.get("shard_order_epoch", -1))
            target_epoch = int(state["epoch"])
            next_batch_idx = int(state["next_batch_idx"])
            if not (
                order_epoch == target_epoch
                or (next_batch_idx == 0 and order_epoch + 1 == target_epoch)
            ):
                raise ValueError(
                    "cannot resume FaM sampler from an ambiguous shard-order epoch")
            if state.get("shard_order_sha256") != self._shard_order_sha256(
                    order_epoch):
                raise ValueError(
                    "cannot resume FaM sampler after the ordered shard/anchor "
                    "schedule changed")
        else:
            if int(state.get("source_batches", -1)) != self._total_source_batches():
                raise ValueError(
                    "cannot resume FaM sampler after the source batch count changed")
            if int(state.get("batches_per_rank", -1)) != self._batches_per_rank():
                raise ValueError(
                    "cannot resume FaM sampler after rank partitioning changed")
        self.epoch = int(state["epoch"])
        self._next_batch_idx = int(state["next_batch_idx"])
        self._exact_layout_cache = None
        self._exact_shard_index_cache.clear()
        self._mixed_block_permutation = None
        self._mixed_shard_permutations.clear()
        self.dataset.set_epoch(self.epoch)


class MultiSourceShardAwareBatchSampler(BatchSampler):
    """Shard-aware sampler that synchronizes source and shard across ranks.

    Every rank follows the same ``(source, shard)`` schedule.  Consequently,
    each distributed step has one action profile and one structural bucket,
    while sample positions inside the shard are split across ranks.  Optional
    source weights resample whole batches; they never mix profiles inside a
    batch.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        chunk_size_range: Tuple[int, int] = (1, 4),
        window_size_range: Tuple[int, int] = (25, 50),
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shard_shuffle: bool = True,
        position_shuffle: bool = True,
        shuffle: Optional[bool] = None,
        drop_last: bool = True,
        seed: int = 0,
        lookahead_shards: int = 0,
        prefill_shards: int = 0,
        split_by: str = "balanced_shard",
        source_schedule_chunk_size: int = 1,
        source_weights: Optional[List[float]] = None,
        steps_per_cycle: Optional[int] = None,
        **kwargs,
    ) -> None:
        if num_replicas is None:
            num_replicas = overwatch.world_size()
        if rank is None:
            rank = overwatch.rank()
        if shuffle is not None:
            shard_shuffle = bool(shuffle)
            position_shuffle = bool(shuffle)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.chunk_size_range = tuple(chunk_size_range)
        self.window_size_range = tuple(window_size_range)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shard_shuffle = bool(shard_shuffle)
        self.position_shuffle = bool(position_shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.lookahead_shards = int(lookahead_shards)
        self.prefill_shards = int(prefill_shards)
        self.split_by = str(split_by)
        self.source_schedule_chunk_size = max(
            1, int(source_schedule_chunk_size)
        )
        if source_weights is None and getattr(
            dataset, "weighted_sampling_enabled", False
        ):
            source_weights = list(dataset.source_weights)
        if source_weights is not None:
            if len(source_weights) != len(dataset.datasets):
                raise ValueError(
                    "source_weights length must equal number of sources: "
                    f"{len(source_weights)} != {len(dataset.datasets)}"
                )
            source_weights = [float(weight) for weight in source_weights]
            if any(weight <= 0 for weight in source_weights):
                raise ValueError("all source_weights must be positive")
        if steps_per_cycle is not None and int(steps_per_cycle) <= 0:
            raise ValueError("steps_per_cycle must be positive")
        self.source_weights = source_weights
        self.steps_per_cycle = (
            int(steps_per_cycle) if steps_per_cycle is not None else None
        )
        self.epoch = 0
        self._cycle = 0
        self._next_batch_idx = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cycle = 0
        self._next_batch_idx = 0

    def _all_shards(self) -> list[tuple[int, int]]:
        keys = []
        for source_idx, ds in enumerate(self.dataset.datasets):
            for shard_idx in range(len(ds.shards)):
                keys.append((source_idx, shard_idx))
        return keys

    def _source_shards(self, source_idx: int) -> list[tuple[int, int]]:
        return [
            (source_idx, shard_idx)
            for shard_idx in range(len(self.dataset.datasets[source_idx].shards))
        ]

    def _shuffled_source_shards(
        self,
        source_idx: int,
        cycle: int,
    ) -> list[tuple[int, int]]:
        keys = self._source_shards(source_idx)
        rng = random.Random(
            self.seed + self.epoch * 10000 + cycle + source_idx * 104729
        )
        if self.shard_shuffle:
            rng.shuffle(keys)
        return keys

    def prefill(self) -> None:
        if self.prefill_shards <= 0 or not hasattr(self.dataset, "prefill"):
            return
        batches = self._build_batches(self._cycle)
        keys = []
        seen = set()
        for key, _ in batches:
            if key in seen:
                continue
            keys.append(key)
            seen.add(key)
            if len(keys) >= self.prefill_shards:
                break
        self.dataset.prefill(keys)

    def _key_batch_count_for_rank(
        self,
        key: tuple[int, int],
        rank: int,
    ) -> int:
        positions_len = len(self.dataset.positions_by_source_shard[key])
        positions_len = len(range(rank, positions_len, self.num_replicas))
        if self.drop_last:
            return positions_len // self.batch_size
        return int(ceil(positions_len / self.batch_size))

    def _key_batch_count(self, key: tuple[int, int]) -> int:
        per_rank = [
            self._key_batch_count_for_rank(key, rank)
            for rank in range(self.num_replicas)
        ]
        return min(per_rank) if per_rank else 0

    def _source_batch_count(self, source_idx: int, cycle: int) -> int:
        count = 0
        for key in self._shuffled_source_shards(source_idx, cycle):
            count += self._key_batch_count(key)
        return count

    def _source_batch_counts(self, cycle: int) -> list[int]:
        counts = []
        for source_idx in range(len(self.dataset.datasets)):
            counts.append(self._source_batch_count(source_idx, cycle))
        return counts

    def _source_targets(self, cycle: int) -> list[int]:
        natural = self._source_batch_counts(cycle)
        if self.source_weights is None:
            return natural
        active = [index for index, count in enumerate(natural) if count > 0]
        if not active:
            return natural
        total = self.steps_per_cycle or sum(natural)
        weight_sum = sum(self.source_weights[index] for index in active)
        exact = {
            index: total * self.source_weights[index] / weight_sum
            for index in active
        }
        targets = [0 for _ in natural]
        for index in active:
            targets[index] = int(exact[index])
        remainder = total - sum(targets)
        order = sorted(
            active,
            key=lambda index: (
                -(exact[index] - targets[index]),
                index,
            ),
        )
        for offset in range(remainder):
            targets[order[offset % len(order)]] += 1
        return targets

    def _source_key_schedule(
        self,
        source_idx: int,
        cycle: int,
        target: int,
    ) -> list[tuple[int, int]]:
        natural: list[tuple[int, int]] = []
        for key in self._shuffled_source_shards(source_idx, cycle):
            natural.extend([key] * self._key_batch_count(key))
        if not natural or target <= 0:
            return []
        start = (
            self.seed + self.epoch * 10000 + cycle * 100003
            + source_idx * 104729
        ) % len(natural)
        rotated = natural[start:] + natural[:start]
        return [rotated[index % len(rotated)] for index in range(target)]

    def _key_schedule(self, cycle: int) -> list[tuple[int, int]]:
        chunks: list[list[tuple[int, int]]] = []
        chunk_size = self.source_schedule_chunk_size
        targets = self._source_targets(cycle)
        for source_idx, target in enumerate(targets):
            source_schedule = self._source_key_schedule(
                source_idx, cycle, target
            )
            for offset in range(0, len(source_schedule), chunk_size):
                chunks.append(source_schedule[offset: offset + chunk_size])
        if self.shard_shuffle:
            rng = random.Random(
                self.seed + self.epoch * 10000 + cycle * 100003 + 17
            )
            rng.shuffle(chunks)
        schedule = []
        for chunk in chunks:
            schedule.extend(chunk)
        return schedule

    def _build_key_batches(
        self,
        key: tuple[int, int],
        cycle: int,
    ) -> list[tuple[tuple[int, int], list[tuple]]]:
        batches: list[tuple[tuple[int, int], list[tuple]]] = []
        source_idx, shard_idx = key
        batch_rng = random.Random(
            self.seed + self.epoch * 10000 + cycle * 100003
            + source_idx * 104729
        )
        positions = list(self.dataset.positions_by_source_shard[key])
        if self.position_shuffle:
            rng = random.Random(
                self.seed + self.epoch * 10000 + cycle
                + source_idx * 104729 + shard_idx * 7919
            )
            rng.shuffle(positions)
        positions = positions[self.rank::self.num_replicas]
        max_batches = self._key_batch_count(key)
        for offset in range(0, len(positions), self.batch_size):
            if len(batches) >= max_batches:
                break
            batch_positions = positions[offset: offset + self.batch_size]
            if len(batch_positions) < self.batch_size and self.drop_last:
                continue
            cs = batch_rng.randint(*self.chunk_size_range)
            ws = batch_rng.randint(*self.window_size_range)
            ws = max((ws // cs) * cs, cs)
            augmentation_rng = random.Random(
                self.seed
                + self.epoch * 10_000
                + cycle * 100_003
                + self.rank * 1_009
                + source_idx * 104_729
                + shard_idx * 7_919
                + offset * 1_000_003
                + 97
            )
            source_dataset = self.dataset.datasets[source_idx]
            if getattr(
                source_dataset,
                "camera_dropout_prob",
                0.0,
            ) > 0.0:
                batch = [
                    (
                        source_idx,
                        shard_idx,
                        ep,
                        t,
                        cs,
                        ws,
                        augmentation_rng.getrandbits(63),
                    )
                    for ep, t in batch_positions
                ]
            else:
                batch = [
                    (source_idx, shard_idx, ep, t, cs, ws)
                    for ep, t in batch_positions
                ]
            batches.append((key, batch))
        return batches

    def _build_batches(self, cycle: int) -> list[tuple[tuple[int, int], list[tuple]]]:
        key_batches = {
            key: self._build_key_batches(key, cycle)
            for key in self._all_shards()
        }
        key_offsets = {key: 0 for key in key_batches}
        batches: list[tuple[tuple[int, int], list[tuple]]] = []
        for key in self._key_schedule(cycle):
            offset = key_offsets[key]
            source_queue = key_batches[key]
            if not source_queue:
                continue
            # Weighted sampling may oversample a source. Cycling is
            # deterministic and identical across ranks; the next cycle
            # reshuffles positions before reuse.
            batches.append(source_queue[offset % len(source_queue)])
            key_offsets[key] = offset + 1
        return batches

    def _future_shards(
        self,
        batches: list[tuple[tuple[int, int], list[tuple]]],
        batch_idx: int,
    ) -> list[tuple[int, int]]:
        future = []
        seen = {batches[batch_idx][0]}
        for idx in range(batch_idx + 1, len(batches)):
            key = batches[idx][0]
            if key in seen:
                continue
            future.append(key)
            seen.add(key)
            if len(future) >= self.lookahead_shards:
                break
        return future

    def __iter__(self):
        cycle = self._cycle
        while True:
            batches = self._build_batches(cycle)
            if not batches:
                return
            start_idx = self._next_batch_idx if cycle == self._cycle else 0
            last_prefetch_key: tuple[int, int] | None = None
            for idx in range(start_idx, len(batches)):
                key, batch = batches[idx]
                if key != last_prefetch_key and hasattr(self.dataset, "update_prefetch"):
                    self.dataset.update_prefetch(key, self._future_shards(batches, idx))
                    last_prefetch_key = key
                self._cycle = cycle
                self._next_batch_idx = idx + 1
                yield batch
            self._next_batch_idx = 0
            cycle += 1

    def __len__(self) -> int:
        return len(self._key_schedule(self._cycle))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cycle": self._cycle,
            "next_batch_idx": self._next_batch_idx,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "num_sources": len(self.dataset.datasets),
            "split_by": self.split_by,
            "source_schedule_chunk_size": self.source_schedule_chunk_size,
            "source_weights": self.source_weights,
            "steps_per_cycle": self.steps_per_cycle,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for key, current in (
            ("batch_size", self.batch_size),
            ("num_replicas", self.num_replicas),
            ("seed", self.seed),
            ("num_sources", len(self.dataset.datasets)),
            ("split_by", self.split_by),
            ("source_schedule_chunk_size", self.source_schedule_chunk_size),
            ("source_weights", self.source_weights),
            ("steps_per_cycle", self.steps_per_cycle),
        ):
            if state.get(key, current) != current:
                raise ValueError(
                    f"Cannot resume multi-source sampler with a different {key}"
                )
        self.epoch = int(state.get("epoch", 0))
        self._cycle = int(state.get("cycle", 0))
        self._next_batch_idx = int(state.get("next_batch_idx", 0))
