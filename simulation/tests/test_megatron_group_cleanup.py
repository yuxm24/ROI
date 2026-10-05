"""NCCL cleanup handles mesh groups that share a backend, without a GPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from zvla.strategies.megatron_fsdp_strategy import MegatronFSDPStrategy


class RegisteredBackend:
    def __init__(self, name, events, error=None):
        self.name = name
        self.events = events
        self.error = error
        self.registered = True

    def deregister_mem_pool(self, pool):
        self.events.append((self.name, pool))
        if self.error is not None:
            raise self.error
        if not self.registered:
            raise RuntimeError("Trying to unregister not previously registered pool")
        self.registered = False


def cleanup_strategy(monkeypatch, *, separate_alias=False, error=None):
    events, destroyed = [], []
    world_backend = RegisteredBackend("world", events)
    ag_backend = RegisteredBackend("ag", events, error)

    def group(backend):
        return SimpleNamespace(_get_backend=lambda device: backend)

    world, ag = group(world_backend), group(ag_backend)
    alias = group(world_backend) if separate_alias else world
    status = SimpleNamespace()
    pool = SimpleNamespace(snapshot=lambda: [{"address": 1}])
    mesh = SimpleNamespace(
        get_group=lambda name: {"shard": world, "replicate": alias}[name]
    )
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=world))
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "barrier", lambda *, group: None)
    monkeypatch.setattr(dist, "destroy_process_group", destroyed.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    strategy = MegatronFSDPStrategy(device_id=0, device_mesh=mesh)
    strategy._registered_pool = pool
    strategy._registered_groups = (world, ag, alias)
    strategy._buffers_registered = True
    strategy._ag_group = ag
    strategy._status_group = status
    return strategy, events, destroyed, pool, world_backend, ag_backend, world, ag


@pytest.mark.parametrize("separate_alias", [False, True])
def test_cleanup_deregisters_each_backend_once(monkeypatch, separate_alias):
    strategy, events, destroyed, pool, world_backend, ag_backend, world, ag = (
        cleanup_strategy(monkeypatch, separate_alias=separate_alias)
    )

    strategy.destroy_process_groups()

    # WORLD is shared by shard/replicate in the single-rank mesh. Distinct
    # ProcessGroup objects can also share a backend; both must be deduplicated.
    assert events == [("world", pool), ("ag", pool)]
    assert not world_backend.registered
    assert not ag_backend.registered
    assert any(group is ag for group in destroyed)
    assert all(group is not world for group in destroyed)
    assert strategy._registered_pool is None
    assert strategy._registered_groups == ()
    assert not strategy._buffers_registered


def test_cleanup_propagates_actual_deregistration_error(monkeypatch):
    failure = RuntimeError("synthetic NCCL deregistration failure")
    strategy, events, destroyed, pool, world_backend, ag_backend, _, _ = (
        cleanup_strategy(monkeypatch, error=failure)
    )

    with pytest.raises(RuntimeError, match="synthetic NCCL deregistration failure") as raised:
        strategy.destroy_process_groups()

    assert raised.value is failure
    assert events == [("world", pool), ("ag", pool)]
    assert not world_backend.registered
    assert ag_backend.registered
    assert destroyed == []
    assert strategy._registered_pool is pool
