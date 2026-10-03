"""Contract checks for the Apache-licensed training logger and metric writer."""

from __future__ import annotations

import json

import torch
from zvla.engines.metrics.vla_metric import VLAMetric
from zvla.engines.utils.overwatch import initialize_overwatch


def test_jsonl_metrics_do_not_replay_conditional_diagnostics(tmp_path):
    metric = VLAMetric(("jsonl",), "example", str(tmp_path), {"batch": 2})
    metric.commit(global_step=1, epoch=0, lr=1e-5, loss=torch.tensor(2.0), grad_norm=3.0)
    assert "Step 000001" in metric.push()
    metric.commit(global_step=2, epoch=0, lr=1e-5, loss=torch.tensor(1.0))
    metric.push()
    metric.finalize()

    records = [json.loads(line) for line in (tmp_path / "example.jsonl").read_text().splitlines()]
    assert records[0]["VLA Train/Loss"] == 2.0
    assert records[0]["VLA Train/grad_norm"] == 3.0
    assert records[1]["VLA Train/Loss"] == 1.0
    assert "VLA Train/grad_norm" not in records[1]


def test_rank_filter_uses_torchrun_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    logger = initialize_overwatch("metric-test")
    called = []

    @logger.rank_zero_only
    def record():
        called.append(True)

    record()
    assert called == []
    assert logger.rank() == 1
    assert logger.world_size() == 2
