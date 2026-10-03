# SPDX-License-Identifier: Apache-2.0
"""Training metric aggregation with optional JSONL, W&B and TensorBoard sinks."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import torch

from zvla.engines.utils import initialize_overwatch

from ..utils.root import METRICS


def _metric_value(value: Any) -> float | torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Metric values must be scalar")
        return value.detach()
    return float(value)


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _mean(values: deque[float | torch.Tensor]) -> float:
    if not values:
        return 0.0
    total = sum(values)
    return _scalar(total / len(values))


class _JsonlSink:
    def __init__(self, run_id: str, run_dir: Path, hparams: Any) -> None:
        self.path = run_dir / f"{run_id}.jsonl"
        metadata = {"run_id": run_id, "hparams": hparams}
        (run_dir / "run-metrics.jsonl").write_text(
            json.dumps(metadata, default=str) + "\n", encoding="utf-8"
        )

    def write(self, step: int, metrics: dict[str, float]) -> None:
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(metrics, allow_nan=False) + "\n")

    def close(self) -> None:
        return None


class _WandbSink:
    def __init__(self, run_id: str, run_dir: Path, hparams: Any, options: dict[str, Any]) -> None:
        import wandb

        self.wandb = wandb
        self.min_step = options["wandb_suppress_through_step"]
        env_tags = os.environ.get("WANDB_TAGS")
        tags = (
            [part.strip() for part in env_tags.split(",") if part.strip()]
            if env_tags is not None
            else list(options["wandb_tags"])
        )
        wandb.init(
            name=run_id,
            dir=str(run_dir),
            config=hparams,
            project=os.environ.get("WANDB_PROJECT", options["wandb_project"] or "zvla"),
            entity=os.environ.get("WANDB_ENTITY", options["wandb_entity"]),
            mode=os.environ.get("WANDB_MODE", options["wandb_mode"]),
            group=os.environ.get("WANDB_GROUP", options["wandb_group"]),
            job_type=os.environ.get("WANDB_JOB_TYPE", options["wandb_job_type"]),
            tags=tags,
            id=options["wandb_run_id"],
            resume=options["wandb_resume"],
            resume_from=options["wandb_resume_from"],
        )

    def write(self, step: int, metrics: dict[str, float]) -> None:
        if self.min_step is None or step > self.min_step:
            self.wandb.log(metrics, step=step)

    def close(self) -> None:
        self.wandb.finish()


class _TensorBoardSink:
    def __init__(self, run_id: str, run_dir: Path, hparams: Any) -> None:
        from torch.utils.tensorboard import SummaryWriter

        target = os.environ.get("TENSORBOARD_LOG_PATH")
        self.writer = SummaryWriter(log_dir=target or str(run_dir / "tensorboard" / run_id))
        self.writer.add_text("hyperparameters", str(hparams))

    def write(self, step: int, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        self.writer.close()


class _DatasetMetrics:
    def __init__(self, window: int) -> None:
        self.values: dict[str, deque[float | torch.Tensor]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def commit(self, **metrics: Any) -> None:
        for name, value in metrics.items():
            if name not in {"global_step", "epoch", "lr"}:
                self.values[name].append(_metric_value(value))


@METRICS.register_module()
class VLAMetric:
    """Aggregate scalar training metrics and write them on the main rank."""

    def __init__(
        self,
        active_trackers: tuple[str, ...],
        run_id: str,
        run_dir: str,
        hparams: dict[str, Any],
        grad_accumulation_steps: int = 1,
        window_size: int = 1,
        resume_step: int | None = None,
        resume_epoch: int | None = None,
        update_step_time: bool = True,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_mode: str = "online",
        wandb_group: str = "vla-train",
        wandb_job_type: str | None = None,
        wandb_tags: tuple[str, ...] = (),
        wandb_run_id: str | None = None,
        wandb_resume: str | None = None,
        wandb_resume_from: str | None = None,
        wandb_suppress_through_step: int | None = None,
    ) -> None:
        if grad_accumulation_steps < 1 or window_size < 1:
            raise ValueError("Metric window sizes must be positive")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.hparams = hparams
        self.global_step = 0 if resume_step is None else resume_step
        self.epoch = 0 if resume_epoch is None else resume_epoch
        self.update_step_time = update_step_time
        self._last_step_time = time.perf_counter()
        self._window = window_size
        self._main_rank = initialize_overwatch(__name__).is_rank_zero()
        self.state: dict[str, deque[float | torch.Tensor]] = {
            "loss_raw": deque(maxlen=grad_accumulation_steps),
            "loss": deque(maxlen=window_size),
            "l1_loss": deque(maxlen=window_size),
            "action_accuracy": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": deque(maxlen=1),
        }
        self.dataset_trackers: dict[str, _DatasetMetrics] = defaultdict(
            lambda: _DatasetMetrics(window_size)
        )
        self.trackers: list[Any] = []
        if self._main_rank:
            options = {
                "wandb_project": wandb_project,
                "wandb_entity": wandb_entity,
                "wandb_mode": wandb_mode,
                "wandb_group": wandb_group,
                "wandb_job_type": wandb_job_type,
                "wandb_tags": wandb_tags,
                "wandb_run_id": wandb_run_id,
                "wandb_resume": wandb_resume,
                "wandb_resume_from": wandb_resume_from,
                "wandb_suppress_through_step": wandb_suppress_through_step,
            }
            for name in active_trackers:
                if name == "jsonl":
                    self.trackers.append(_JsonlSink(run_id, self.run_dir, hparams))
                elif name == "wandb":
                    self.trackers.append(_WandbSink(run_id, self.run_dir, hparams, options))
                elif name == "tensorboard":
                    self.trackers.append(_TensorBoardSink(run_id, self.run_dir, hparams))
                else:
                    raise ValueError(f"Unsupported metric tracker: {name}")

    def get_status(self, loss: Any | None = None) -> str:
        rate = self.state["lr"][-1] if self.state["lr"] else 0.0
        status = f"Epoch {self.epoch:03d} | Step {self.global_step:06d} | LR {rate:.6g}"
        if loss is not None:
            status += f" | Loss {_scalar(loss):.4f}"
        return status

    def commit(
        self,
        *,
        global_step: int | None = None,
        epoch: int | None = None,
        lr: float | None = None,
        **metrics: Any,
    ) -> None:
        if global_step is not None:
            self.global_step = global_step
        if epoch is not None:
            self.epoch = epoch
        if not self._main_rank:
            return
        if lr is not None:
            self.state["lr"].append(float(lr))
        if global_step is not None and self.update_step_time:
            now = time.perf_counter()
            self.state["step_time"].append(now - self._last_step_time)
            self._last_step_time = now
        for name, value in metrics.items():
            scalar = _metric_value(value)
            if name == "loss":
                self.state["loss_raw"].append(scalar)
            if name not in self.state:
                self.state[name] = deque(maxlen=self._window)
            self.state[name].append(scalar)

    def commit_for_dataset(self, dataset_name: str, **metrics: Any) -> None:
        if self._main_rank:
            self.dataset_trackers[dataset_name].commit(**metrics)

    def log(self, global_step: int, metrics: dict[str, float]) -> None:
        if self._main_rank:
            for tracker in self.trackers:
                tracker.write(global_step, metrics)

    def push(self) -> str:
        status = self.get_status(_mean(self.state["loss"]))
        if not self._main_rank:
            return status
        metrics: dict[str, float] = {
            "VLA Train/Step": float(self.global_step),
            "VLA Train/Epoch": float(self.epoch),
            "VLA Train/Loss": _mean(self.state["loss"]),
            "VLA Train/Loss (Raw)": _mean(self.state["loss_raw"]),
            "VLA Train/LR": self.state["lr"][-1] if self.state["lr"] else 0.0,
            "VLA Train/Step Time": _mean(self.state["step_time"]),
        }
        if self.state["l1_loss"]:
            metrics["VLA Train/L1 Loss"] = _mean(self.state["l1_loss"])
        if self.state["action_accuracy"]:
            metrics["VLA Train/Action Token Accuracy"] = _mean(
                self.state["action_accuracy"]
            )
        for dataset, tracker in self.dataset_trackers.items():
            for name, values in tracker.values.items():
                if values:
                    metrics[f"{dataset}/{name}"] = _mean(values)
        fixed = {"loss_raw", "loss", "l1_loss", "action_accuracy", "step_time", "lr"}
        for name, values in self.state.items():
            if name not in fixed and values:
                metrics[f"VLA Train/{name}"] = _mean(values)
                values.clear()
        self.log(self.global_step, metrics)
        return status

    def finalize(self) -> None:
        for tracker in self.trackers:
            tracker.close()
