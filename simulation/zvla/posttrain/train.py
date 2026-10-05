"""Post-train Zerith VA on one benchmark with Megatron-FSDP.

Launch with torchrun on any number of GPUs; the shard axis defaults to the
GPUs of one node and the replicate axis covers the remaining nodes::

    torchrun --nproc-per-node 8 --module zvla.posttrain.train --config configs/posttrain/libero.py \\
        --work-dir runs/libero --cfg-options model.pretrained_checkpoint=/path/zvla-base.safetensors
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import uuid
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmengine import Config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--cfg-options", nargs="+", action="append")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue from the newest complete checkpoint in --work-dir; "
            "use only a training directory you created or whose source you trust"
        ),
    )
    args = parser.parse_args(argv)
    if args.cfg_options is not None:
        from mmengine import DictAction

        groups = args.cfg_options
        args.cfg_options = None
        action = DictAction(option_strings=["--cfg-options"], dest="cfg_options", nargs="+")
        for values in groups:
            action(parser, args, values, "--cfg-options")
    return args


def _latest_checkpoint(work_dir: str):
    from zvla.utils.checkpoint_io import torch_load_checkpoint

    root = os.path.join(work_dir, "checkpoints")
    best = None
    if os.path.isdir(root):
        for entry in os.scandir(root):
            if (
                entry.name.startswith("step-")
                and entry.is_dir()
                and os.path.isfile(os.path.join(entry.path, ".complete"))
            ):
                meta = torch_load_checkpoint(
                    os.path.join(entry.path, "metadata.pt"), map_location="cpu"
                )
                key = (int(meta["global_step"]), entry.path)
                best = key if best is None or key > best else best
    return None if best is None else best[1]


def _run_state(work_dir: str, cfg: Config, resume_from) -> dict:
    """run_id / W&B id / seed survive restarts of the same work dir."""
    path = os.path.join(work_dir, "run_state.json")
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    if resume_from:
        from zvla.utils.checkpoint_io import torch_load_checkpoint

        meta = torch_load_checkpoint(os.path.join(resume_from, "metadata.pt"), map_location="cpu")
    else:
        meta = {}
    stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefix = cfg.training.get("run_name_prefix", "zvla-posttrain")
    state = {
        "run_id": meta.get("run_id", f"{prefix}_{stamp}"),
        "wandb_run_id": meta.get("wandb_run_id", uuid.uuid4().hex),
        "run_seed": int(meta.get("run_seed", cfg.training.get("run_seed", 0))),
    }
    with open(path, "w") as handle:
        json.dump(state, handle, indent=2)
    return state


def _broadcast(value):
    import torch.distributed as dist

    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def main(argv=None):
    args = parse_args(argv)

    import torch
    import torch.distributed as dist
    from mmengine import Config

    from zvla.datasets.utils.dataloader_builder import build_train_dataloader
    from zvla.engines import build_metric_from_cfg, build_vla_from_cfg, initialize_overwatch
    from zvla.engines.utils.torch_utils import set_rank_random_seed, set_seed_everywhere
    from zvla.strategies import build_strategy
    from zvla.trainer import VLATrainer, build_optimizer_and_scheduler
    from zvla.utils.cuda_runtime import (
        build_fsdp2_device_mesh,
        configure_cuda_runtime,
        configure_distributed_environment,
    )

    for module_name in ("zvla.models", "zvla.datasets", "zvla.collators"):
        import_module(module_name)
    overwatch = initialize_overwatch(__name__)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    os.makedirs(args.work_dir, exist_ok=True)
    runtime_cfg = cfg.get("runtime", {}) or {}
    configure_distributed_environment(runtime_cfg)

    device_id = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(seconds=int(os.environ.get("ZVLA_TRAIN_TIMEOUT_SEC", "7200"))),
    )
    rank = dist.get_rank()
    resume_from = _latest_checkpoint(args.work_dir) if (args.resume and rank == 0) else None
    resume_from = _broadcast(resume_from)
    configure_cuda_runtime(runtime_cfg, device_id=device_id)

    distributed_cfg = cfg.distributed
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", dist.get_world_size()))
    mesh_cfg = dict(distributed_cfg.get("device_mesh") or {})
    mesh_cfg.setdefault("shape", (-1, local_world, 1))
    mesh_cfg.setdefault("dim_names", ("replicate", "shard", "tp"))
    strategy = build_strategy(
        strategy_type=distributed_cfg.strategy,
        device_id=device_id,
        mixed_precision=distributed_cfg.get("mixed_precision", "bf16"),
        sharding_strategy=distributed_cfg.get("sharding_strategy", "hybrid_shard"),
        sync_module_states=distributed_cfg.get("sync_module_states", True),
        resume_from_checkpoint=resume_from is not None,
        device_mesh=build_fsdp2_device_mesh(mesh_cfg),
        megatron_prefetch_elements=distributed_cfg.get("megatron_prefetch_elements", 327_313_408),
        megatron_ubr_registration_mode=distributed_cfg.get(
            "megatron_ubr_registration_mode", "local"
        ),
    )

    run_state = _run_state(args.work_dir, cfg, resume_from) if rank == 0 else None
    run_state = _broadcast(run_state)
    set_seed_everywhere(int(run_state["run_seed"]))
    training_cfg = dict(cfg.training)
    training_cfg["run_seed"] = int(run_state["run_seed"])
    training_cfg["grad_accumulation_steps"] = 1

    metric = trainer = None
    try:
        model = strategy.setup(build_vla_from_cfg(cfg.model))
        root_model = strategy.unwrap_model(model)
        set_rank_random_seed(int(run_state["run_seed"]), strategy.rank())
        training_cfg["initialization"] = getattr(root_model, "initialization_report", None)

        data_cfg = cfg.data.train
        train_dataloader = build_train_dataloader(
            dataset_cfg=data_cfg.dataset,
            collator_cfg=data_cfg.collator,
            per_device_batch_size=data_cfg.per_device_batch_size,
            num_workers=data_cfg.get("num_workers", 8),
            prefetch_factor=data_cfg.get("prefetch_factor", 2),
            pin_memory=data_cfg.get("pin_memory", True),
            strategy=strategy,
            seed=int(run_state["run_seed"]),
        )
        training_cfg["per_device_batch_size"] = data_cfg.per_device_batch_size
        training_cfg["steps_per_epoch"] = len(train_dataloader)
        optimizer, scheduler = build_optimizer_and_scheduler(
            model, training_cfg, int(training_cfg["max_steps"]), strategy=strategy
        )

        metric_cfg = dict(training_cfg.get("metric", {}))
        metric_cfg.update(
            hparams=cfg.to_dict(),
            run_id=run_state["run_id"],
            wandb_run_id=run_state["wandb_run_id"],
            wandb_resume="must" if resume_from else "allow",
            run_dir=args.work_dir,
        )
        metric = build_metric_from_cfg(metric_cfg)
        if strategy.is_rank_zero():
            cfg.dump(os.path.join(args.work_dir, "config.py"))
            with open(os.path.join(args.work_dir, "config.json"), "w") as handle:
                json.dump(cfg.to_dict(), handle, indent=2, default=str)

        trainer = VLATrainer(
            strategy=strategy,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_dataloader=train_dataloader,
            metric=metric,
            training_cfg=training_cfg,
        )
        if resume_from is not None:
            trainer.resume_from(resume_from)
        checkpoint = trainer.fit()
        if strategy.is_rank_zero():
            overwatch.info(f"Training complete. Latest checkpoint: {checkpoint}")
    finally:
        strategy.finalize()
        if metric is not None:
            metric.finalize()
        trainer = metric = None
        strategy.release_runtime()
        strategy.destroy_process_groups()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
