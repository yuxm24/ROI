"""Run with python -m zvla.posttrain."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .recipes import BENCHMARKS, recipe


def assets(parser):
    for name in ("wan-repo", "wan-model", "vggt-repo", "vggt-model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")


def main():
    parser = argparse.ArgumentParser(description="Zerith VA benchmark post-training")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="encode public benchmark demonstrations")
    p.add_argument("--bench", choices=BENCHMARKS, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume a matching preparation cache after checking input, asset and output content",
    )
    assets(p)
    p = commands.add_parser("train", help="launch torchrun, without gradient accumulation")
    p.add_argument("--bench", choices=BENCHMARKS, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpus", type=int, default=1, help="GPUs per node")
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--master-addr", default="127.0.0.1")
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--batch-size", type=int, default=10, help="per GPU")
    p.add_argument("--steps", type=int)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume training only from a run directory you created or whose source you trust",
    )
    p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("export", help="export a trained model-step safetensors file")
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("serve", help="serve the exported policy")
    p.add_argument("--artifact", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    assets(p)
    p = commands.add_parser("eval", help="run the benchmark simulator client")
    p.add_argument("--bench", choices=BENCHMARKS, required=True)
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--output", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--episodes", type=int, help="episodes per task for LIBERO/Plus/MetaWorld")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--benchmark-root", help="upstream benchmark source checkout")
    p.add_argument("--task", help="task name for RoboTwin/RoboDojo")
    args = parser.parse_args()
    if args.command == "eval" and args.episodes is None:
        args.episodes = {"libero": 10, "libero-plus": 1, "metaworld": 50}.get(args.bench)
    if args.command == "prepare":
        from .prepare import prepare

        prepare(args)
    elif args.command == "train":
        cfg = recipe(
            args.bench,
            str(Path(args.data).resolve()),
            str(Path(args.checkpoint).resolve()),
            args.batch_size,
        )
        cfg["training"]["learning_rate"] = args.learning_rate
        cfg["data"]["train"]["num_workers"] = args.workers
        if args.steps is not None:
            cfg["training"]["max_steps"] = args.steps
            cfg["training"]["checkpoint"]["export_model_steps"] = (args.steps,)
            if args.bench == "metaworld":
                cfg["data"]["train"]["dataset"]["sampler"]["steps"] = args.steps
        if args.dry_run:
            print(
                json.dumps(
                    dict(config=cfg, global_batch=args.nodes * args.gpus * args.batch_size),
                    indent=2,
                )
            )
            return
        from mmengine import Config

        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        path = output / "launch-config.json"
        Config(cfg).dump(str(path))
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(args.gpus)]
        cmd += (
            ["--standalone"]
            if args.nodes == 1
            else [
                "--nnodes",
                str(args.nodes),
                "--node-rank",
                str(args.node_rank),
                "--master-addr",
                args.master_addr,
                "--master-port",
                str(args.master_port),
            ]
        )
        cmd += [
            "--module",
            "zvla.posttrain.train",
            "--config",
            str(path),
            "--work-dir",
            str(output),
        ]
        if args.resume:
            cmd.append("--resume")
        subprocess.run(cmd, check=True)
    elif args.command == "export":
        from .export import export

        export(args)
    elif args.command == "serve":
        from .serve import serve

        serve(args)
    else:
        from .evaluate import evaluate

        evaluate(args)


if __name__ == "__main__":
    main()
