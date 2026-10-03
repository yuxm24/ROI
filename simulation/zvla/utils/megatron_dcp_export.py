"""Export selected model modules from a Megatron DCP checkpoint.

The exporter intentionally runs without an initialized process group. DCP
reconstructs each selected full tensor from the one saved writer replica on
CPU, avoiding an uneven-DTensor all-gather and its temporary GPU memory peak.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def export_model_artifact(
    checkpoint: Path,
    output: Path,
    prefixes: Iterable[str],
    *,
    global_step: int,
    epoch: int,
) -> dict[str, object]:
    """Reconstruct selected ``model.<prefix>.*`` tensors and save safetensors."""

    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    requested = tuple(dict.fromkeys(prefixes))
    if not requested or any(not value or "." in value for value in requested):
        raise ValueError(
            "artifact prefixes must be unique, non-empty path components")
    if not (checkpoint / ".complete").is_file():
        raise RuntimeError(f"DCP checkpoint is not atomically complete: {checkpoint}")
    if output.exists():
        raise FileExistsError(f"artifact output already exists: {output}")

    reader = FileSystemReader(str(checkpoint))
    metadata = reader.read_metadata()
    selected: dict[str, torch.Tensor] = {}
    counts = {prefix: 0 for prefix in requested}
    for fqn, entry in metadata.state_dict_metadata.items():
        if not fqn.startswith("model."):
            continue
        key = fqn[len("model."):]
        prefix = next((value for value in requested
                       if key.startswith(f"{value}.")), None)
        if prefix is None:
            continue
        if not isinstance(entry, TensorStorageMetadata):
            raise TypeError(f"artifact entry is not a tensor: {fqn}")
        if key in selected:
            raise RuntimeError(f"duplicate DCP artifact key: {key}")
        selected[key] = torch.empty(
            tuple(entry.size), dtype=entry.properties.dtype, device="cpu")
        counts[prefix] += 1
    missing = [prefix for prefix, count in counts.items() if count == 0]
    if missing:
        raise KeyError(f"artifact modules absent from DCP metadata: {missing}")

    dcp.load({"model": selected}, storage_reader=reader)
    selected = {
        key: value.detach().contiguous()
        for key, value in sorted(selected.items())
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_metadata = {
        "artifact_kind": "model_handoff",
        "source_format": "megatron_fsdp_dcp_v3",
        "global_step": str(int(global_step)),
        "epoch": str(int(epoch)),
        "module_prefixes": json.dumps(requested, separators=(",", ":")),
    }
    save_file(selected, output, metadata=artifact_metadata)

    with safe_open(output, framework="pt", device="cpu") as handle:
        actual_keys = list(handle.keys())
        actual_metadata = handle.metadata()
    if actual_keys != sorted(selected):
        raise RuntimeError("saved artifact key inventory differs from DCP selection")
    if actual_metadata != artifact_metadata:
        raise RuntimeError("saved artifact metadata failed round-trip validation")
    return {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "tensor_count": len(selected),
        "element_count": sum(value.numel() for value in selected.values()),
        "byte_count": output.stat().st_size,
        "module_counts": counts,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module-prefix", action="append", required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("artifact exporter must run with CUDA disabled")
    result = export_model_artifact(
        args.checkpoint,
        args.output,
        args.module_prefix,
        global_step=args.global_step,
        epoch=args.epoch,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
