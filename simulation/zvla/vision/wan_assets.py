"""Pinned Wan source/model validation and UMT5 task embedding materialization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from .wan_imports import import_pinned_wan_module

WAN_GIT_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
WAN_MODEL_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
NODE_LOCAL_ASSET_FORMAT = "fam-wam-v5.2-node-local-assets-v1"
NODE_LOCAL_ASSET_ROOT = Path("/tmp/zvla-v5.2-assets")


def _configured_node_asset_root() -> Path:
    """Use the same explicit scratch root as the node-local staging producer.

    The legacy default is unchanged. An override must still be a dedicated
    directory under /tmp or /dev/shm, never a mount root or persistent CFS.
    """
    configured = Path(os.environ.get(
        "V52_NODE_ASSET_ROOT", str(NODE_LOCAL_ASSET_ROOT)))
    if not configured.is_absolute():
        raise ValueError(
            "node asset cache must be a specific path under /tmp or /dev/shm")
    root = configured.resolve()
    scratch_mounts = (Path("/tmp"), Path("/dev/shm"))
    if (root in scratch_mounts
            or not any(mount in root.parents for mount in scratch_mounts)):
        raise ValueError(
            "node asset cache must be a specific path under /tmp or /dev/shm")
    return root


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_relocated_asset_bundle(
    receipt_path: str | Path,
    source_repo: str | Path,
    model_root: str | Path,
    *,
    kind: str,
) -> Mapping[str, Any]:
    """Validate an explicitly enabled node-local mirror by its small manifest."""
    if os.environ.get("ZVLA_ALLOW_RELOCATED_MODEL_ASSETS") != "1":
        raise ValueError("relocated model assets were not explicitly enabled")
    cache_root = _configured_node_asset_root()
    source_repo = Path(source_repo).resolve()
    model_root = Path(model_root).resolve()
    if source_repo.parent != model_root.parent:
        raise ValueError("relocated source and model must share one bundle root")
    bundle_root = source_repo.parent
    if bundle_root == cache_root or cache_root not in bundle_root.parents:
        raise ValueError(
            f"relocated model bundle must be below {cache_root}")
    manifest_path = bundle_root / "manifest.json"
    complete_path = bundle_root / ".complete"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError(f"node-local asset bundle is incomplete: {bundle_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != NODE_LOCAL_ASSET_FORMAT
            or manifest.get("kind") != kind):
        raise ValueError(f"node-local asset manifest differs: {manifest_path}")
    receipt_path = Path(receipt_path).resolve()
    if (Path(str(manifest.get("receipt", ""))).resolve() != receipt_path
            or manifest.get("receipt_sha256") != sha256_file(receipt_path)):
        raise ValueError("node-local bundle receipt identity differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("node-local asset manifest has no files")
    for relative, size in files.items():
        path = model_root / str(relative)
        if not path.is_file() or path.stat().st_size != int(size):
            raise FileNotFoundError(f"node-local asset size differs: {path}")
    return manifest


def validate_wan_assets(
    receipt_path: str | Path,
    source_repo: str | Path,
    model_root: str | Path,
) -> str:
    """Verify every persistent Wan asset consumed by preprocessing."""
    receipt_path = Path(receipt_path).resolve()
    source_repo = Path(source_repo).resolve()
    model_root = Path(model_root).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("format") != "fam-wam-v5-cfs-asset-receipt-v1":
        raise ValueError(f"unsupported asset receipt: {receipt_path}")
    source = receipt.get("source", {}).get("wan", {})
    relocated = (Path(str(source.get("path", ""))).resolve() != source_repo
                 or Path(str(receipt.get("wan", {}).get("root", ""))).resolve()
                 != model_root)
    if relocated:
        validate_relocated_asset_bundle(
            receipt_path, source_repo, model_root, kind="wan")
    if source.get("revision") != WAN_GIT_REVISION:
        raise ValueError("Wan source revision differs from pinned revision")
    head = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True).stdout.strip()
    if head != WAN_GIT_REVISION:
        raise ValueError(f"Wan checkout HEAD differs: {head}")
    model = receipt.get("wan", {})
    if model.get("revision") != WAN_MODEL_REVISION:
        raise ValueError("Wan model revision differs from pinned revision")
    required = (
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
    )
    recorded = model.get("files", {})
    for relative in required:
        expected = recorded.get(relative)
        if not isinstance(expected, Mapping):
            raise ValueError(f"asset receipt misses {relative}")
        path = model_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(expected.get("bytes", -1)):
            raise ValueError(f"asset size differs for {relative}")
        if (os.environ.get("ZVLA_FAST_ASSET_VALIDATION") != "1"
                and sha256_file(path) != expected.get("sha256")):
            raise ValueError(f"asset hash differs for {relative}")
    return sha256_file(receipt_path)


def _load_text_encoder(
    model_root: Path,
    source_repo: Path,
    device: torch.device,
    *,
    initialize_on_cpu: bool = False,
):
    T5EncoderModel = import_pinned_wan_module(source_repo,
                                              "modules.t5").T5EncoderModel

    tokenizer = model_root / "google" / "umt5-xxl"
    checkpoint = model_root / "models_t5_umt5-xxl-enc-bf16.pth"
    encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device='cpu' if initialize_on_cpu else str(device),
        checkpoint_path=str(checkpoint),
        tokenizer_path=str(tokenizer),
    )
    if initialize_on_cpu:
        # The pinned constructor first allocates the full FP32 encoder before
        # casting to BF16. Avoid that temporary GPU peak on 24-GiB cards; the
        # same checkpoint and final BF16 weights still execute on the GPU.
        encoder.model.to(device=device)
        encoder.device = str(device)
    return encoder


def prepare_wan_text_embeddings(
    output_root: str | Path,
    tasks: Sequence[str],
    *,
    model_root: str | Path,
    source_repo: str | Path,
    device: torch.device,
    asset_receipt_sha256: str,
    expected_sequence_length: Optional[int] = None,
) -> None:
    """Atomically materialize one pinned UMT5 embedding per instruction."""
    output_root = Path(output_root)
    model_root = Path(model_root).resolve()
    source_repo = Path(source_repo).resolve()
    tasks = [str(value) for value in tasks]
    if not tasks or any(not value.strip() for value in tasks):
        raise ValueError("task inventory must contain non-empty instructions")
    meta_path = output_root / "text_embeddings_meta.json"
    array_path = output_root / "text_embeddings.npy"
    tasks_path = output_root / "tasks.json"
    expected_header = {
        "format":
        "fam-wam-umt5-text-embeddings-v1",
        "tasks_sha256":
        hashlib.sha256(
            json.dumps(
                tasks,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
        "asset_receipt_sha256":
        str(asset_receipt_sha256),
        "dtype":
        "float16",
        "width":
        4096,
    }
    if meta_path.is_file() and array_path.is_file() and tasks_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if {
                key: metadata.get(key)
                for key in expected_header
        } != expected_header:
            raise ValueError("existing text embedding contract differs")
        if json.loads(tasks_path.read_text(encoding="utf-8")) != tasks:
            raise ValueError("existing task inventory differs")
        values = np.load(array_path, mmap_mode="r")
        expected_shape = (len(tasks), int(metadata["sequence_length"]), 4096)
        if values.dtype != np.float16 or tuple(values.shape) != expected_shape:
            raise ValueError("existing text embedding tensor differs")
        if array_path.stat().st_size != int(metadata["bytes"]):
            raise ValueError("existing text embedding byte size differs")
        if sha256_file(array_path) != metadata["sha256"]:
            raise ValueError("existing text embedding hash differs")
        if (expected_sequence_length is not None
                and expected_shape[1] != int(expected_sequence_length)):
            raise ValueError("existing text sequence length differs")
        return

    encoder = _load_text_encoder(model_root, source_repo, device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(tasks), 8):
            batch = encoder(tasks[start:start + 8], str(device))
            outputs.extend(value.float().cpu().numpy().astype(np.float16)
                           for value in batch)
    natural_maximum = max(len(value) for value in outputs)
    if not 0 < natural_maximum <= 512:
        raise ValueError(f"unexpected UMT5 sequence length {natural_maximum}")
    maximum = (natural_maximum if expected_sequence_length is None else
               int(expected_sequence_length))
    if natural_maximum > maximum:
        raise ValueError(
            f"UMT5 text would be truncated: natural={natural_maximum}, stored={maximum}"
        )
    embeddings = np.zeros((len(outputs), maximum, 4096), np.float16)
    for index, value in enumerate(outputs):
        if value.ndim != 2 or value.shape[1] != 4096:
            raise ValueError(f"unexpected UMT5 output {value.shape}")
        embeddings[index, :len(value)] = value
    temporary = array_path.with_name(f".{array_path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, embeddings, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, array_path)
    atomic_json(tasks_path, tasks)
    metadata = dict(expected_header)
    metadata.update({
        "tasks": len(tasks),
        "sequence_length": maximum,
        "natural_max_sequence_length": natural_maximum,
        "truncated_tasks": 0,
        "bytes": array_path.stat().st_size,
        "sha256": sha256_file(array_path),
    })
    atomic_json(meta_path, metadata)
    del encoder, outputs, embeddings
    torch.cuda.empty_cache()


__all__ = [
    "WAN_GIT_REVISION",
    "WAN_MODEL_REVISION",
    "NODE_LOCAL_ASSET_FORMAT",
    "atomic_json",
    "prepare_wan_text_embeddings",
    "sha256_file",
    "validate_relocated_asset_bundle",
    "validate_wan_assets",
]
