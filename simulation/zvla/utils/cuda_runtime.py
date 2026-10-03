"""Fail-closed CUDA runtime tuning and FSDP2 device-mesh construction."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist


def _parse_cpu_list(value: str) -> list[int]:
    result = []
    for part in str(value).strip().split(","):
        if not part:
            continue
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            if start > stop:
                raise ValueError(f"invalid CPU range {part!r}")
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    if not result or len(result) != len(set(result)):
        raise ValueError(f"invalid CPU list {value!r}")
    return result


def _partition_core_groups(
    groups: list[list[int]],
    partitions: int,
    index: int,
) -> list[int]:
    if partitions <= 0 or not 0 <= index < partitions:
        raise ValueError("invalid CPU affinity partition")
    start = len(groups) * index // partitions
    stop = len(groups) * (index + 1) // partitions
    if start == stop:
        raise ValueError("fewer physical CPU cores than local GPU ranks")
    return sorted(cpu for group in groups[start:stop] for cpu in group)


def _allowed_physical_core_groups() -> list[list[int]]:
    """Return scheduler-allowed SMT siblings grouped by physical core."""
    allowed = sorted(os.sched_getaffinity(0))
    core_map: dict[tuple[int, int], list[int]] = {}
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
        core_map.setdefault((package, core), []).append(cpu)
    if not core_map:
        raise RuntimeError("scheduler CPU affinity contains no physical cores")
    return [sorted(core_map[key]) for key in sorted(core_map)]


def configure_cpu_worker_affinity(
    config: Mapping[str, Any] | None,
    *,
    worker_id: int,
    workers: int,
) -> dict[str, Any]:
    """Partition all allowed physical cores across CPU-only worker ranks."""
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"enabled": False}
    groups = _allowed_physical_core_groups()
    cpus = _partition_core_groups(groups, int(workers), int(worker_id))
    os.sched_setaffinity(0, cpus)
    actual = sorted(os.sched_getaffinity(0))
    if actual != cpus:
        raise RuntimeError(
            f"CPU worker affinity did not apply exactly: {actual} != {cpus}")
    return {
        "enabled": True,
        "worker_id": int(worker_id),
        "workers": int(workers),
        "logical_cpus": cpus,
        "physical_cores": len(cpus) if all(
            len(group) == 1 for group in groups) else len({
                (int((Path(f"/sys/devices/system/cpu/cpu{cpu}/topology") /
                      "physical_package_id").read_text().strip()),
                 int((Path(f"/sys/devices/system/cpu/cpu{cpu}/topology") /
                      "core_id").read_text().strip()))
                for cpu in cpus
            }),
    }


def _visible_gpu_inventory() -> list[dict[str, str]]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    physical = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 3:
            raise ValueError(f"unexpected nvidia-smi GPU row: {line!r}")
        physical.append({
            "index": values[0],
            "uuid": values[1],
            "pci_bus_id": values[2],
        })
    if not physical:
        raise RuntimeError("nvidia-smi returned no GPUs")
    selectors = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not selectors:
        return physical
    visible = []
    for selector in selectors.split(","):
        selector = selector.strip()
        matches = [
            gpu for gpu in physical
            if gpu["index"] == selector
            or gpu["uuid"] == selector
            or gpu["uuid"].startswith(selector)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"cannot uniquely map CUDA_VISIBLE_DEVICES entry {selector!r}")
        visible.append(matches[0])
    return visible


def _pci_sysfs_path(bus_id: str) -> Path:
    domain, bus, device = str(bus_id).lower().split(":")
    normalized = f"{int(domain, 16):04x}:{bus}:{device}"
    return Path("/sys/bus/pci/devices") / normalized


def configure_local_cpu_affinity(
    config: Mapping[str, Any] | None,
    *,
    device_id: int,
) -> dict[str, Any]:
    """Bind one local GPU rank and its workers to GPU-local CPU cores."""
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"enabled": False}
    inventory = _visible_gpu_inventory()
    if not 0 <= int(device_id) < len(inventory):
        raise ValueError(
            f"local CUDA rank {device_id} is outside {len(inventory)} GPUs")
    gpu = inventory[int(device_id)]
    sysfs = _pci_sysfs_path(gpu["pci_bus_id"])
    numa_path = sysfs / "numa_node"
    if not numa_path.is_file():
        raise FileNotFoundError(f"GPU NUMA metadata is missing: {numa_path}")
    numa_node = int(numa_path.read_text(encoding="utf-8").strip())
    if numa_node < 0:
        raise RuntimeError(f"GPU {device_id} has no NUMA node")
    node_cpu_path = Path(
        f"/sys/devices/system/node/node{numa_node}/cpulist")
    node_cpus = _parse_cpu_list(node_cpu_path.read_text(encoding="utf-8"))
    allowed = set(os.sched_getaffinity(0))
    node_cpus = [cpu for cpu in node_cpus if cpu in allowed]
    if not node_cpus:
        raise RuntimeError(
            f"scheduler CPU affinity excludes GPU NUMA node {numa_node}")

    core_map: dict[tuple[int, int], list[int]] = {}
    for cpu in node_cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
        core_map.setdefault((package, core), []).append(cpu)
    core_groups = [
        sorted(core_map[key]) for key in sorted(core_map)
    ]

    local_on_numa = []
    for local_index, candidate in enumerate(inventory):
        candidate_path = _pci_sysfs_path(candidate["pci_bus_id"]) / "numa_node"
        candidate_numa = int(candidate_path.read_text().strip())
        if candidate_numa == numa_node:
            local_on_numa.append(local_index)
    numa_ordinal = local_on_numa.index(int(device_id))
    cpus = _partition_core_groups(
        core_groups, len(local_on_numa), numa_ordinal)
    os.sched_setaffinity(0, cpus)
    actual = sorted(os.sched_getaffinity(0))
    if actual != cpus:
        raise RuntimeError(
            f"CPU affinity did not apply exactly: {actual} != {cpus}")
    return {
        "enabled": True,
        "local_rank": int(device_id),
        "gpu_uuid": gpu["uuid"],
        "gpu_pci_bus_id": gpu["pci_bus_id"],
        "numa_node": numa_node,
        "numa_gpu_ordinal": numa_ordinal,
        "numa_gpu_count": len(local_on_numa),
        "logical_cpus": cpus,
        "physical_cores": len({
            (int((Path(f"/sys/devices/system/cpu/cpu{cpu}/topology") /
                 "physical_package_id").read_text().strip()),
             int((Path(f"/sys/devices/system/cpu/cpu{cpu}/topology") /
                 "core_id").read_text().strip()))
            for cpu in cpus
        }),
    }


def configure_distributed_environment(config: Mapping[str, Any]) -> None:
    """Set conservative NCCL/allocator defaults before CUDA initialization."""
    if not config or not bool(config.get("enabled", True)):
        return
    defaults = config.get("environment", {}) or {}
    if not isinstance(defaults, Mapping):
        raise TypeError("runtime.environment must be a mapping")
    for key, value in defaults.items():
        key = str(key)
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"unsafe runtime environment key: {key!r}")
        os.environ.setdefault(key, str(value))


def configure_cuda_runtime(
    config: Mapping[str, Any],
    *,
    device_id: int,
) -> dict[str, Any]:
    """Apply explicit Hopper-safe math and SDPA backend preferences."""
    if not config or not bool(config.get("enabled", True)):
        return {"enabled": False}
    require_cuda = bool(config.get("require_cuda", False))
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("this training profile requires CUDA")
        return {"enabled": False, "reason": "cuda_unavailable"}

    torch.cuda.set_device(device_id)
    name = torch.cuda.get_device_name(device_id)
    capability = tuple(torch.cuda.get_device_capability(device_id))
    required_name = config.get("require_device_name_contains")
    if required_name is not None and str(required_name).lower() not in name.lower():
        raise RuntimeError(
            f"CUDA device {name!r} does not contain {required_name!r}")
    minimum = tuple(int(value) for value in config.get(
        "minimum_compute_capability", (0, 0)))
    if len(minimum) != 2 or capability < minimum:
        raise RuntimeError(
            f"CUDA capability {capability} is below required {minimum}")

    matmul_precision = str(config.get("float32_matmul_precision", "high"))
    if matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError(
            "float32_matmul_precision must be highest, high, or medium")
    torch.set_float32_matmul_precision(matmul_precision)

    allow_tf32 = bool(config.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))
    torch.backends.cudnn.deterministic = bool(
        config.get("cudnn_deterministic", False))

    reduced_bf16 = bool(config.get("allow_bf16_reduced_precision_reduction", True))
    if hasattr(torch.backends.cuda.matmul,
               "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_bf16

    fa3 = config.get("flash_attention3", {}) or {}
    if not isinstance(fa3, Mapping):
        raise TypeError("runtime.flash_attention3 must be a mapping")
    require_fa3 = bool(fa3.get("required", False))
    fa3_report = None
    if require_fa3:
        os.environ["ZVLA_REQUIRE_FLASH_ATTENTION3"] = "1"
        from fa2_custom_mask import (
            flash_attention3_build_info,
            flash_attention3_supported,
        )
        if not flash_attention3_supported():
            raise RuntimeError(
                "runtime requires custom FlashAttention-3 on an SM90 GPU")
        fa3_report = flash_attention3_build_info(load_extension=True)

    sdp = config.get("sdp", {}) or {}
    if not isinstance(sdp, Mapping):
        raise TypeError("runtime.sdp must be a mapping")
    sdp_values = {
        "flash": bool(sdp.get("flash", True)),
        "mem_efficient": bool(sdp.get("mem_efficient", True)),
        # Keep a correctness fallback for uncommon dense masks. Eligible H800
        # calls are still dispatched to Flash/cuDNN ahead of this backend.
        "math": bool(sdp.get("math", True)),
        "cudnn": bool(sdp.get("cudnn", True)),
    }
    torch.backends.cuda.enable_flash_sdp(sdp_values["flash"])
    torch.backends.cuda.enable_mem_efficient_sdp(sdp_values["mem_efficient"])
    torch.backends.cuda.enable_math_sdp(sdp_values["math"])
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(sdp_values["cudnn"])

    return {
        "enabled": True,
        "device": name,
        "compute_capability": capability,
        "float32_matmul_precision": matmul_precision,
        "allow_tf32": allow_tf32,
        "allow_bf16_reduced_precision_reduction": reduced_bf16,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "flash_attention3": fa3_report,
        "sdp": sdp_values,
    }


def build_fsdp2_device_mesh(
    config: Optional[Mapping[str, Any]],
):
    """Build a strict FSDP/HSDP mesh, optionally with a trivial TP axis."""
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError("distributed.device_mesh must be a mapping")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("device_mesh requires an initialized process group")

    raw_shape = config.get("shape")
    if not isinstance(raw_shape, (list, tuple)) or not raw_shape:
        raise ValueError("device_mesh.shape must be a non-empty sequence")
    shape = [int(value) for value in raw_shape]
    if len(shape) not in {1, 2, 3}:
        raise ValueError("FSDP device meshes must have one to three dimensions")
    if shape.count(-1) > 1 or any(value == 0 or value < -1 for value in shape):
        raise ValueError("device_mesh.shape permits at most one inferred -1")
    world_size = dist.get_world_size()
    if -1 in shape:
        known = math.prod(value for value in shape if value != -1)
        if known <= 0 or world_size % known:
            raise ValueError(
                f"world_size={world_size} cannot infer mesh shape {shape}")
        shape[shape.index(-1)] = world_size // known
    if math.prod(shape) != world_size:
        raise ValueError(
            f"device mesh {tuple(shape)} does not cover world_size={world_size}")

    raw_names = config.get("dim_names")
    names = None
    if raw_names is not None:
        if not isinstance(raw_names, (list, tuple)):
            raise TypeError("device_mesh.dim_names must be a sequence")
        names = tuple(str(value) for value in raw_names)
        if len(names) != len(shape) or len(set(names)) != len(names):
            raise ValueError("device_mesh.dim_names must be unique and match shape")
    if len(shape) == 1 and names is not None and names != ("shard", ):
        raise ValueError("a 1-D FSDP mesh dimension must be named shard")
    if len(shape) == 2 and names != ("replicate", "shard"):
        raise ValueError(
            "a 2-D HSDP mesh must be ordered and named replicate, shard")
    if (len(shape) == 3
            and (names != ("replicate", "shard", "tp") or shape[2] != 1)):
        raise ValueError(
            "a 3-D Megatron HSDP mesh must be ordered and named "
            "replicate, shard, tp with tp=1")

    from torch.distributed.device_mesh import init_device_mesh
    return init_device_mesh(
        "cuda",
        tuple(shape),
        mesh_dim_names=names,
    )
