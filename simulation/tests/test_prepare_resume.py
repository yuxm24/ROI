"""CPU checks for explicit preparation resume and immutable completed caches."""

from __future__ import annotations

import copy
import importlib
import io
import json
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

prepare_module = importlib.import_module("zvla.posttrain.prepare")


def file_snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def write_demo(path, *, marker, state_offset=0, prompt=None, image_offset=0):
    """Write a real small RLDS record so source changes exercise the reader."""
    from zvla.posttrain.tfrecord import TF_EXAMPLE

    example = TF_EXAMPLE()
    features = example.features.feature
    state = np.zeros((4, 8), np.float32)
    state[:, 0] = marker / 100 + state_offset + np.arange(4) / 100
    action = np.zeros((4, 7), np.float32)
    action[:, 0] = np.arange(4) / 10
    action[:, 6] = -1
    features["steps/observation/state"].float_list.value.extend(state.ravel())
    features["steps/action"].float_list.value.extend(action.ravel())
    instruction = prompt or f"move object {marker}"
    features["steps/language_instruction"].bytes_list.value.extend(
        [instruction.encode()] * len(state)
    )
    for camera in ("image", "wrist_image"):
        frames = []
        for frame_index in range(len(state)):
            buffer = io.BytesIO()
            Image.fromarray(np.full((2, 2, 3), marker + image_offset + frame_index, np.uint8)).save(
                buffer, format="PNG"
            )
            frames.append(buffer.getvalue())
        features[f"steps/observation/{camera}"].bytes_list.value.extend(frames)
    payload = example.SerializeToString()
    path.write_bytes(struct.pack("<Q", len(payload)) + b"\0" * 4 + payload + b"\0" * 4)


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    source.mkdir()
    write_demo(source / "train.tfrecord-00000", marker=10)
    write_demo(source / "train.tfrecord-00001", marker=20)
    args = SimpleNamespace(
        bench="libero",
        source=str(source),
        output=str(tmp_path / "prepared"),
        limit=None,
        resume=False,
        wan_model=str(tmp_path / "wan-model"),
        wan_repo=str(tmp_path / "wan-source"),
        vggt_model=str(tmp_path / "vggt-model"),
        vggt_repo=str(tmp_path / "vggt-source"),
        device="cpu",
    )
    case = SimpleNamespace(
        args=args,
        source=source,
        output=Path(args.output),
        identity={"revision": "synthetic-reviewed-assets-v1"},
        loads=[],
        visual_calls=[],
        text_calls=[],
        forbid_encoders=False,
        fail_visual_marker=None,
        fail_text=False,
        after_visual=None,
        visual_entered=None,
        visual_release=None,
    )
    asset_identity = importlib.import_module("zvla.posttrain.asset_identity")
    monkeypatch.setattr(
        asset_identity, "prepare_asset_identity", lambda _: copy.deepcopy(case.identity)
    )

    def load(name):
        def loader(*args, **kwargs):
            if case.forbid_encoders:
                pytest.fail(f"resume validation unexpectedly loaded the {name} encoder")
            case.loads.append(name)
            return text_encoder if name == "text" else object()

        return loader

    def text_encoder(prompts, device):
        assert device.type == "cpu"
        case.text_calls.extend(prompts)
        if case.fail_text:
            raise RuntimeError("synthetic text summary interruption")
        return [torch.full((2, 4096), len(prompt), dtype=torch.float32) for prompt in prompts]

    for module_name, function, label in (
        ("zvla.vision.wan_vae", "load_wan22_vae", "wan"),
        ("zvla.vision.vggt_features", "load_vggt_aggregator", "vggt"),
        ("zvla.vision.wan_assets", "_load_text_encoder", "text"),
    ):
        monkeypatch.setattr(importlib.import_module(module_name), function, load(label))

    def encode_visual(streams, vae, geometry, device):
        assert device.type == "cpu"
        marker = int(streams[0][0, 0, 0, 0])
        case.visual_calls.append(marker)
        if case.visual_entered is not None:
            case.visual_entered.set()
            if not case.visual_release.wait(10):
                raise RuntimeError("synthetic visual blocking timeout")
        if marker == case.fail_visual_marker:
            raise RuntimeError("synthetic second episode interruption")
        if case.after_visual is not None:
            case.after_visual(marker)
        anchors = (len(streams[0]) + 3) // 4
        return (
            np.full((anchors, 48, 42, 14), marker, np.float16),
            np.full((anchors, 3, 49, 2048), marker, np.float16),
        )

    monkeypatch.setattr(prepare_module, "visual_cache", encode_visual)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return case


def run(case, *, resume=False):
    case.args.resume = resume
    prepare_module.prepare(case.args)


def assert_final_dataset(case):
    manifest = json.loads((case.output / "dataset.json").read_text())
    assert manifest["benchmark"] == "libero"
    assert [episode["native_rows"] for episode in manifest["episodes"]] == [4, 4]
    assert [episode["text_ids"] for episode in manifest["episodes"]] == [[0], [1]]
    assert json.loads((case.output / "tasks.json").read_text()) == [
        "move object 10",
        "move object 20",
    ]
    for index, marker in enumerate((10, 20)):
        shard = case.output / "episodes" / f"{index:06d}"
        assert (shard / ".complete").is_file()
        assert np.all(np.load(shard / "video.npy", allow_pickle=False) == marker)
    embeddings = np.load(case.output / "text_embeddings.npy", allow_pickle=False)
    assert embeddings.shape == (2, 64, 4096)
    assert np.all(embeddings[:, :2] == len("move object 10"))
    assert np.all(embeddings[:, 2:] == 0)
    codec = json.loads((case.output / "codec.json").read_text())
    assert codec["state_min"][0][0] == pytest.approx(0.1)
    assert codec["state_max"][0][0] == pytest.approx(0.2)


def test_interrupted_second_episode_reuses_completed_first_episode(preparation):
    case = preparation
    case.fail_visual_marker = 20
    with pytest.raises(RuntimeError, match="second episode interruption"):
        run(case)
    first = case.output / "episodes" / "000000"
    assert (first / ".complete").is_file()
    original_first = file_snapshot(first)
    case.fail_visual_marker = None
    case.loads.clear()
    case.visual_calls.clear()

    run(case, resume=True)

    assert case.visual_calls == [20]
    assert case.loads == ["wan", "vggt", "text"]
    assert file_snapshot(first) == original_first
    assert_final_dataset(case)


def test_completed_resume_skips_all_encoders_and_keeps_every_output_byte(preparation):
    case = preparation
    run(case)
    original = file_snapshot(case.output)
    case.loads.clear()
    case.visual_calls.clear()
    case.text_calls.clear()
    case.forbid_encoders = True

    run(case, resume=True)

    assert not case.loads and not case.visual_calls and not case.text_calls
    assert file_snapshot(case.output) == original


def test_changed_pending_episode_cannot_resume_a_partially_completed_run(preparation):
    case = preparation
    case.fail_visual_marker = 20
    with pytest.raises(RuntimeError, match="second episode interruption"):
        run(case)
    original = file_snapshot(case.output)
    write_demo(case.source / "train.tfrecord-00001", marker=20, state_offset=0.5)
    case.loads.clear()
    case.visual_calls.clear()
    case.forbid_encoders = True

    with pytest.raises((ValueError, RuntimeError)):
        run(case, resume=True)

    assert not case.loads and not case.visual_calls
    assert file_snapshot(case.output) == original


def test_dataset_fingerprint_binds_text_outputs(preparation, monkeypatch):
    from zvla.datasets.benchmark import BenchmarkDataset
    from zvla.vision import wan_assets

    case = preparation
    run(case)
    first = BenchmarkDataset(case.output)
    original = case.output

    def different_text_encoder(*args, **kwargs):
        return lambda prompts, device: [torch.full((2, 4096), 7.0) for _ in prompts]

    monkeypatch.setattr(wan_assets, "_load_text_encoder", different_text_encoder)
    case.output = original.with_name("prepared-other")
    case.args.output = str(case.output)
    run(case)
    second = BenchmarkDataset(case.output)
    assert first.manifest["prepare_identity_sha256"] == second.manifest["prepare_identity_sha256"]
    assert first.dataset_fingerprint != second.dataset_fingerprint


def test_nonempty_output_requires_explicit_resume(preparation):
    case = preparation
    run(case)
    original = file_snapshot(case.output)
    case.forbid_encoders = True
    with pytest.raises((ValueError, RuntimeError), match="resume"):
        run(case)
    assert file_snapshot(case.output) == original


@pytest.mark.parametrize("change", ("state", "prompt", "key", "images"))
def test_changed_episode_is_rejected_before_encoder_loading(preparation, change):
    case = preparation
    run(case)
    original = file_snapshot(case.output)
    first = case.source / "train.tfrecord-00000"
    if change == "key":
        first.rename(case.source / "train.tfrecord-00000-renamed")
    else:
        write_demo(
            first,
            marker=10,
            state_offset=0.5 if change == "state" else 0,
            prompt="a different instruction" if change == "prompt" else None,
            image_offset=1 if change == "images" else 0,
        )
    case.loads.clear()
    case.visual_calls.clear()
    case.forbid_encoders = True

    with pytest.raises((ValueError, RuntimeError)):
        run(case, resume=True)

    assert not case.loads and not case.visual_calls
    assert file_snapshot(case.output) == original


@pytest.mark.parametrize("change", ("limit", "assets"))
def test_changed_preparation_identity_is_rejected_without_writes(preparation, change):
    case = preparation
    run(case)
    original = file_snapshot(case.output)
    if change == "limit":
        case.args.limit = 1
    else:
        case.identity["revision"] = "different-reviewed-assets-v2"
    case.loads.clear()
    case.visual_calls.clear()
    case.forbid_encoders = True

    with pytest.raises((ValueError, RuntimeError)):
        run(case, resume=True)

    assert not case.loads and not case.visual_calls
    assert file_snapshot(case.output) == original


def test_legacy_complete_marker_without_receipts_is_not_trusted(preparation):
    case = preparation
    shard = case.output / "episodes" / "000000"
    shard.mkdir(parents=True)
    np.save(shard / "states.npy", np.zeros((4, 20), np.float32))
    (shard / ".complete").touch()
    original = file_snapshot(case.output)
    case.forbid_encoders = True

    with pytest.raises((ValueError, RuntimeError)):
        run(case, resume=True)

    assert not case.loads and not case.visual_calls
    assert file_snapshot(case.output) == original


@pytest.mark.parametrize("corruption", ("truncated", "modified_values", "missing"))
def test_corrupt_completed_episode_is_rejected_instead_of_reencoded(preparation, corruption):
    case = preparation
    run(case)
    path = case.output / "episodes" / "000000" / "video.npy"
    if corruption == "truncated":
        path.write_bytes(path.read_bytes()[:32])
    elif corruption == "modified_values":
        values = np.load(path, allow_pickle=False)
        values.flat[0] += 1
        np.save(path, values)
    else:
        path.unlink()
    original = file_snapshot(case.output)
    case.loads.clear()
    case.visual_calls.clear()
    case.forbid_encoders = True

    with pytest.raises((ValueError, RuntimeError)):
        run(case, resume=True)

    assert not case.loads and not case.visual_calls
    assert file_snapshot(case.output) == original


def test_interrupted_root_summary_resumes_without_reencoding_episodes(preparation):
    case = preparation
    case.fail_text = True
    with pytest.raises(RuntimeError, match="text summary interruption"):
        run(case)
    original_episodes = file_snapshot(case.output / "episodes")
    assert all((case.output / "episodes" / f"{i:06d}" / ".complete").is_file() for i in range(2))
    case.fail_text = False
    case.loads.clear()
    case.visual_calls.clear()

    run(case, resume=True)

    assert case.loads == ["text"]
    assert not case.visual_calls
    assert file_snapshot(case.output / "episodes") == original_episodes
    assert_final_dataset(case)


def test_assets_replaced_during_encoding_cancel_new_completions_and_allow_retry(preparation):
    case = preparation
    original_identity = copy.deepcopy(case.identity)

    def replace_assets(marker):
        if marker == 20:
            case.identity["revision"] = "assets-replaced-during-encoding"

    case.after_visual = replace_assets
    with pytest.raises((ValueError, RuntimeError)):
        run(case)
    assert not (case.output / "dataset.json").exists()
    assert not list((case.output / "episodes").rglob(".complete"))
    case.identity = original_identity
    case.after_visual = None
    case.loads.clear()
    case.visual_calls.clear()

    run(case, resume=True)

    assert case.visual_calls == [10, 20]
    assert case.loads == ["wan", "vggt", "text"]
    assert_final_dataset(case)


def test_interrupted_dataset_commit_reuses_shards_and_recovers_summary(preparation, monkeypatch):
    case = preparation
    cache = importlib.import_module("zvla.posttrain.cache")
    original_replace = cache.os.replace
    failed = []

    def interrupt_dataset_commit(source, destination, *args, **kwargs):
        source, destination = Path(source), Path(destination)
        if (
            not failed
            and source.name == "dataset.json"
            and source.parent.name.startswith(".summary-staging-")
            and destination == case.output / "dataset.json"
        ):
            failed.append(destination)
            raise OSError("synthetic final dataset commit interruption")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(cache.os, "replace", interrupt_dataset_commit)
    with pytest.raises(OSError, match="final dataset commit interruption"):
        run(case)
    assert failed
    assert not (case.output / "dataset.json").exists()
    original_episodes = file_snapshot(case.output / "episodes")
    case.loads.clear()
    case.visual_calls.clear()

    run(case, resume=True)

    assert case.loads == ["text"]
    assert not case.visual_calls
    assert file_snapshot(case.output / "episodes") == original_episodes
    assert_final_dataset(case)


def test_concurrent_prepare_is_rejected_while_first_run_holds_lock(preparation):
    case = preparation
    case.visual_entered = threading.Event()
    case.visual_release = threading.Event()
    failures = []

    def active_run():
        try:
            run(case)
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=active_run, daemon=True)
    worker.start()
    try:
        assert case.visual_entered.wait(10), "first preparation did not reach visual encoding"
        previous_loads = list(case.loads)
        with pytest.raises((ValueError, RuntimeError), match="lock|already|another process"):
            prepare_module.prepare(copy.copy(case.args))
        assert case.loads == previous_loads
        assert case.visual_calls == [10]
    finally:
        case.visual_release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert not failures
    assert_final_dataset(case)
