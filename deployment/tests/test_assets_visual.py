import json

import numpy as np
import pytest
from zwam.assets import load_text_bank, sha256, verified_file
from zwam.h1.preflight import acquire_lock


def test_asset_digest_and_containment(tmp_path):
    asset = tmp_path / "a.txt"
    asset.write_text("verified test")
    assert verified_file(tmp_path, "a.txt", sha256(asset)) == asset
    with pytest.raises(ValueError):
        verified_file(tmp_path, "a.txt", "0" * 64)
    link = tmp_path / "escape"
    link.symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        verified_file(tmp_path, "escape/anything", "0" * 64)


def test_frozen_text_bank_exact_prompt(tmp_path):
    (tmp_path / "tasks.json").write_text(json.dumps(["synthetic task"]))
    np.save(tmp_path / "text_embeddings.npy", np.zeros((1, 64, 4096), np.float16))
    (tmp_path / "text_manifest.json").write_text(
        json.dumps(
            dict(
                tasks_sha256=sha256(tmp_path / "tasks.json"),
                embeddings_sha256=sha256(tmp_path / "text_embeddings.npy"),
            )
        )
    )
    tasks, lookup = load_text_bank(tmp_path)
    assert tasks == ["synthetic task"]
    assert lookup(tasks[0]).shape == (64, 4096)
    with pytest.raises(KeyError):
        lookup("different task")


def test_controller_lock_and_symlink(tmp_path):
    lock = tmp_path / "controller.lock"
    with acquire_lock(lock):
        with pytest.raises(BlockingIOError):
            acquire_lock(lock)
    alias = tmp_path / "alias"
    alias.symlink_to(lock)
    with pytest.raises(OSError):
        acquire_lock(alias)


def test_letterbox_preserves_aspect_and_padding():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from zwam_core.vision.letterbox import resize_canonical_canvas, resize_with_pad_rgb

    original = np.full((100, 200, 3), 255, np.uint8)
    canvas, _, _ = resize_with_pad_rgb(original)
    assert canvas.shape == (512, 512, 3)
    assert np.all(canvas[:128] == 0) and np.all(canvas[384:] == 0)
    assert np.all(canvas[128:384] == 255)
    assert resize_canonical_canvas(canvas, 224).shape == (224, 224, 3)


def test_camera_profile_slot_order():
    torch = pytest.importorskip("torch")
    from zwam_core.vision.schema import CameraProfile

    p = CameraProfile("test", ("head", "left", "right"), (0, 1, 2), ("h", "l", "r"))
    x = torch.arange(12).reshape(1, 6, 2)
    torch.testing.assert_close(p.pack_vertical_tensor(x, height_axis=-2), x)
    with pytest.raises(ValueError):
        p.validate_source_camera_names(("left", "head", "right"), label="test")
