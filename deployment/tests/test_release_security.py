import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.error import HTTPError

import pytest
from test_doctor import server_files
from zwam.assets import check_visual_assets
from zwam.client import PolicyClient
from zwam.contract import metadata
from zwam.source import source_sha256


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_redirect_never_forwards_credentials_or_body(code, method, monkeypatch):
    received = []

    class Sink(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

        do_POST = do_GET

        def log_message(self, *_):
            pass

    sink = ThreadingHTTPServer(("127.0.0.1", 0), Sink)

    class Origin(Sink):
        def do_GET(self):
            if self.path == "/health":
                body = json.dumps(metadata()).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(code)
                self.send_header("Location", f"http://127.0.0.1:{sink.server_port}/sink")
                self.end_headers()

        do_POST = do_GET

    origin = ThreadingHTTPServer(("127.0.0.1", 0), Origin)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True) for server in (sink, origin)
    ]
    for thread in threads:
        thread.start()
    try:
        monkeypatch.setenv("ZWAM_API_KEY", "synthetic-api-credential")
        client = PolicyClient(f"http://127.0.0.1:{origin.server_port}")
        client.token = "synthetic-episode-credential"
        with pytest.raises(HTTPError) as error:
            client._request(
                "/redirect", b"synthetic observation" if method == "POST" else None, binary=True
            )
        assert error.value.code == code
        error.value.close()
        assert received == []
    finally:
        for server in (origin, sink):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()


def visual_config(tmp_path):
    server_files(tmp_path)
    return json.loads((tmp_path / "visual.json").read_text())


@pytest.mark.parametrize("name", ["config.json", "model.safetensors"])
def test_visual_checksums_require_both_files(tmp_path, name):
    assets = visual_config(tmp_path)
    del assets["vggt_files"][name]
    with pytest.raises(ValueError, match="exactly"):
        check_visual_assets(assets)


@pytest.mark.parametrize("name", ["config.json", "model.safetensors"])
def test_visual_mutation_rejected(tmp_path, name):
    assets = visual_config(tmp_path)
    (tmp_path / "vggt" / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        check_visual_assets(assets)


def test_source_fingerprint_covers_contents_paths_and_rejects_symlinks(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("value = 1\n")
    original = source_sha256(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"cache")
    assert source_sha256(tmp_path) == original
    path.write_text("value = 2\n")
    assert source_sha256(tmp_path) != original
    path.write_text("value = 1\n")
    path.rename(tmp_path / "renamed.py")
    assert source_sha256(tmp_path) != original
    path.symlink_to(tmp_path / "renamed.py")
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        source_sha256(tmp_path)


@pytest.mark.parametrize(
    "component,path",
    [("wan", "wan_source/wan/modules/vae2_2.py"), ("vggt", "vggt_source/vggt/models/vggt.py")],
)
def test_visual_source_mutation_rejected_before_import(tmp_path, component, path):
    assets = visual_config(tmp_path)
    (tmp_path / path).write_text('raise AssertionError("must not execute")')
    with pytest.raises(ValueError, match=f"{component} source checksum mismatch"):
        check_visual_assets(assets)


def test_vggt_load_is_local_and_strict(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from zwam_core.vision.streaming import StreamingVGGTGeometryEncoder

    calls = []

    class VGGT:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return SimpleNamespace(aggregator=torch.nn.Identity())

    for name in ("vggt", "vggt.models", "vggt.models.vggt"):
        module = ModuleType(name)
        module.__file__ = str(tmp_path / name.replace(".", "/") / "__init__.py")
        module.VGGT = VGGT
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(tmp_path))
    StreamingVGGTGeometryEncoder(tmp_path, tmp_path / "weights", None, torch.device("cpu"))
    assert calls == [(str(tmp_path / "weights"), {"local_files_only": True, "strict": True})]
    sys.modules["vggt.models.vggt"].__file__ = str(tmp_path.parent / "other.py")
    with pytest.raises(RuntimeError, match="another source"):
        StreamingVGGTGeometryEncoder(tmp_path, tmp_path / "weights", None, torch.device("cpu"))


def load_release_checker():
    path = Path(__file__).resolve().parents[1] / "scripts/check_release.py"
    spec = importlib.util.spec_from_file_location("release_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracked_ignored_asset_is_not_skipped(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("dist/\n.env\n")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist/private.txt").write_text("restricted")
    (tmp_path / ".env").write_text("TOKEN=synthetic")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "-f", ".env", "dist/private.txt"], check=True
    )
    _, errors = load_release_checker().check(tmp_path)
    assert len(errors) == 2
    assert any(".env" in error for error in errors)
    assert any("dist/private.txt" in error for error in errors)


def test_source_audit_redacts_matches(tmp_path):
    (tmp_path / "secret.py").write_text('address = "' + "192." + '168.1.2"\n')
    _, errors = load_release_checker().check(tmp_path)
    assert errors and "192." not in str(errors)
