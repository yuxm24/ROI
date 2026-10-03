import io
import secrets
import threading
import zipfile
from urllib.error import HTTPError

import numpy as np
import pytest
from zwam.client import PolicyClient
from zwam.protocol import SessionGate, pack, unpack
from zwam.server import create_server


class Session:
    active = False
    calls = 0
    fail = False

    def reset(self, **_):
        self.active = True

    def infer(self, *_, **__):
        self.calls += 1
        if self.fail:
            raise RuntimeError("mock model failure")
        return np.zeros((4, 23), np.float32), np.ones((4, 23), bool)


def payload(frames=1):
    return pack(
        images=np.zeros((frames, 3, 8, 8, 3), np.uint8),
        state23=np.zeros(23, np.float32),
        valid23=np.ones(23, bool),
    )


def test_npz_roundtrip():
    data = unpack(pack(a=np.ones(3)), ("a",))
    np.testing.assert_array_equal(data["a"], np.ones(3))


def test_pickle_rejected():
    with pytest.raises(ValueError):
        unpack(pack(a=np.array([object()], dtype=object)), ("a",))


def test_npy_forged_shape_rejected_before_allocation():
    entry = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        entry, dict(descr="<f8", fortran_order=False, shape=(10**12,))
    )
    container = io.BytesIO()
    with zipfile.ZipFile(container, "w") as z:
        z.writestr("a.npy", entry.getvalue())
    with pytest.raises(ValueError, match="allocation"):
        unpack(container.getvalue(), ("a",))


def test_zip_duplicate_and_expansion_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.npy", b"x")
        with pytest.warns(UserWarning):
            z.writestr("a.npy", b"y")
    with pytest.raises(ValueError):
        unpack(buf.getvalue(), ("a",))
    with pytest.raises(ValueError):
        unpack(pack(a=np.ones(1000)), ("a",), limit=20)


def test_ownership_sequence_reset():
    s = Session()
    gate = SessionGate(s, ["task"])
    token = gate.reset("task", 0, secrets.token_urlsafe(32))["token"]
    with pytest.raises(PermissionError):
        gate.reset("task", 0)
    gate.infer(token, 0, payload())
    with pytest.raises(ValueError):
        gate.infer(token, 0, payload())
    assert s.calls == 1
    gate.infer(token, 1, payload(4))
    gate.release(token)
    assert not s.active
    assert gate.reset("task", 0, secrets.token_urlsafe(32))["token"] != token


def test_failed_inference_invalidates_causal_state():
    s = Session()
    s.fail = True
    gate = SessionGate(s, ["task"])
    token = gate.reset("task", 0, secrets.token_urlsafe(32))["token"]
    with pytest.raises(RuntimeError):
        gate.infer(token, 0, payload())
    with pytest.raises(ValueError):
        gate.infer(token, 0, payload())
    assert s.calls == 1


@pytest.mark.parametrize("seed", [-1, True, 2**63, 0.5])
def test_seed_validation(seed):
    with pytest.raises(ValueError):
        SessionGate(Session(), ["task"]).reset("task", seed, secrets.token_urlsafe(32))


def test_external_bind_requires_auth():
    with pytest.raises(ValueError):
        create_server("0.0.0.0", 0, Session(), ["task"])


def test_http_integration(monkeypatch):
    monkeypatch.delenv("ZWAM_API_KEY", raising=False)
    server = create_server("127.0.0.1", 0, Session(), ["task"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = PolicyClient(f"http://127.0.0.1:{server.server_port}")
        client.reset("task")
        data = unpack(payload(), ("images", "state23", "valid23"))
        out, valid = client.infer(data["images"], data["state23"], data["valid23"])
        assert out.shape == (4, 23) and valid.all()
        # A lost or rejected response is not silently retried.
        with pytest.raises(HTTPError):
            client.infer(data["images"], data["state23"], data["valid23"])
        assert not client.active
        client.reset("task")
        client.close()
        assert server.gate.owner is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
