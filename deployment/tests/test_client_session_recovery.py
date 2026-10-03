"""Recover owned sessions explicitly when reset or release replies are lost."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from zwam.client import PolicyClient
from zwam.contract import PROTOCOL, metadata
from zwam.server import create_server


class RecoverySession:
    def __init__(self):
        self.active = False
        self.resets = []
        self.inferences = 0

    def reset(self, *, prompt, seed):
        self.resets.append((prompt, seed))
        self.active = True

    def infer(self, *_, **__):
        self.inferences += 1
        return np.zeros((4, 23), np.float32), np.ones((4, 23), bool)


@pytest.fixture
def service(monkeypatch):
    monkeypatch.delenv("ZWAM_API_KEY", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    session = RecoverySession()
    server = create_server("127.0.0.1", 0, session, ["task"])
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server, session, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("recovery", ["reset", "close"])
@pytest.mark.parametrize("failure", ["timeout", "invalid-json"])
def test_first_accepted_reset_lost_reply_preserves_ownership_for_explicit_recovery(
    service, monkeypatch, recovery, failure
):
    server, session, url = service
    client = PolicyClient(url)
    original_request = client._request
    lost = False

    def lose_first_reset_reply(path, *args, **kwargs):
        nonlocal lost
        reply = original_request(path, *args, **kwargs)
        if path == "/reset" and not lost:
            lost = True
            if failure == "invalid-json":
                raise json.JSONDecodeError("Lost or invalid reset reply", "invalid", 0)
            raise TimeoutError("Reset reply lost after the server accepted it")
        return reply

    monkeypatch.setattr(client, "_request", lose_first_reset_reply)
    with pytest.raises((TimeoutError, json.JSONDecodeError)):
        client.reset("task", seed=7)
    token = client.token
    assert isinstance(token, str) and len(token) >= 43
    assert server.gate.owner == token
    assert session.resets == [("task", 7)]
    assert not client.active
    with pytest.raises(RuntimeError, match="Reset"):
        client.infer(None, None, None)
    assert session.inferences == 0

    if recovery == "reset":
        client.reset("task", seed=9)
        assert client.token == server.gate.owner == token
        assert client.active and client.sequence == 0
        assert session.resets == [("task", 7), ("task", 9)]
    client.close()
    assert client.token is None and not client.active and client.sequence == 0
    assert server.gate.owner is None and not session.active


MALFORMED_RESET_REPLIES = [
    pytest.param(lambda _token: None, id="null"),
    pytest.param(lambda _token: [], id="array"),
    pytest.param(lambda token: [{"token": token, "sequence": 0}], id="wrapped-object"),
    pytest.param(lambda _token: {}, id="missing-both-fields"),
    pytest.param(lambda token: {"token": token}, id="missing-sequence"),
    pytest.param(lambda _token: {"sequence": 0}, id="missing-token"),
    pytest.param(lambda _token: {"token": "different-owner", "sequence": 0}, id="wrong-token"),
    pytest.param(
        lambda token: {
            "token": ("B" if token[0] != "B" else "A") + token[1:],
            "sequence": 0,
        },
        id="valid-format-different-token",
    ),
    pytest.param(lambda _token: {"token": None, "sequence": 0}, id="null-token"),
    pytest.param(lambda _token: {"token": 0, "sequence": 0}, id="integer-token"),
    pytest.param(lambda _token: {"token": "", "sequence": 0}, id="empty-token"),
    pytest.param(lambda token: {"token": token, "sequence": False}, id="boolean-sequence"),
    pytest.param(lambda token: {"token": token, "sequence": 0.0}, id="float-sequence"),
    pytest.param(lambda token: {"token": token, "sequence": "0"}, id="string-sequence"),
    pytest.param(lambda token: {"token": token, "sequence": -1}, id="negative-sequence"),
    pytest.param(lambda token: {"token": token, "sequence": 1}, id="nonzero-sequence"),
]


@pytest.mark.parametrize("reply", MALFORMED_RESET_REPLIES)
def test_malformed_reset_reply_cannot_replace_secret_or_activate_client(
    service, monkeypatch, reply
):
    server, session, url = service
    client = PolicyClient(url)
    original_request = client._request

    def malformed_reset(path, *args, **kwargs):
        result = original_request(path, *args, **kwargs)
        return reply(client.token) if path == "/reset" else result

    monkeypatch.setattr(client, "_request", malformed_reset)
    with pytest.raises(ValueError) as error:
        client.reset("task")
    token = client.token
    assert isinstance(token, str) and token
    assert server.gate.owner == token
    assert not client.active
    assert token not in str(error.value)
    assert session.resets == [("task", 0)]
    monkeypatch.setattr(client, "_request", original_request)
    client.reset("task")
    assert client.active and client.sequence == 0 and client.token == token
    client.close()


def test_failed_reset_of_active_episode_preserves_secret_and_blocks_inference(service, monkeypatch):
    server, session, url = service
    client = PolicyClient(url)
    client.reset("task")
    token = client.token
    original_request = client._request

    def disconnected(path, *args, **kwargs):
        if path == "/reset":
            raise ConnectionError("Reset request could not be delivered")
        return original_request(path, *args, **kwargs)

    monkeypatch.setattr(client, "_request", disconnected)
    with pytest.raises(ConnectionError):
        client.reset("task")
    assert client.token == server.gate.owner == token
    assert not client.active
    with pytest.raises(RuntimeError, match="Reset"):
        client.infer(None, None, None)
    assert session.inferences == 0
    monkeypatch.setattr(client, "_request", original_request)
    client.close()


@pytest.mark.parametrize("delivered", [False, True], ids=["request-lost", "reply-lost"])
def test_failed_close_preserves_secret_and_explicit_reset_can_recover(
    service, monkeypatch, delivered
):
    server, session, url = service
    client = PolicyClient(url)
    client.reset("task")
    token = client.token
    original_request = client._request

    def lose_release(path, *args, **kwargs):
        if path == "/release":
            if delivered:
                original_request(path, *args, **kwargs)
            raise TimeoutError("Release could not be confirmed")
        return original_request(path, *args, **kwargs)

    monkeypatch.setattr(client, "_request", lose_release)
    with pytest.raises(TimeoutError):
        client.close()
    assert client.token == token and not client.active
    assert server.gate.owner == (None if delivered else token)
    with pytest.raises(RuntimeError, match="Reset"):
        client.infer(None, None, None)
    monkeypatch.setattr(client, "_request", original_request)
    client.reset("task")
    assert client.active and client.token == server.gate.owner == token
    assert client.sequence == 0 and len(session.resets) == 2
    client.close()
    assert client.token is None and not client.active and client.sequence == 0


def test_first_reset_sends_client_token_with_existing_authentication_headers(service, monkeypatch):
    server, _, url = service
    secret = "synthetic-api-key-" * 3
    server.api_key = secret
    monkeypatch.setenv("ZWAM_API_KEY", secret)
    client = PolicyClient(url)
    original_open = client._opener.open
    requests = []

    def record_request(request, *args, **kwargs):
        requests.append(
            (request.full_url, {name.lower(): value for name, value in request.header_items()})
        )
        return original_open(request, *args, **kwargs)

    monkeypatch.setattr(client._opener, "open", record_request)
    client.reset("task")
    token = client.token
    assert requests[0][0].endswith("/reset")
    assert requests[0][1]["x-episode-token"] == token
    assert requests[0][1]["authorization"] == "Bearer " + secret
    assert requests[0][1]["x-sequence"] == "0"
    assert requests[0][1]["content-type"] == "application/json"
    assert requests[0][1]["x-zwam-protocol"] == PROTOCOL
    client.infer(
        np.zeros((1, 3, 8, 8, 3), np.uint8),
        np.zeros(23, np.float32),
        np.ones(23, bool),
    )
    assert requests[1][0].endswith("/infer")
    assert requests[1][1]["x-episode-token"] == token
    assert requests[1][1]["authorization"] == "Bearer " + secret
    assert requests[1][1]["x-zwam-protocol"] == PROTOCOL
    assert requests[1][1]["x-sequence"] == "0"
    assert requests[1][1]["content-type"] == "application/octet-stream"
    client.close()
    assert requests[2][0].endswith("/release")
    assert requests[2][1]["x-episode-token"] == token
    assert requests[2][1]["authorization"] == "Bearer " + secret
    assert requests[2][1]["x-zwam-protocol"] == PROTOCOL
    assert requests[2][1]["x-sequence"] == "1"
    assert client.token is None and not client.active and client.sequence == 0


@pytest.mark.parametrize(
    "reply", [None, [], {}, {"released": False}, {"released": 1}, {"released": "true"}]
)
def test_unconfirmed_release_reply_preserves_secret_and_requires_explicit_recovery(
    service, monkeypatch, reply
):
    server, _, url = service
    client = PolicyClient(url)
    client.reset("task")
    token = client.token
    original_request = client._request

    def malformed_release(path, *args, **kwargs):
        result = original_request(path, *args, **kwargs)
        return reply if path == "/release" else result

    monkeypatch.setattr(client, "_request", malformed_release)
    with pytest.raises(ValueError) as error:
        client.close()
    assert client.token == token and not client.active
    assert token not in str(error.value)
    assert server.gate.owner is None
    monkeypatch.setattr(client, "_request", original_request)
    client.reset("task")
    assert client.token == server.gate.owner == token
    client.close()


@pytest.mark.parametrize("path", ["/reset", "/infer", "/release"])
@pytest.mark.parametrize("version", [None, "zwam-h1-command23-http-npz-v2"])
def test_server_rejects_legacy_post_before_touching_session(service, path, version):
    server, session, url = service
    headers = {"Content-Type": "application/json"}
    if version is not None:
        headers["X-ZWAM-Protocol"] = version
    request = Request(url + path, data=b"{}", headers=headers)
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=5)
    assert error.value.code == 400
    with error.value as response:
        message = json.loads(response.read())["error"]
    assert "upgrade" in message.lower()
    assert "client" in message.lower() and "server" in message.lower()
    assert session.resets == [] and session.inferences == 0
    assert server.gate.owner is None


def test_close_is_local_and_idempotent_after_confirmed_release(service, monkeypatch):
    _, _, url = service
    client = PolicyClient(url)
    client.reset("task")
    client.close()

    def no_request(*_, **__):
        pytest.fail("Closing a released client made another request")

    monkeypatch.setattr(client, "_request", no_request)
    client.close()
    assert client.token is None and not client.active and client.sequence == 0


def test_health_advertises_v3_and_client_rejects_legacy_server_before_reset(service):
    server, session, url = service
    assert PROTOCOL == "zwam-h1-command23-http-npz-v3"
    assert metadata()["protocol"] == PROTOCOL
    server.metadata["protocol"] = "zwam-h1-command23-http-npz-v2"
    with pytest.raises(ValueError, match="protocol"):
        PolicyClient(url)
    assert server.gate.owner is None and session.resets == []
