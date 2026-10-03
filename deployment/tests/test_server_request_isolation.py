"""Only a request that changed its owned session can invalidate it on reply loss."""

import io
import json
import secrets
from types import SimpleNamespace

import pytest
from test_protocol import Session
from zwam.contract import PROTOCOL
from zwam.protocol import SessionGate
from zwam.server import Handler


class Unreadable:
    def read(self, *_):
        pytest.fail("A rejected protocol request attempted to read its body")


def owned_handler(path, *, headers=None, body=b""):
    session = Session()
    gate = SessionGate(session, ["task"])
    token = secrets.token_urlsafe(32)
    gate.reset("task", 0, token)
    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(api_key=None, gate=gate)
    handler.path = path
    handler.headers = headers or {}
    handler.rfile = io.BytesIO(body)
    return handler, gate, session, token


@pytest.mark.parametrize("path", ["/reset", "/infer", "/release"])
@pytest.mark.parametrize("version", [None, "zwam-h1-command23-http-npz-v2"])
@pytest.mark.parametrize("disconnect", [BrokenPipeError, ConnectionResetError])
def test_rejected_protocol_reply_loss_preserves_existing_owner_and_active_state(
    path, version, disconnect
):
    handler, gate, session, token = owned_handler(path)
    # Even a rejected version carrying the owner's token must not disturb it.
    handler.headers = {"X-Episode-Token": token}
    if version is not None:
        handler.headers["X-ZWAM-Protocol"] = version
    handler.rfile = Unreadable()

    def lost_reply(code, *_):
        assert code == 400
        raise disconnect("Rejected request reply could not be delivered")

    handler.respond = lost_reply
    handler.do_POST()
    assert gate.owner == token and gate.active and session.active
    assert gate.sequence == 0


@pytest.mark.parametrize("disconnect", [BrokenPipeError, ConnectionResetError])
def test_accepted_reset_reply_loss_invalidates_only_that_owned_session(disconnect):
    body = json.dumps(dict(prompt="task", seed=9)).encode()
    handler, gate, session, token = owned_handler("/reset", body=body)
    handler.headers = {
        "X-ZWAM-Protocol": PROTOCOL,
        "X-Episode-Token": token,
        "Content-Length": str(len(body)),
    }

    def lost_reply(code, *_):
        assert code == 200
        raise disconnect("Accepted reset reply could not be delivered")

    handler.respond = lost_reply
    handler.do_POST()
    assert gate.owner == token and not gate.active
    gate.reset("task", 0, token)
    assert gate.owner == token and gate.active and session.active


@pytest.mark.parametrize("use_owner_token", [False, True], ids=["other-client", "owner"])
def test_json_parser_failure_preserves_existing_session_before_gate_is_called(
    use_owner_token, monkeypatch
):
    body = b"{}"
    handler, gate, session, token = owned_handler("/reset", body=body)
    handler.headers = {
        "X-ZWAM-Protocol": PROTOCOL,
        "X-Episode-Token": token if use_owner_token else secrets.token_urlsafe(32),
        "Content-Length": str(len(body)),
    }
    responses = []
    handler.respond = lambda code, response: responses.append((code, response))

    def parser_failure(*_, **__):
        # Inject the parser failure independently of the runner's recursion limit.
        raise RecursionError("Synthetic JSON parser failure")

    monkeypatch.setattr("zwam.server.json.loads", parser_failure)
    handler.do_POST()
    assert responses[0][0] == 500
    assert gate.owner == token and gate.active and session.active
    assert gate.sequence == 0
