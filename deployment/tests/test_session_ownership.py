"""A caller retains ownership through ambiguous reset and model failures."""

import secrets

import numpy as np
import pytest
from zwam.protocol import SessionGate, pack, validate_episode_token


class Session:
    def __init__(self):
        self.active = False
        self.reset_calls = []
        self.infer_calls = 0
        self.fail_reset = False
        self.fail_infer = False

    def reset(self, **arguments):
        self.reset_calls.append(arguments)
        self.active = True
        if self.fail_reset:
            raise RuntimeError("partial reset failure")

    def infer(self, *_, **__):
        self.infer_calls += 1
        if self.fail_infer:
            raise RuntimeError("model failure")
        return np.zeros((4, 23), np.float32), np.ones((4, 23), bool)


def observation(frames=1):
    return pack(
        images=np.zeros((frames, 3, 8, 8, 3), np.uint8),
        state23=np.zeros(23, np.float32),
        valid23=np.ones(23, bool),
    )


INVALID_TOKENS = [
    None,
    "",
    "a" * 42,
    "a" * 44,
    "a" * 42 + "+",
    "a" * 42 + "/",
    "a" * 42 + "=",
    "a" * 42 + " ",
    "a" * 42 + "\n",
    "a" * 42 + "é",
    b"a" * 43,
    ["a" * 43],
]


@pytest.mark.parametrize("token", INVALID_TOKENS)
def test_first_reset_rejects_invalid_tokens_before_claim_or_model_reset(token):
    session = Session()
    gate = SessionGate(session, ["task"])
    with pytest.raises(ValueError, match="43 ASCII"):
        gate.reset("task", 0, token)
    assert gate.owner is None
    assert not gate.active and not session.active
    assert session.reset_calls == []


def test_first_reset_uses_the_secret_already_known_to_the_client(monkeypatch):
    token = secrets.token_urlsafe(32)
    validate_episode_token(token)

    def refuse_generation(*_):
        pytest.fail("The server generated an owner unknown to the caller")

    monkeypatch.setattr(secrets, "token_urlsafe", refuse_generation)
    session = Session()
    gate = SessionGate(session, ["task"])
    result = gate.reset("task", 7, token)
    assert result == {"token": token, "sequence": 0}
    assert gate.owner == token
    assert gate.active and session.active
    assert session.reset_calls == [{"prompt": "task", "seed": 7}]


@pytest.mark.parametrize("active", [False, True], ids=["inactive", "active"])
@pytest.mark.parametrize("operation", ["reset", "release", "infer"])
@pytest.mark.parametrize("token_kind", ["absent", "other-client", "non-ascii"])
def test_owner_cannot_be_replaced_even_when_inactive(active, operation, token_kind):
    session = Session()
    gate = SessionGate(session, ["task"])
    owner = secrets.token_urlsafe(32)
    gate.reset("task", 0, owner)
    gate.active = session.active = active
    intruder = {
        "absent": None,
        "other-client": secrets.token_urlsafe(32),
        "non-ascii": "é" * 43,
    }[token_kind]
    with pytest.raises(PermissionError, match="Invalid episode token"):
        if operation == "reset":
            gate.reset("task", 1, intruder)
        elif operation == "release":
            gate.release(intruder)
        else:
            gate.infer(intruder, 0, observation())
    assert gate.owner == owner
    assert gate.active is active and session.active is active
    assert len(session.reset_calls) == 1
    assert session.infer_calls == 0


@pytest.mark.parametrize("recovery", ["reset", "release"])
def test_lost_first_reset_response_can_be_recovered_with_the_saved_token(recovery):
    session = Session()
    gate = SessionGate(session, ["task"])
    token = secrets.token_urlsafe(32)
    # Discard the first response; the caller knew this token before sending it.
    gate.reset("task", 0, token)
    # The HTTP handler may deactivate a session after response delivery fails.
    gate.active = False
    if recovery == "reset":
        result = gate.reset("task", 1, token)
        assert result == {"token": token, "sequence": 0}
        gate.infer(token, 0, observation())
        assert session.infer_calls == 1
    else:
        gate.release(token)
        assert gate.owner is None
        assert not gate.active and not session.active


@pytest.mark.parametrize("owned", [False, True], ids=["first-reset", "owned-reset"])
@pytest.mark.parametrize("recovery", ["reset", "release"])
def test_partial_reset_failure_retains_owner_and_blocks_inference_until_explicit_recovery(
    owned, recovery
):
    session = Session()
    gate = SessionGate(session, ["task"])
    token = secrets.token_urlsafe(32)
    if owned:
        gate.reset("task", 0, token)
        gate.infer(token, 0, observation())
    calls_before_failure = session.infer_calls
    session.fail_reset = True
    with pytest.raises(RuntimeError, match="partial reset"):
        gate.reset("task", 1, token)
    assert gate.owner == token
    assert not gate.active and not session.active
    with pytest.raises(ValueError, match="inactive session"):
        gate.infer(token, gate.sequence, observation())
    assert session.infer_calls == calls_before_failure
    with pytest.raises(PermissionError):
        gate.reset("task", 2, secrets.token_urlsafe(32))
    if recovery == "reset":
        session.fail_reset = False
        assert gate.reset("task", 2, token) == {"token": token, "sequence": 0}
        gate.infer(token, 0, observation())
        assert session.infer_calls == calls_before_failure + 1
    else:
        gate.release(token)
        assert gate.owner is None
        assert not gate.active and not session.active


@pytest.mark.parametrize("prompt,seed", [("unknown", 0), ("task", -1), ("task", True)])
@pytest.mark.parametrize("owned", [False, True], ids=["unowned", "owned"])
def test_invalid_business_parameters_do_not_claim_or_disrupt_an_owner(prompt, seed, owned):
    session = Session()
    gate = SessionGate(session, ["task"])
    token = secrets.token_urlsafe(32)
    if owned:
        gate.reset("task", 0, token)
        gate.infer(token, 0, observation())
    before = gate.owner, gate.active, gate.sequence, session.active, len(session.reset_calls)
    with pytest.raises(ValueError, match="instruction"):
        gate.reset(prompt, seed, token)
    assert (
        gate.owner,
        gate.active,
        gate.sequence,
        session.active,
        len(session.reset_calls),
    ) == before
    if owned:
        gate.infer(token, 1, observation(4))
    else:
        other = secrets.token_urlsafe(32)
        assert gate.reset("task", 0, other)["token"] == other


def test_inference_failure_keeps_owner_for_reset_and_requires_a_fresh_first_observation():
    session = Session()
    gate = SessionGate(session, ["task"])
    token = secrets.token_urlsafe(32)
    gate.reset("task", 0, token)
    session.fail_infer = True
    with pytest.raises(RuntimeError, match="model failure"):
        gate.infer(token, 0, observation())
    assert gate.owner == token and not gate.active
    with pytest.raises(PermissionError):
        gate.reset("task", 0, secrets.token_urlsafe(32))
    session.fail_infer = False
    gate.reset("task", 0, token)
    with pytest.raises(ValueError, match="Expected RGB"):
        gate.infer(token, 0, observation(4))
    gate.infer(token, 0, observation())
    assert gate.sequence == 1


def test_released_owner_cannot_reset_infer_or_release_a_new_owners_session():
    session = Session()
    gate = SessionGate(session, ["task"])
    former, current = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    gate.reset("task", 0, former)
    gate.release(former)
    gate.reset("task", 0, current)
    with pytest.raises(PermissionError):
        gate.reset("task", 0, former)
    with pytest.raises(PermissionError):
        gate.infer(former, 0, observation())
    with pytest.raises(PermissionError):
        gate.release(former)
    assert gate.owner == current and gate.active
    gate.infer(current, 0, observation())
