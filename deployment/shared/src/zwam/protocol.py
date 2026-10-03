"""Bounded, pickle-free NPZ transport and single-owner causal sessions."""

import io
import math
import re
import secrets
import zipfile

import numpy as np

from .contract import CAMERAS, commands23, images_rgb, state23

MAX_BYTES = 64 * 1024 * 1024
_EPISODE_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")


def validate_episode_token(token):
    """Require the ASCII wire format of a client-generated token_urlsafe(32)."""
    if not isinstance(token, str) or _EPISODE_TOKEN.fullmatch(token) is None:
        raise ValueError("Episode token must be 43 ASCII URL-safe characters")


def pack(**arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def unpack(payload, keys, limit=MAX_BYTES):
    if len(payload) > limit:
        raise ValueError("NPZ exceeds byte limit")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        if (
            len(entries) != len(keys)
            or {e.filename for e in entries} != {k + ".npy" for k in keys}
            or sum(e.file_size for e in entries) > limit
        ):
            raise ValueError("Unexpected or oversized NPZ members")
        # A small ZIP member can declare an enormous NPY shape. Validate its
        # header/data byte agreement BEFORE np.load allocates an array.
        for entry in entries:
            with archive.open(entry) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError("Unsupported NPY version")
                if (
                    dtype.hasobject
                    or dtype.fields
                    or dtype.kind not in "buif"
                    or len(shape) > 6
                    or any(n < 0 for n in shape)
                ):
                    raise ValueError("Unsupported NPY dtype/shape")
                size = math.prod(shape) * dtype.itemsize
                if size > limit or size != entry.file_size - member.tell():
                    raise ValueError("NPY declared allocation differs from payload")
    with np.load(io.BytesIO(payload), allow_pickle=False) as values:
        return {key: values[key].copy() for key in keys}


class SessionGate:
    """Caller serializes all methods. Never retry ambiguous inference requests."""

    def __init__(self, session, tasks):
        self.session, self.tasks = session, tuple(tasks)
        self.owner = None
        self.active = False
        self.sequence = 0

    def authorize(self, token):
        try:
            validate_episode_token(token)
        except ValueError as error:
            raise PermissionError("Invalid episode token") from error
        if not self.owner or not secrets.compare_digest(token, self.owner):
            raise PermissionError("Invalid episode token")

    def reset(self, prompt, seed, token=None):
        if self.owner:
            self.authorize(token)
        else:
            validate_episode_token(token)
        if prompt not in self.tasks or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Exact text-bank instruction and nonnegative integer seed required")
        self.active = False
        # Claim only after business validation, but before a reset can partially
        # mutate model state. Retain ownership on reset or response failure so
        # only this client can explicitly recover or release the session.
        self.owner = self.owner or token
        self.session.active = False
        try:
            self.session.reset(prompt=prompt, seed=seed)
        except Exception:
            self.session.active = False
            raise
        self.active, self.sequence = True, 0
        return dict(token=self.owner, sequence=0)

    def release(self, token):
        self.authorize(token)
        self.active = False
        self.session.active = False
        self.owner = None

    def infer(self, token, sequence, payload):
        self.authorize(token)
        if not self.active or type(sequence) is not int or sequence != self.sequence:
            raise ValueError("Stale sequence/inactive session; explicitly reset, never retry")
        data = unpack(payload, ("images", "state23", "valid23"))
        images_rgb(data["images"], 1 if sequence == 0 else 4)
        state23(data["state23"], data["valid23"])
        try:
            commands, valid = self.session.infer(
                data["images"], data["state23"], data["valid23"], camera_names=CAMERAS
            )
            commands23(commands, valid)
            self.sequence += 1
            return commands, valid
        except Exception:
            self.active = False
            raise
