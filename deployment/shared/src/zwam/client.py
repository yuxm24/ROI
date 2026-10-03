"""Synchronous HTTP policy client with explicit inference-session ownership."""

import json
import os
import secrets
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .contract import PROTOCOL, commands23, validate_metadata
from .protocol import MAX_BYTES, pack, unpack, validate_episode_token


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The inference protocol never redirects. Do not forward credentials
        # or observations to a destination chosen by a response.
        raise HTTPError(req.full_url, code, "Inference redirects are disabled", headers, fp)


class PolicyClient:
    """Connect to a ZWAM service and validate its state/action contract.

    Construction fetches server metadata. Call ``reset(prompt)`` before
    inference and ``close()`` when finished. Authentication uses the optional
    ``ZWAM_API_KEY`` environment variable.
    """

    def __init__(self, url, timeout=60):
        parsed = urlparse(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError("Explicit HTTP(S) URL without credentials required")
        self.url, self.timeout = url.rstrip("/"), timeout
        self._opener = build_opener(_RejectRedirects())
        self.token, self.sequence, self.active = None, 0, False
        self.metadata = self._request("/health")
        validate_metadata(self.metadata)

    def _request(self, path, payload=None, *, binary=False):
        headers = {"X-ZWAM-Protocol": PROTOCOL}
        if os.environ.get("ZWAM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["ZWAM_API_KEY"]
        if self.token:
            headers["X-Episode-Token"] = self.token
        headers["X-Sequence"] = str(self.sequence)
        headers["Content-Type"] = "application/octet-stream" if binary else "application/json"
        if payload is not None and not binary:
            payload = json.dumps(payload, allow_nan=False).encode()
        request = Request(self.url + path, data=payload, headers=headers)
        with self._opener.open(request, timeout=self.timeout) as response:
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                raise ValueError("Oversized server response")
            return body if binary else json.loads(body)

    def reset(self, prompt, seed=0):
        """Explicitly reset using a retained secret, including after a lost reply."""
        self.active = False
        if self.token is None:
            self.token = secrets.token_urlsafe(32)
        result = self._request("/reset", dict(prompt=prompt, seed=seed))
        if not isinstance(result, dict):
            raise ValueError("Invalid reset response; explicit reset required")
        validate_episode_token(result.get("token"))
        if (
            not secrets.compare_digest(result["token"], self.token)
            or type(result.get("sequence")) is not int
            or result["sequence"] != 0
        ):
            raise ValueError("Invalid reset response; explicit reset required")
        self.sequence = 0
        self.active = True

    def infer(self, images, state, valid):
        """Return four absolute Command23 targets and their validity masks.

        ``images`` is uint8 RGB [T, 3, H, W, 3], ordered head/left/right;
        T is 1 after reset and 4 on subsequent calls. ``state`` is float32
        [23] and ``valid`` is bool [23]. See the Zerith-VA README for units.
        A failed request invalidates the session; reset before trying again.
        """
        if not self.active:
            raise RuntimeError("Reset before inference")
        try:
            body = self._request(
                "/infer", pack(images=images, state23=state, valid23=valid), binary=True
            )
            result = unpack(body, ("commands", "valid"))
            commands23(result["commands"], result["valid"])
            self.sequence += 1
            return result["commands"], result["valid"]
        except Exception:
            self.active = False
            raise

    def close(self):
        """Release this client's episode without affecting robot hardware."""
        self.active = False
        if self.token:
            result = self._request("/release", {})
            if not isinstance(result, dict) or result.get("released") is not True:
                raise ValueError("Invalid release response; explicit reset or close required")
            self.token = None
            self.sequence = 0
