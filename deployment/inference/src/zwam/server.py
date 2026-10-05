"""Serve Zerith VA policy inference over HTTP."""

import argparse
import json
import logging
import os
import secrets
import signal
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from .assets import read_json
from .contract import PROTOCOL, metadata
from .protocol import MAX_BYTES, SessionGate, pack


class Handler(BaseHTTPRequestHandler):
    server_version = "ZWAM/0.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, fmt, *args):
        # Request paths/headers may contain identifiers; don't echo them.
        return

    def respond(self, code, body, content_type="application/json"):
        if content_type == "application/json":
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def authorize(self):
        secret = self.server.api_key
        if secret and not secrets.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + secret
        ):
            raise PermissionError("API key required")

    def do_GET(self):
        try:
            self.authorize()
            if self.path == "/health":
                self.respond(200, self.server.metadata)
            else:
                self.respond(404, dict(error="Unknown endpoint"))
        except PermissionError:
            self.respond(403, dict(error="Unauthorized"))

    def do_POST(self):
        token = None
        session_touched = False
        try:
            self.authorize()
            if self.path not in ("/reset", "/infer", "/release"):
                self.respond(404, dict(error="Unknown endpoint"))
                return
            if self.headers.get("X-ZWAM-Protocol") != PROTOCOL:
                self.respond(
                    400,
                    dict(error=f"Expected protocol {PROTOCOL}; upgrade client and server together"),
                )
                return
            size = int(self.headers.get("Content-Length", "-1"))
            limit = MAX_BYTES if self.path == "/infer" else 65536
            if self.headers.get("Transfer-Encoding") or not 0 <= size <= limit:
                raise ValueError("Invalid request length")
            payload = self.rfile.read(size)
            if len(payload) != size:
                raise ValueError("Incomplete request")
            gate = self.server.gate
            token = self.headers.get("X-Episode-Token")
            if self.path == "/reset":
                data = json.loads(payload)
                result = gate.reset(data["prompt"], data["seed"], token)
                session_touched = True
                self.respond(200, result)
            elif self.path == "/release":
                gate.release(token)
                session_touched = True
                self.respond(200, dict(released=True))
            else:
                start = time.monotonic()
                commands, valid = gate.infer(
                    token, int(self.headers.get("X-Sequence", "-1")), payload
                )
                session_touched = True
                logging.info(
                    "inference sequence=%d elapsed_ms=%.2f",
                    gate.sequence - 1,
                    (time.monotonic() - start) * 1000,
                )
                self.respond(200, pack(commands=commands, valid=valid), "application/octet-stream")
        except PermissionError:
            self.respond(403, dict(error="Unauthorized or another episode owns the service"))
        except (ValueError, KeyError, TypeError, TimeoutError):
            self.respond(
                400, dict(error="Invalid request or stale session; reset before continuing")
            )
        except (BrokenPipeError, ConnectionResetError):
            if session_touched and self.server.gate.owner == token:
                self.server.gate.active = False
        except Exception:
            if session_touched and self.server.gate.owner == token:
                self.server.gate.active = False
            logging.exception("Inference failed; explicit reset required")
            self.respond(500, dict(error="Inference failed; inspect server log, then reset"))


def create_server(host, port, session, tasks, api_key=None):
    if host not in ("127.0.0.1", "localhost") and (not api_key or len(api_key) < 32):
        raise ValueError("Non-loopback binding requires a >=32-character ZWAM_API_KEY")
    server = HTTPServer((host, port), Handler)
    server.timeout = 0.5
    server.api_key = api_key
    server.gate = SessionGate(session, tasks)
    server.metadata = dict(metadata(), tasks=list(tasks))
    return server


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--config", required=True, help="Server JSON configuration file")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    host, port = config.get("bind", "127.0.0.1"), int(config.get("port", 55558))
    api_key = os.environ.get("ZWAM_API_KEY")
    # Validate bind/auth and reserve port before allocating GPU resources.
    server = create_server(host, port, None, (), api_key)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = [False]

    def request_stop(*_):
        stop[0] = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    try:
        from .model import load_session

        session, tasks = load_session(config)
        server.gate = SessionGate(session, tasks)
        server.metadata = dict(metadata(), tasks=tasks)
        logging.info("Zerith VA inference ready on %s:%d; tasks=%d", host, port, len(tasks))
        while not stop[0]:
            server.handle_request()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
