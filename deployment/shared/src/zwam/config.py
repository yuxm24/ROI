"""Bounded JSON loading shared by the inference and H1 processes."""

import json
from pathlib import Path


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("JSON document too large")

    def invalid(value):
        raise ValueError(f"Invalid JSON number: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)
