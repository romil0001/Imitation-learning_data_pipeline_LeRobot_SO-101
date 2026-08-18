"""Decode LeRobot-style ZMQ JSON camera payloads."""

from __future__ import annotations

import base64


def jpegs_from_zmq_payload(data: dict) -> dict[str, bytes]:
    images = data.get("images") or {}
    out: dict[str, bytes] = {}
    for name, b64 in images.items():
        if not isinstance(b64, str) or not b64:
            continue
        out[str(name)] = base64.b64decode(b64)
    return out
