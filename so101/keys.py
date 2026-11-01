"""Keyboard helpers for record/eval/dagger: stdin and dashboard live keys."""

from __future__ import annotations

import json
import select
import sys
import time
from pathlib import Path
from typing import Protocol

from so101.config import project_root

LIVE_KEYS_PATH = project_root() / "outputs" / "live_keys.json"

# Keys the operator site may inject (record / dagger / eval).
DASHBOARD_KEYS = frozenset({"n", "r", "q", "y", "p", "c", "s", "f", " "})


class KeySource(Protocol):
    def poll(self) -> str | None: ...


def push_live_key(ch: str, *, path: Path | None = None) -> str:
    """Queue one operator key for record/eval/dagger (dashboard → Mac process)."""
    key = ch.lower() if ch != " " else " "
    if len(key) != 1 or key not in DASHBOARD_KEYS:
        raise ValueError(f"unsupported key {ch!r}")
    out = path or LIVE_KEYS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    if out.is_file():
        try:
            data = json.loads(out.read_text())
            raw = data.get("keys") if isinstance(data, dict) else None
            if isinstance(raw, list):
                keys = [str(x) for x in raw if isinstance(x, str)]
        except (OSError, json.JSONDecodeError):
            keys = []
    keys.append(key)
    # Bound the queue so a stuck consumer cannot grow forever.
    keys = keys[-32:]
    payload = {"ts": time.time(), "keys": keys}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    tmp.replace(out)
    return key


def clear_live_keys(*, path: Path | None = None) -> None:
    out = path or LIVE_KEYS_PATH
    if out.is_file():
        try:
            out.unlink()
        except OSError:
            pass


class StdinKeys:
    def __init__(self, allowed: set[str] | frozenset[str] | None = None) -> None:
        self._fd = sys.stdin.fileno()
        self._old = None
        self._allowed = frozenset(ch.lower() for ch in allowed) if allowed is not None else frozenset({"n", "r", "q"})

    def __enter__(self) -> StdinKeys:
        try:
            import termios
            import tty

            self._termios = termios
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self._old = None
        return self

    def __exit__(self, *exc: object) -> None:
        if self._old is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old)

    def poll(self) -> str | None:
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = sys.stdin.read(1).lower()
        if ch in self._allowed:
            return ch
        return None


class FileKeys:
    """Pop keys written by the dashboard (`push_live_key`)."""

    def __init__(
        self,
        allowed: set[str] | frozenset[str] | None = None,
        *,
        path: Path | None = None,
    ) -> None:
        self._path = path or LIVE_KEYS_PATH
        self._allowed = (
            frozenset(ch.lower() if ch != " " else " " for ch in allowed)
            if allowed is not None
            else frozenset(DASHBOARD_KEYS)
        )

    def __enter__(self) -> FileKeys:
        clear_live_keys(path=self._path)
        return self

    def __exit__(self, *exc: object) -> None:
        clear_live_keys(path=self._path)

    def poll(self) -> str | None:
        if not self._path.is_file():
            return None
        try:
            data = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        raw = data.get("keys") if isinstance(data, dict) else None
        if not isinstance(raw, list) or not raw:
            return None
        keys = [str(x) for x in raw if isinstance(x, str)]
        chosen: str | None = None
        rest: list[str] = []
        for ch in keys:
            if ch not in self._allowed:
                continue
            if chosen is None:
                chosen = ch
            else:
                rest.append(ch)
        payload = {"ts": time.time(), "keys": rest}
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload) + "\n")
            tmp.replace(self._path)
        except OSError:
            return chosen
        return chosen


class AnyKeys:
    """First non-None poll from several key sources (stdin + dashboard)."""

    def __init__(self, *sources: KeySource) -> None:
        self._sources = sources

    def poll(self) -> str | None:
        for src in self._sources:
            ch = src.poll()
            if ch is not None:
                return ch
        return None


class OperatorKeys:
    """Stdin + dashboard file queue; use as ``with OperatorKeys(...) as keys``."""

    def __init__(self, allowed: set[str] | frozenset[str] | None = None) -> None:
        self._stdin = StdinKeys(allowed=allowed)
        self._file = FileKeys(allowed=allowed)
        self._any = AnyKeys(self._stdin, self._file)

    def __enter__(self) -> AnyKeys:
        self._stdin.__enter__()
        self._file.__enter__()
        return self._any

    def __exit__(self, *exc: object) -> None:
        self._file.__exit__(*exc)
        self._stdin.__exit__(*exc)
