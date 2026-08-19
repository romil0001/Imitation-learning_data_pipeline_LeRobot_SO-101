"""Keyboard helper for record: n save, r redo, q quit."""

from __future__ import annotations

import select
import sys
from typing import Protocol


class KeySource(Protocol):
    def poll(self) -> str | None: ...


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
