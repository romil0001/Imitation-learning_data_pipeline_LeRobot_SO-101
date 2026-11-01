"""Stdin/dashboard operator key bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from so101.keys import AnyKeys, FileKeys, clear_live_keys, push_live_key


def test_push_and_poll_live_key(tmp_path: Path) -> None:
    path = tmp_path / "live_keys.json"
    push_live_key("n", path=path)
    push_live_key("r", path=path)
    src = FileKeys(allowed={"n", "r", "q"}, path=path)
    assert src.poll() == "n"
    assert src.poll() == "r"
    assert src.poll() is None


def test_file_keys_filters_allowed(tmp_path: Path) -> None:
    path = tmp_path / "live_keys.json"
    push_live_key("y", path=path)
    push_live_key("n", path=path)
    src = FileKeys(allowed={"n", "r", "q"}, path=path)
    assert src.poll() == "n"
    assert src.poll() is None


def test_push_rejects_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        push_live_key("z", path=tmp_path / "live_keys.json")


def test_any_keys_prefers_first_source(tmp_path: Path) -> None:
    path = tmp_path / "live_keys.json"
    push_live_key("q", path=path)

    class Once:
        def __init__(self, ch: str | None) -> None:
            self._ch = ch

        def poll(self) -> str | None:
            ch, self._ch = self._ch, None
            return ch

    combined = AnyKeys(Once("n"), FileKeys(path=path))
    assert combined.poll() == "n"
    assert combined.poll() == "q"


def test_file_keys_context_clears_queue(tmp_path: Path) -> None:
    path = tmp_path / "live_keys.json"
    push_live_key("n", path=path)
    with FileKeys(path=path) as keys:
        assert keys.poll() is None
    clear_live_keys(path=path)
    assert not path.is_file()
