"""Eval launcher helpers + dashboard API options."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import app as dash  # noqa: E402
from dashboard.eval_launch import list_checkpoints, list_protocols, options_payload, resolve_checkpoint  # noqa: E402


@pytest.fixture()
def tmp_train(tmp_path: Path) -> Path:
    run = tmp_path / "outputs" / "train" / "act_so101_bag_clip_bowl"
    (run / "best_val").mkdir(parents=True)
    (run / "best_val" / "model.pt").write_bytes(b"fake")
    (run / "last").mkdir(parents=True)
    (run / "last" / "model.pt").write_bytes(b"fake")
    proto = tmp_path / "config" / "eval"
    proto.mkdir(parents=True)
    (proto / "bag_clip_in_bowl.yaml").write_text("task: test\n", encoding="utf-8")
    return tmp_path


def test_list_checkpoints_and_protocols(tmp_train: Path) -> None:
    ckpts = list_checkpoints(tmp_train)
    assert len(ckpts) == 1
    assert ckpts[0]["name"] == "act_so101_bag_clip_bowl"
    assert ckpts[0]["kind"] == "best_val"
    assert "best_val" in ckpts[0]["weights"]
    protos = list_protocols(tmp_train)
    assert any(p["name"] == "bag_clip_in_bowl.yaml" for p in protos)


def test_resolve_checkpoint_weights(tmp_train: Path) -> None:
    run = tmp_train / "outputs" / "train" / "act_so101_bag_clip_bowl"
    assert resolve_checkpoint(str(run), "best_val") == run.resolve()
    assert resolve_checkpoint(str(run), "last") == (run / "last").resolve()


def test_options_payload(tmp_train: Path) -> None:
    payload = options_payload(tmp_train)
    assert "act" in payload["policies"]
    assert payload["checkpoints"]
    assert payload["defaults"]["mode"] == "protocol"


def test_api_eval_options(monkeypatch, tmp_train: Path) -> None:
    monkeypatch.setattr(dash, "ROOT", tmp_train)
    client = TestClient(dash.app)
    r = client.get("/api/eval/options")
    assert r.status_code == 200
    data = r.json()
    assert data["checkpoints"]
    assert "running" in data["status"]
    assert data["defaults"]["n_action_steps"] == 25
    assert data["defaults"]["control_mode"] == "act25"


def test_api_eval_start_missing_checkpoint(monkeypatch, tmp_train: Path) -> None:
    monkeypatch.setattr(dash, "ROOT", tmp_train)
    client = TestClient(dash.app)
    r = client.post(
        "/api/eval/start",
        json={
            "policy": "act",
            "checkpoint": str(tmp_train / "missing"),
            "mode": "timed",
            "duration": 5,
        },
    )
    assert r.status_code == 404
