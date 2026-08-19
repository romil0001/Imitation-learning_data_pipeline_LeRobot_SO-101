"""GDM-style scored eval protocol (no real arms)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from so101.cli import build_parser
from so101.config import project_root
from so101.dataset.schema import JOINT_NAMES
from so101.eval_protocol import load_protocol, run_protocol
from so101.hardware import qpos_from_obs

PROTOCOL_PATH = project_root() / "config" / "eval" / "pen_to_keys.yaml"


class FakeFollower:
    def __init__(self) -> None:
        self.t = 0
        self.sent: list[np.ndarray] = []

    def get_observation(self) -> dict:
        q = np.zeros(6, dtype=np.float32)
        q[0] = float(self.t)
        img = np.zeros((24, 32, 3), dtype=np.uint8)
        obs = {f"{n}.pos": float(q[i]) for i, n in enumerate(JOINT_NAMES)}
        obs["front"] = img
        self.t += 1
        return obs

    def send_action(self, action: dict) -> dict:
        self.sent.append(qpos_from_obs(action))
        return action


class StubPolicy:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, image_rgb: np.ndarray, state: np.ndarray) -> np.ndarray:
        assert image_rgb.shape == (24, 32, 3)
        out = state.copy()
        out[1] = 42.0
        return out


class QueueKeys:
    def __init__(self, items: list[str | None]) -> None:
        self.items = list(items)

    def poll(self) -> str | None:
        if not self.items:
            return None
        return self.items.pop(0)


def _tiny_protocol_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(
        "\n".join(
            [
                "task: Pick pen and place it next to the keys",
                "timeout_s: 20",
                "n_trials_per_condition: 5",
                "success_criterion: Pen in keys zone, gripper clear, no knockdown.",
                "conditions:",
                "  - id: pen=left",
                "    pen: left",
                "    lighting: overhead",
                "    distractors: none",
                "    split: id",
                "  - id: pen=center",
                "    pen: center",
                "    lighting: overhead",
                "    distractors: none",
                "    split: id",
            ]
        )
        + "\n"
    )
    return path


def test_load_pen_to_keys_protocol() -> None:
    proto = load_protocol(PROTOCOL_PATH)
    assert proto.task == "Pick pen and place it next to the keys"
    assert proto.timeout_s == 60
    assert proto.n_trials_per_condition == 5
    assert "keys" in proto.success_criterion.lower()
    assert [c.id for c in proto.conditions] == ["pen=left", "pen=center", "pen=right"]
    assert all(c.lighting == "overhead" and c.distractors == "none" and c.split == "id" for c in proto.conditions)
    assert [c.pen for c in proto.conditions] == ["left", "center", "right"]


def test_run_protocol_writes_summary_and_trial_logs(tmp_path: Path) -> None:
    proto = load_protocol(_tiny_protocol_yaml(tmp_path))
    run_dir = tmp_path / "run"
    # 2 conditions × 2 trials (override), 2 control steps each → 2 None polls, then a score.
    keys = QueueKeys(
        [
            None,
            None,
            "y",
            None,
            None,
            "n",
            None,
            None,
            "s",
            None,
            None,
            "f",
        ]
    )
    resets: list[str] = []

    def on_reset(condition, trial_i, n_trials) -> None:
        resets.append(f"{condition.id}:{trial_i + 1}/{n_trials}")

    summary = run_protocol(
        protocol=proto,
        follower=FakeFollower(),
        policy=StubPolicy(),
        camera_name="front",
        fps=15,
        run_dir=run_dir,
        keys=keys,
        n_trials=2,
        reset_confirm=on_reset,
        sleep=lambda _s: None,
        max_steps=2,
    )
    assert resets == ["pen=left:1/2", "pen=left:2/2", "pen=center:1/2", "pen=center:2/2"]
    assert summary["n_trials"] == 4
    assert summary["n_success"] == 2
    assert summary["pc_success"] == 50.0
    assert summary["per_condition"] == {
        "pen=left": {"n": 2, "n_success": 1, "pc_success": 50.0},
        "pen=center": {"n": 2, "n_success": 1, "pc_success": 50.0},
    }
    written = json.loads((run_dir / "summary.json").read_text())
    assert written == summary
    dumped = json.loads((run_dir / "protocol.json").read_text())
    assert dumped["task"] == proto.task
    assert dumped["n_trials_per_condition"] == 2
    result = json.loads((run_dir / "trials" / "0001" / "result.json").read_text())
    condition = json.loads((run_dir / "trials" / "0001" / "condition.json").read_text())
    assert result["success"] is True
    assert result["early_stop"] is False
    assert result["steps"] == 2
    assert condition["id"] == "pen=left"
    last = json.loads((run_dir / "trials" / "0004" / "result.json").read_text())
    assert last["success"] is False


def test_run_protocol_early_stop_on_y_during_rollout(tmp_path: Path) -> None:
    path = tmp_path / "one.yaml"
    path.write_text(
        "\n".join(
            [
                "task: Pick pen and place it next to the keys",
                "timeout_s: 20",
                "n_trials_per_condition: 1",
                "success_criterion: Pen in keys zone, gripper clear, no knockdown.",
                "conditions:",
                "  - id: pen=left",
                "    pen: left",
                "    lighting: overhead",
                "    distractors: none",
                "    split: id",
            ]
        )
        + "\n"
    )
    proto = load_protocol(path)
    keys = QueueKeys([None, None, "y"])
    policy = StubPolicy()
    summary = run_protocol(
        protocol=proto,
        follower=FakeFollower(),
        policy=policy,
        camera_name="front",
        fps=15,
        run_dir=tmp_path / "run",
        keys=keys,
        reset_confirm=lambda *_a: None,
        sleep=lambda _s: None,
        max_steps=50,
    )
    assert summary["n_success"] == 1
    result = json.loads((tmp_path / "run" / "trials" / "0001" / "result.json").read_text())
    assert result["early_stop"] is True
    assert result["success"] is True
    assert 0 < result["steps"] < 50
    assert policy.resets >= 1


def test_eval_cli_accepts_protocol_and_n_trials() -> None:
    args = build_parser().parse_args(
        ["eval", "--protocol", "config/eval/pen_to_keys.yaml", "--n-trials", "5"]
    )
    assert args.protocol == "config/eval/pen_to_keys.yaml"
    assert args.n_trials == 5
    default = build_parser().parse_args(["eval"])
    assert default.protocol == ""
    assert default.n_trials is None
    assert default.duration == 60.0
