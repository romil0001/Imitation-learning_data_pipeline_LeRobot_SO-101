"""Scored eval protocol tests (no real arms)."""

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
BAG_CLIP_PROTOCOL_PATH = project_root() / "config" / "eval" / "bag_clip_in_bowl.yaml"


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

    def select_action(self, image_rgb: np.ndarray, state: np.ndarray, depth_u16=None) -> np.ndarray:
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
    assert proto.task == "Place keys into metal cup"
    assert proto.timeout_s == 60
    assert proto.n_trials_per_condition == 5
    assert "metal cup" in proto.success_criterion.lower()
    assert [c.id for c in proto.conditions] == ["keys=left", "keys=center", "keys=right"]
    assert all(c.lighting == "overhead" and c.distractors == "none" and c.split == "id" for c in proto.conditions)
    assert [c.pen for c in proto.conditions] == ["left", "center", "right"]


def test_load_bag_clip_in_bowl_protocol() -> None:
    proto = load_protocol(BAG_CLIP_PROTOCOL_PATH)
    assert proto.task == "Place bag clip in bowl"
    assert proto.timeout_s == 60
    assert proto.n_trials_per_condition == 5
    assert "bowl" in proto.success_criterion.lower()
    assert [c.id for c in proto.conditions] == ["clip=left", "clip=center", "clip=right"]
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
    assert default.save_successes is True
    assert default.success_dataset == ""
    off = build_parser().parse_args(["eval", "--no-save-successes"])
    assert off.save_successes is False


def test_run_protocol_saves_success_episodes(tmp_path: Path) -> None:
    from so101.dataset import DatasetMeta, JOINT_NAMES, SO101Dataset

    base_root = tmp_path / "data" / "datasets" / "demo"
    success_root = tmp_path / "data" / "datasets" / "demo_eval"
    SO101Dataset.create(
        base_root,
        DatasetMeta(
            name="demo",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="front",
            camera_names=("front",),
            camera_width=32,
            camera_height=24,
            task="Pick pen",
            episode_count=0,
            use_depth=False,
        ),
    )
    # Reuse open_success_dataset against tmp project root layout
    from so101.eval import open_success_dataset

    success_ds = open_success_dataset(
        base_dataset_name="demo",
        success_dataset_name="demo_eval",
        root=tmp_path,
    )
    assert success_ds.root == success_root

    proto = load_protocol(_tiny_protocol_yaml(tmp_path))
    keys = QueueKeys(
        [
            None,
            None,
            "y",  # success → save
            None,
            None,
            "n",  # fail → discard
            None,
            None,
            "y",
            None,
            None,
            "n",
        ]
    )
    summary = run_protocol(
        protocol=proto,
        follower=FakeFollower(),
        policy=StubPolicy(),
        camera_name="front",
        fps=15,
        run_dir=tmp_path / "run",
        keys=keys,
        n_trials=2,
        reset_confirm=lambda *_a: None,
        sleep=lambda _s: None,
        max_steps=2,
        success_dataset=success_ds,
        policy_tag="act",
    )
    assert summary["n_success"] == 2
    assert summary["n_saved_success"] == 2
    assert success_ds.meta.episode_count == 2
    ep0 = success_root / "episodes" / "000000"
    assert (ep0 / "front.mp4").is_file()
    assert (ep0 / "state.npy").is_file()
    assert (ep0 / "action.npy").is_file()
    meta = json.loads((ep0 / "episode.json").read_text())
    assert meta["source"] == "eval_success"
    assert "eval_success" in meta["tags"]
    r0 = json.loads((tmp_path / "run" / "trials" / "0001" / "result.json").read_text())
    assert r0["saved_episode"] == 0
    r1 = json.loads((tmp_path / "run" / "trials" / "0002" / "result.json").read_text())
    assert r1["success"] is False
    assert r1["saved_episode"] is None
    assert (success_root / "stats.json").is_file()
