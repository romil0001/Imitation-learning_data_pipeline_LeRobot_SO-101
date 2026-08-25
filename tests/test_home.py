"""Slow return-to-home before eval (no real arms)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset
from so101.eval_protocol import load_protocol, run_protocol
from so101.hardware import qpos_from_obs
from so101.home import interpolate_to_qpos, mean_start_qpos


class FakeFollower:
    def __init__(self, start: np.ndarray) -> None:
        self.start = np.asarray(start, dtype=np.float32)
        self.sent: list[np.ndarray] = []

    def get_observation(self) -> dict:
        q = self.start if not self.sent else self.sent[-1]
        obs = {f"{n}.pos": float(q[i]) for i, n in enumerate(JOINT_NAMES)}
        obs["front"] = np.zeros((24, 32, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict) -> dict:
        self.sent.append(qpos_from_obs(action))
        return action


class StubPolicy:
    def reset(self) -> None:
        return None

    def select_action(self, image_rgb: np.ndarray, state: np.ndarray, depth_u16=None) -> np.ndarray:
        return np.full(6, 99.0, dtype=np.float32)


class QueueKeys:
    def __init__(self, items: list[str | None]) -> None:
        self.items = list(items)

    def poll(self) -> str | None:
        if not self.items:
            return None
        return self.items.pop(0)


def test_interpolate_to_qpos_reaches_target() -> None:
    start = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 40.0], dtype=np.float32)
    home = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0], dtype=np.float32)
    follower = FakeFollower(start)
    interpolate_to_qpos(follower, home, n_steps=4, dt=0.0, sleep=lambda _s: None)
    assert len(follower.sent) == 4
    np.testing.assert_allclose(follower.sent[0], start + (home - start) / 4, atol=1e-5)
    np.testing.assert_allclose(follower.sent[-1], home, atol=1e-5)


def test_mean_start_qpos_averages_first_states(tmp_path: Path) -> None:
    ds = SO101Dataset.create(
        tmp_path,
        DatasetMeta(
            name="synth",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="front",
            camera_width=8,
            camera_height=8,
            task="synth",
            episode_count=0,
        ),
    )
    starts = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        np.array([3.0, 2.0, 1.0, 0.0, -1.0, 10.0], dtype=np.float32),
    ]
    for i, start in enumerate(starts):
        w = EpisodeWriter(ds, i)
        for k in range(3):
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            w.add_frame(start + k, start + k, k / 15.0, img)
        w.save()
    got = mean_start_qpos(SO101Dataset(tmp_path))
    np.testing.assert_allclose(got, (starts[0] + starts[1]) / 2, atol=1e-5)


def test_run_protocol_homes_before_each_trial(tmp_path: Path) -> None:
    path = tmp_path / "one.yaml"
    path.write_text(
        "\n".join(
            [
                "task: Pick pen",
                "timeout_s: 20",
                "n_trials_per_condition: 1",
                "success_criterion: done",
                "home_qpos: [0.0, 1.0, 2.0, 3.0, 4.0, 10.0]",
                "home_steps: 3",
                "home_dt: 0.0",
                "conditions:",
                "  - id: pen=left",
                "    pen: left",
            ]
        )
        + "\n"
    )
    proto = load_protocol(path)
    np.testing.assert_allclose(proto.home_qpos, [0.0, 1.0, 2.0, 3.0, 4.0, 10.0])
    assert proto.home_steps == 3
    follower = FakeFollower(np.array([9.0, 9.0, 9.0, 9.0, 9.0, 40.0], dtype=np.float32))
    keys = QueueKeys([None, "y"])
    run_protocol(
        protocol=proto,
        follower=follower,
        policy=StubPolicy(),
        camera_name="front",
        fps=15,
        run_dir=tmp_path / "run",
        keys=keys,
        reset_confirm=lambda *_a: None,
        sleep=lambda _s: None,
        max_steps=1,
    )
    home = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0], dtype=np.float32)
    np.testing.assert_allclose(follower.sent[2], home, atol=1e-5)
    np.testing.assert_allclose(follower.sent[3], np.full(6, 99.0), atol=1e-5)
