"""Record and replay loops with injected hardware (no real arms in tests)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset
from so101.hardware import action_dict_from_qpos, qpos_from_obs, rgb_from_obs
from so101.record import run_record_loop
from so101.replay import replay_episode


class FakeKeys:
    def __init__(self, by_iter: dict[int, str]) -> None:
        self.by_iter = by_iter
        self.i = 0

    def poll(self) -> str | None:
        key = self.by_iter.get(self.i)
        self.i += 1
        return key


class FakeFollower:
    def __init__(self, h: int = 24, w: int = 32) -> None:
        self.h = h
        self.w = w
        self.sent: list[np.ndarray] = []
        self.t = 0

    def get_observation(self) -> dict:
        q = np.arange(6, dtype=np.float32) + self.t
        img = np.full((self.h, self.w, 3), min(self.t * 20, 255), dtype=np.uint8)
        obs = {f"{n}.pos": float(q[i]) for i, n in enumerate(JOINT_NAMES)}
        obs["front"] = img
        self.t += 1
        return obs

    def send_action(self, action: dict) -> dict:
        self.sent.append(qpos_from_obs(action))
        return action


class FakeLeader:
    def get_action(self) -> dict:
        return {f"{n}.pos": float(i + 10) for i, n in enumerate(JOINT_NAMES)}


def _ds(tmp_path: Path) -> SO101Dataset:
    meta = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="front",
        camera_width=32,
        camera_height=24,
        task="Pick pen",
        episode_count=0,
    )
    return SO101Dataset.create(tmp_path, meta)


def test_record_saves_two_episodes_on_n(tmp_path: Path) -> None:
    ds = _ds(tmp_path)
    follower = FakeFollower()
    leader = FakeLeader()
    # poll after each frame: 3 frames, n; reset n; 3 frames, n
    keys = FakeKeys({2: "n", 4: "n", 7: "n"})
    saved = run_record_loop(
        dataset=ds,
        follower=follower,
        leader=leader,
        camera_name="front",
        fps=15,
        keys=keys,
        num_episodes=2,
        sleep=lambda _s: None,
    )
    assert saved == 2
    ds = SO101Dataset(tmp_path)
    assert ds.meta.episode_count == 2
    ep0 = ds.load_episode(0)
    assert ep0.num_frames == 3
    np.testing.assert_allclose(ep0.action[0], np.arange(6) + 10)
    assert len(follower.sent) >= 3


def test_record_r_discards_then_saves(tmp_path: Path) -> None:
    ds = _ds(tmp_path)
    follower = FakeFollower()
    keys = FakeKeys({2: "r", 4: "n"})
    saved = run_record_loop(
        dataset=ds,
        follower=follower,
        leader=FakeLeader(),
        camera_name="front",
        fps=15,
        keys=keys,
        num_episodes=1,
        sleep=lambda _s: None,
    )
    assert saved == 1
    ds = SO101Dataset(tmp_path)
    assert ds.meta.episode_count == 1
    assert ds.load_episode(0).num_frames == 2


def test_replay_sends_episode_actions() -> None:
    follower = FakeFollower()
    action = np.stack([np.arange(6, dtype=np.float32) + i for i in range(4)])
    ep = type("Ep", (), {"action": action, "num_frames": 4})()
    replay_episode(follower, ep, fps=15, sleep=lambda _s: None)
    assert len(follower.sent) == 4
    np.testing.assert_allclose(follower.sent[2], np.arange(6) + 2)


def test_helpers_used_by_record() -> None:
    obs = FakeFollower().get_observation()
    assert rgb_from_obs(obs, "front").shape[-1] == 3
    d = action_dict_from_qpos(np.zeros(6, np.float32))
    assert set(d) == {f"{n}.pos" for n in JOINT_NAMES}


class DualFakeFollower(FakeFollower):
    def get_observation(self) -> dict:
        obs = super().get_observation()
        obs["wrist"] = obs["front"]
        obs["overhead"] = np.full((self.h, self.w, 3), 7, dtype=np.uint8)
        return obs


def test_record_two_cameras(tmp_path: Path) -> None:
    meta = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="wrist",
        camera_names=("wrist", "overhead"),
        camera_width=32,
        camera_height=24,
        task="Pick pen",
        episode_count=0,
    )
    ds = SO101Dataset.create(tmp_path, meta)
    keys = FakeKeys({1: "n"})
    saved = run_record_loop(
        dataset=ds,
        follower=DualFakeFollower(),
        leader=FakeLeader(),
        camera_names=["wrist", "overhead"],
        fps=15,
        keys=keys,
        num_episodes=1,
        sleep=lambda _s: None,
    )
    assert saved == 1
    ep = SO101Dataset(tmp_path).load_episode(0)
    assert ep.load_frames("wrist").shape[0] == 2
    assert ep.load_frames("overhead").shape[0] == 2
