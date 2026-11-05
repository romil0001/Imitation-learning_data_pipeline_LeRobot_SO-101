"""Record and replay loops with injected hardware (no real arms in tests)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset
from so101.hardware import action_dict_from_qpos, qpos_from_obs, rgb_from_obs
from so101.record import _wait_for_observation, run_record_loop
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


def test_wait_for_observation_parks_home_on_stale_zmq() -> None:
    class _Bus:
        def sync_read(self, *_a, **_k):
            # Joints still readable while cameras are stale.
            return {n: float(i + 10) for i, n in enumerate(JOINT_NAMES)}

    class FlakyFollower(FakeFollower):
        def __init__(self) -> None:
            super().__init__()
            self.fails_left = 2
            self.bus = _Bus()
            self.config = type("Cfg", (), {"num_read_retries": 0})()

        def get_observation(self) -> dict:
            if self.fails_left > 0:
                self.fails_left -= 1
                raise TimeoutError("latest frame is too old: 1027.3 ms")
            return super().get_observation()

    follower = FlakyFollower()
    home = np.zeros(6, dtype=np.float32)
    obs, did_park = _wait_for_observation(
        follower, sleep=lambda _s: None, home_qpos=home, retries=5
    )
    assert "front" in obs
    assert did_park is True
    assert follower.fails_left == 0
    assert len(follower.sent) > 0
    np.testing.assert_allclose(follower.sent[-1], home, atol=1e-5)


def test_record_discards_episode_after_camera_stall(tmp_path: Path) -> None:
    ds = _ds(tmp_path)

    class _Bus:
        def sync_read(self, *_a, **_k):
            return {n: 0.0 for n in JOINT_NAMES}

    class StallThenOk(FakeFollower):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.bus = _Bus()
            self.config = type("Cfg", (), {"num_read_retries": 0})()

        def get_observation(self) -> dict:
            self.calls += 1
            # After a few good frames, one stale burst, then recover.
            if 3 <= self.calls <= 4:
                raise TimeoutError("latest frame is too old")
            return super().get_observation()

    follower = StallThenOk()
    # After stall+recover, save on next n
    keys = FakeKeys({8: "n"})
    saved = run_record_loop(
        dataset=ds,
        follower=follower,
        leader=FakeLeader(),
        camera_name="front",
        fps=15,
        keys=keys,
        num_episodes=1,
        home_qpos=np.zeros(6, dtype=np.float32),
        sleep=lambda _s: None,
    )
    assert saved == 1
    assert ds.meta.episode_count == 1



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


def test_empty_dataset_meta_can_be_rewritten(tmp_path: Path) -> None:
    old = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="front",
        camera_names=("front",),
        camera_width=32,
        camera_height=24,
        task="old",
        episode_count=0,
        use_depth=False,
    )
    SO101Dataset.create(tmp_path, old)
    fresh = DatasetMeta(
        name="unit_demo",
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="wrist",
        camera_names=("wrist", "overhead"),
        camera_width=32,
        camera_height=24,
        task="Place keys into metal cup",
        episode_count=0,
        use_depth=True,
        depth_camera_name="wrist",
    )
    ds = SO101Dataset.create(tmp_path, fresh)
    assert list(ds.meta.names()) == ["wrist", "overhead"]
    assert ds.meta.use_depth is True
    assert ds.meta.episode_count == 0
