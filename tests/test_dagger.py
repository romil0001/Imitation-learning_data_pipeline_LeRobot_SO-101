"""Leader sync and human-gated DAgger without real arms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from so101.dagger import run_dagger_loop
from so101.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset
from so101.dataset.schema import EpisodeMeta
from so101.hardware import action_dict_from_qpos, qpos_from_obs
from so101.leader_sync import sync_leader_to_follower
from so101.train import split_episodes


class FakeKeys:
    def __init__(self, by_iter: dict[int, str]) -> None:
        self.by_iter = by_iter
        self.i = 0

    def poll(self) -> str | None:
        key = self.by_iter.get(self.i)
        self.i += 1
        return key


class FakeFollower:
    def __init__(self) -> None:
        self.t = 0
        self.sent: list[np.ndarray] = []

    def get_observation(self) -> dict:
        q = np.arange(6, dtype=np.float32) + self.t
        img = np.zeros((24, 32, 3), dtype=np.uint8)
        obs = {f"{n}.pos": float(q[i]) for i, n in enumerate(JOINT_NAMES)}
        obs["front"] = img
        self.t += 1
        return obs

    def send_action(self, action: dict) -> dict:
        self.sent.append(qpos_from_obs(action))
        return action


class FakeBus:
    def __init__(self, leader: "BusLeader") -> None:
        self.leader = leader
        self.torque_log: list[bool] = []
        self.goals: list[np.ndarray] = []

    def enable_torque(self) -> None:
        self.torque_log.append(True)

    def disable_torque(self) -> None:
        self.torque_log.append(False)

    def sync_write(self, name: str, values: dict[str, float]) -> None:
        assert name == "Goal_Position"
        q = np.array([values[n] for n in JOINT_NAMES], dtype=np.float32)
        self.goals.append(q.copy())
        self.leader.q = q.copy()


class BusLeader:
    def __init__(self, q: np.ndarray | None = None) -> None:
        self.q = np.zeros(6, dtype=np.float32) if q is None else np.asarray(q, dtype=np.float32)
        self.bus = FakeBus(self)

    def get_action(self) -> dict:
        return action_dict_from_qpos(self.q)

    def send_feedback(self, action: dict) -> None:
        raise NotImplementedError


class FeedbackLeader:
    def __init__(self) -> None:
        self.q = np.zeros(6, dtype=np.float32)
        self.bus = FakeBus(BusLeader())
        self.feedback: list[np.ndarray] = []

    def get_action(self) -> dict:
        return action_dict_from_qpos(self.q)

    def send_feedback(self, action: dict) -> None:
        self.q = qpos_from_obs(action)
        self.feedback.append(self.q.copy())


class FakePolicy:
    def __init__(self) -> None:
        self.resets = 0
        self.calls = 0

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, image, qpos, depth_u16=None):
        self.calls += 1
        return np.zeros(6, dtype=np.float32)


def _ds(tmp_path: Path) -> SO101Dataset:
    meta = DatasetMeta(
        name="unit_demo",
        fps=30,
        joint_names=list(JOINT_NAMES),
        camera_name="front",
        camera_width=32,
        camera_height=24,
        task="Place keys into metal cup",
        episode_count=0,
    )
    return SO101Dataset.create(tmp_path, meta)


def test_sync_uses_bus_when_send_feedback_is_unimplemented() -> None:
    follower = FakeFollower()
    leader = BusLeader(np.zeros(6, dtype=np.float32))
    held = sync_leader_to_follower(leader, follower, n_steps=4, dt=0, sleep=lambda _s: None)
    assert leader.bus.torque_log == [True, False]
    assert len(leader.bus.goals) == 4
    np.testing.assert_allclose(leader.bus.goals[-1], held)
    np.testing.assert_allclose(leader.q, held)
    assert leader.bus.torque_log[-1] is False


def test_sync_prefers_send_feedback() -> None:
    follower = FakeFollower()
    leader = FeedbackLeader()
    held = sync_leader_to_follower(leader, follower, n_steps=3, dt=0, sleep=lambda _s: None)
    assert len(leader.feedback) == 3
    np.testing.assert_allclose(leader.feedback[-1], held)
    assert leader.bus.goals == []
    assert leader.bus.torque_log == [True, False]


def test_dagger_records_only_correction_frames(tmp_path: Path) -> None:
    ds = _ds(tmp_path)
    follower = FakeFollower()
    leader = BusLeader()
    policy = FakePolicy()
    keys = FakeKeys({1: "p", 2: "c", 4: "n"})
    saved = run_dagger_loop(
        dataset=ds,
        follower=follower,
        leader=leader,
        policy=policy,
        camera_names=["front"],
        fps=30,
        keys=keys,
        num_episodes=1,
        sync_steps=2,
        sync_dt=0,
        sleep=lambda _s: None,
    )
    assert saved == 1
    ep = SO101Dataset(tmp_path).load_episode(0)
    assert ep.num_frames == 2
    assert ep.source == "dagger"
    assert ep.intervention is True
    raw = json.loads((tmp_path / "episodes" / "000000" / "episode.json").read_text())
    assert raw["source"] == "dagger"
    assert policy.calls >= 1
    # Policy actions are zeros; recorded actions are the leader pose after sync.
    assert not np.allclose(ep.action, 0)


def test_dagger_pause_does_not_move_leader_until_p(tmp_path: Path) -> None:
    ds = _ds(tmp_path)
    leader = BusLeader()
    run_dagger_loop(
        dataset=ds,
        follower=FakeFollower(),
        leader=leader,
        policy=FakePolicy(),
        camera_names=["front"],
        fps=30,
        keys=FakeKeys({2: "q"}),
        sync_steps=2,
        sync_dt=0,
        sleep=lambda _s: None,
    )
    assert leader.bus.goals == []
    assert leader.bus.torque_log == []
    assert SO101Dataset(tmp_path).meta.episode_count == 0


def test_dagger_c_before_pause_does_not_record(tmp_path: Path) -> None:
    ds = _ds(tmp_path)
    run_dagger_loop(
        dataset=ds,
        follower=FakeFollower(),
        leader=BusLeader(),
        policy=FakePolicy(),
        camera_names=["front"],
        fps=30,
        keys=FakeKeys({0: "c", 1: "q"}),
        sync_steps=2,
        sync_dt=0,
        sleep=lambda _s: None,
    )
    assert SO101Dataset(tmp_path).meta.episode_count == 0


def test_corrections_stay_in_train_split() -> None:
    train, val = split_episodes(4, val_frac=0.5, seed=0, force_train=[2, 3])
    assert set(val).isdisjoint({2, 3})
    assert {2, 3}.issubset(train)
    assert sorted(train + val) == [0, 1, 2, 3]


def test_old_episode_json_defaults_to_teleop(tmp_path: Path) -> None:
    ds = _ds(tmp_path)
    writer = EpisodeWriter(ds, 0)
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    writer.add_frame(np.zeros(6, np.float32), np.ones(6, np.float32), 0.0, img)
    folder = writer.save()
    legacy = EpisodeMeta(index=0, num_frames=1, task="Place keys into metal cup").to_dict()
    legacy.pop("source")
    legacy.pop("intervention")
    (folder / "episode.json").write_text(json.dumps(legacy))
    ep = SO101Dataset(tmp_path).load_episode(0)
    assert ep.source == "teleop"
    assert ep.intervention is False
