"""Replay-capture writes multi-cam + multi-depth without a robot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset
from so101.dataset.schema import DatasetMeta as Meta
from so101.replay_capture import replay_capture_episode


class FakeFollower:
    def __init__(self, q: np.ndarray) -> None:
        self.q = np.asarray(q, dtype=np.float32)
        self.sent: list[np.ndarray] = []

    def get_observation(self) -> dict:
        obs = {f"{n}.pos": float(self.q[i]) for i, n in enumerate(JOINT_NAMES)}
        for name in ("wrist", "overhead", "base"):
            obs[name] = np.zeros((32, 32, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict) -> dict:
        from so101.hardware import qpos_from_obs

        self.q = qpos_from_obs(action)
        self.sent.append(self.q.copy())
        return action


class FakeZMQ:
    def __init__(self) -> None:
        self._rgb = {
            "wrist": np.full((32, 32, 3), 1, dtype=np.uint8),
            "overhead": np.full((32, 32, 3), 2, dtype=np.uint8),
            "base": np.full((32, 32, 3), 3, dtype=np.uint8),
        }
        self._depth = {
            "wrist": np.full((32, 32), 100, dtype=np.uint16),
            "base": np.full((32, 32), 200, dtype=np.uint16),
        }

    def latest_images(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._rgb.items()}

    def latest_depth(self, name: str) -> np.ndarray | None:
        z = self._depth.get(name)
        return None if z is None else z.copy()

    def wait_for_depth(self, name: str, timeout_s: float = 15.0) -> np.ndarray:
        z = self.latest_depth(name)
        assert z is not None
        return z


class SrcEpisode:
    def __init__(self, t: int = 4) -> None:
        self.num_frames = t
        self.action = np.zeros((t, 6), dtype=np.float32)
        self.action[:, 0] = np.arange(t, dtype=np.float32)
        self.state = np.zeros((t, 6), dtype=np.float32)


def test_multi_depth_writer_roundtrip(tmp_path: Path) -> None:
    ds = SO101Dataset.create(
        tmp_path,
        DatasetMeta(
            name="keys",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="wrist",
            camera_names=("wrist", "overhead", "base"),
            camera_width=32,
            camera_height=32,
            task="keys in the cup",
            episode_count=0,
            use_depth=True,
            depth_camera_name="wrist",
            depth_camera_names=("wrist", "base"),
        ),
    )
    w = EpisodeWriter(ds, 0, source="replay_capture")
    for k in range(3):
        w.add_frame(
            np.zeros(6, np.float32),
            np.ones(6, np.float32),
            k / 15.0,
            images={
                "wrist": np.zeros((32, 32, 3), np.uint8),
                "overhead": np.zeros((32, 32, 3), np.uint8),
                "base": np.zeros((32, 32, 3), np.uint8),
            },
            depths={
                "wrist": np.full((32, 32), 111, np.uint16),
                "base": np.full((32, 32), 222, np.uint16),
            },
        )
    w.save()
    ep = SO101Dataset(tmp_path).load_episode(0)
    assert ep.depth_paths is not None
    assert set(ep.depth_paths) == {"wrist", "base"}
    assert int(ep.load_depth("base")[0, 0, 0]) == 222
    assert Meta.from_dict(ds.meta.to_dict()).depth_names() == ("wrist", "base")


def test_replay_capture_episode_writes_frames(tmp_path: Path) -> None:
    ds = SO101Dataset.create(
        tmp_path,
        DatasetMeta(
            name="keys",
            fps=15,
            joint_names=list(JOINT_NAMES),
            camera_name="wrist",
            camera_names=("wrist", "overhead", "base"),
            camera_width=32,
            camera_height=32,
            task="keys in the cup",
            episode_count=0,
            use_depth=True,
            depth_camera_name="wrist",
            depth_camera_names=("wrist", "base"),
        ),
    )
    writer = EpisodeWriter(ds, 0, source="replay_capture")
    n = replay_capture_episode(
        follower=FakeFollower(np.zeros(6, np.float32)),
        src_episode=SrcEpisode(4),
        writer=writer,
        camera_names=["wrist", "overhead", "base"],
        depth_names=["wrist", "base"],
        fps=15,
        depth_source=FakeZMQ(),
        sleep=lambda _s: None,
    )
    assert n == 4
    writer.save()
    ep = SO101Dataset(tmp_path).load_episode(0)
    assert ep.num_frames == 4
    assert ep.source == "replay_capture"
