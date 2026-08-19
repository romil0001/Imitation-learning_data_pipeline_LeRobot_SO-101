"""Replay a saved episode's actions on the follower (no leader, no camera)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from so101.hardware import action_dict_from_qpos


def replay_episode(
    follower: Any,
    episode: Any,
    fps: int,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    for i in range(episode.num_frames):
        t0 = time.perf_counter()
        follower.send_action(action_dict_from_qpos(episode.action[i]))
        leftover = (1.0 / fps) - (time.perf_counter() - t0)
        if leftover > 0:
            sleep(leftover)


def replay_from_env(*, dataset_name: str = "so101_d405_demo", episode: int = 0, confirm: bool = True) -> None:
    from so101.config import load_ports, project_root
    from so101.dataset import SO101Dataset
    from so101.hardware import make_follower

    ports = load_ports()
    root = project_root() / "data" / "datasets" / dataset_name
    ds = SO101Dataset(root)
    if episode < 0 or episode >= ds.meta.episode_count:
        raise IndexError(f"Episode {episode} does not exist (count={ds.meta.episode_count})")
    ep = ds.load_episode(episode)
    print("FOLLOWER ONLY: USB + 12V ON. Leader can be OFF. Clear the workspace.")
    print(f"Replay episode {episode}  frames={ep.num_frames}  fps={ds.meta.fps}")
    if confirm:
        input("Confirm hardware, then press Enter (Ctrl-C to abort) ")
    follower = make_follower(ports, camera=None, with_cameras=False)
    follower.connect()
    try:
        replay_episode(follower, ep, ds.meta.fps)
    finally:
        follower.disconnect()
