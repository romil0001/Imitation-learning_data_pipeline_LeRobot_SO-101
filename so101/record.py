"""Teleop record loop: leader actions, follower state + RGB, custom dataset."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from so101.dataset import JOINT_NAMES, EpisodeWriter, SO101Dataset
from so101.hardware import action_dict_from_qpos, qpos_from_obs, rgb_from_obs
from so101.keys import KeySource


def _busy_wait(elapsed: float, fps: int, sleep: Callable[[float], None]) -> None:
    leftover = (1.0 / fps) - elapsed
    if leftover > 0:
        sleep(leftover)


def run_record_loop(
    *,
    dataset: SO101Dataset,
    follower: Any,
    leader: Any,
    camera_name: str | None = None,
    camera_names: list[str] | None = None,
    fps: int,
    keys: KeySource,
    num_episodes: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Record until num_episodes are saved (or until q). Returns saved count."""
    names = list(camera_names) if camera_names else [str(camera_name)]
    saved = 0
    phase = "record"
    index = dataset.next_episode_index()
    writer = EpisodeWriter(dataset, index)
    print(f"Recording episode {index:06d}  n=save  r=redo  q=quit", flush=True)

    while True:
        t0 = time.perf_counter()
        obs = follower.get_observation()
        action = leader.get_action()
        sent = follower.send_action(action)
        if phase == "record":
            writer.add_frame(
                state=qpos_from_obs(obs),
                action=qpos_from_obs(sent if sent is not None else action),
                timestamp=time.time(),
                images={name: rgb_from_obs(obs, name) for name in names},
            )
        key = keys.poll()
        if phase == "record":
            if key == "n":
                writer.save()
                saved += 1
                print(f"Saved episode {index:06d} ({saved} total)", flush=True)
                if num_episodes is not None and saved >= num_episodes:
                    break
                phase = "reset"
                print("Reset the scene, then press n for the next episode (q to stop)", flush=True)
            elif key == "r":
                writer.abort()
                writer = EpisodeWriter(dataset, index)
                print(f"Redo episode {index:06d}", flush=True)
            elif key == "q":
                writer.abort()
                print("Quit without saving the current episode", flush=True)
                break
        else:
            if key == "n":
                index = dataset.next_episode_index()
                writer = EpisodeWriter(dataset, index)
                phase = "record"
                print(f"Recording episode {index:06d}  n=save  r=redo  q=quit", flush=True)
            elif key == "q":
                print("Stopped after reset", flush=True)
                break
        _busy_wait(time.perf_counter() - t0, fps, sleep)

    return saved


def record_from_env(
    *,
    dataset_name: str = "so101_d405_demo",
    num_episodes: int = 50,
    task: str = "Pick pen and place it next to the keys",
    confirm: bool = True,
) -> int:
    from so101.config import load_cameras, load_ports, project_root
    from so101.dataset import DatasetMeta, compute_stats, write_stats
    from so101.hardware import make_follower, make_leader
    from so101.keys import StdinKeys

    ports = load_ports()
    cameras = load_cameras()
    cam0 = cameras[0]
    names = [c.name for c in cameras]
    root = project_root() / "data" / "datasets" / dataset_name
    if (root / "meta.json").is_file():
        dataset = SO101Dataset(root)
        print(f"Resuming {root} with {dataset.meta.episode_count} episodes", flush=True)
    else:
        dataset = SO101Dataset.create(
            root,
            DatasetMeta(
                name=dataset_name,
                fps=cam0.fps,
                joint_names=list(JOINT_NAMES),
                camera_name=names[0],
                camera_names=tuple(names),
                camera_width=cam0.width,
                camera_height=cam0.height,
                task=task,
                episode_count=0,
            ),
        )
    remaining = num_episodes - dataset.meta.episode_count
    if remaining < 1:
        print(f"Already have {dataset.meta.episode_count} episodes (target {num_episodes})")
        return 0

    print("BOTH ARMS: follower + leader USB and 12V ON. Omen publisher must be running.")
    print(f"Dataset {root}  this_run={remaining}  target={num_episodes}  task={task!r}")
    print(f"Cameras: {', '.join(names)}  {cam0.width}x{cam0.height}@{cam0.fps}")
    if confirm:
        input("Confirm hardware, then press Enter (Ctrl-C to abort) ")

    follower = make_follower(ports, cameras, with_cameras=True)
    leader = make_leader(ports)
    follower.connect()
    leader.connect()
    try:
        with StdinKeys() as keys:
            saved = run_record_loop(
                dataset=dataset,
                follower=follower,
                leader=leader,
                camera_names=names,
                fps=cam0.fps,
                keys=keys,
                num_episodes=remaining,
            )
    finally:
        leader.disconnect()
        follower.disconnect()

    if dataset.meta.episode_count > 0:
        write_stats(root, compute_stats(SO101Dataset(root)))
        print(f"Wrote stats.json  episodes={dataset.meta.episode_count}")
    return saved
