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


def _resolve_record_home(dataset: SO101Dataset) -> np.ndarray | None:
    """Prefer protocol home_qpos, else mean of episode start poses."""
    from so101.config import project_root
    from so101.home import mean_start_qpos

    proto = project_root() / "config" / "eval" / "bag_clip_in_bowl.yaml"
    if proto.is_file():
        try:
            from so101.eval_protocol import load_protocol

            loaded = load_protocol(proto)
            if loaded.home_qpos:
                return np.asarray(loaded.home_qpos, dtype=np.float32)
        except Exception:
            pass
    if dataset.meta.episode_count > 0:
        try:
            return mean_start_qpos(dataset)
        except Exception:
            return None
    return None


def _park_follower_home(
    follower: Any,
    home_qpos: np.ndarray | None,
    *,
    sleep: Callable[[float], None],
) -> None:
    if home_qpos is None:
        print("No home pose available — holding current joints until cameras recover.", flush=True)
        return
    from so101.home import interpolate_to_qpos

    print("Stale camera frame — slowly returning follower to home…", flush=True)
    try:
        interpolate_to_qpos(follower, home_qpos, n_steps=80, dt=0.04, sleep=sleep)
        print("Follower at home. Waiting for fresh ZMQ frames…", flush=True)
    except Exception as exc:  # noqa: BLE001 — keep session alive
        print(f"Home park failed ({exc}); waiting for cameras anyway.", flush=True)


def _wait_for_observation(
    follower: Any,
    *,
    sleep: Callable[[float], None],
    home_qpos: np.ndarray | None,
    retries: int = 120,
    pause_s: float = 0.05,
) -> tuple[Any, bool]:
    """Return ``(obs, did_park)``. Parks to home once on the first stale frame."""
    parked = False
    last: BaseException | None = None
    for attempt in range(max(1, retries)):
        try:
            return follower.get_observation(), parked
        except TimeoutError as e:
            last = e
            if not parked:
                _park_follower_home(follower, home_qpos, sleep=sleep)
                parked = True
            if attempt + 1 >= retries:
                break
            if (attempt + 1) % 20 == 0:
                print(
                    f"Still waiting for fresh ZMQ ({e}) — retry {attempt + 1}/{retries}",
                    flush=True,
                )
            sleep(pause_s)
    assert last is not None
    raise last


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
    depth_source: Any | None = None,
    depth_camera_name: str = "wrist",
    depth_camera_names: list[str] | None = None,
    home_qpos: np.ndarray | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Record until num_episodes are saved (or until q). Returns saved count."""
    names = list(camera_names) if camera_names else [str(camera_name)]
    depth_names = list(depth_camera_names) if depth_camera_names else (
        list(dataset.meta.depth_names()) if dataset.meta.use_depth else []
    )
    if not depth_names and dataset.meta.use_depth:
        depth_names = [depth_camera_name]
    if home_qpos is None:
        home_qpos = _resolve_record_home(dataset)
    saved = 0
    phase = "record"
    index = dataset.next_episode_index()
    from so101.dataset.tags import collection_tags

    tags = collection_tags()
    writer = EpisodeWriter(dataset, index, tags=tags)
    print(f"Recording episode {index:06d}  n=save  r=redo  q=quit", flush=True)

    while True:
        t0 = time.perf_counter()
        try:
            obs, did_park = _wait_for_observation(follower, sleep=sleep, home_qpos=home_qpos)
        except TimeoutError:
            if phase == "record":
                writer.abort()
                print(
                    f"Quit: cameras stayed stale; discarded unfinished episode {index:06d}",
                    flush=True,
                )
            else:
                print("Quit: cameras stayed stale during reset.", flush=True)
            break
        if did_park and phase == "record":
            # Drop any frames captured before the stall; restart the same index.
            writer.abort()
            writer = EpisodeWriter(dataset, index, tags=tags)
            print(
                f"Camera stall — discarded in-progress episode {index:06d}; "
                "recording restarts when you continue teleop (n=save r=redo q=quit)",
                flush=True,
            )
        action = leader.get_action()
        sent = follower.send_action(action)
        depths: dict[str, np.ndarray] | None = None
        if dataset.meta.use_depth and depth_source is not None:
            depths = {}
            for dname in depth_names:
                z = depth_source.latest_depth(dname)
                if z is None and phase == "record":
                    z = depth_source.wait_for_depth(dname, timeout_s=2.0)
                if z is not None:
                    depths[dname] = z
        if phase == "record":
            if dataset.meta.use_depth:
                if depth_source is None:
                    raise RuntimeError("depth_source required when recording with use_depth")
                if depths is None or any(n not in depths for n in depth_names):
                    raise RuntimeError(f"No depth frame yet for {depth_names!r}")
            writer.add_frame(
                state=qpos_from_obs(obs),
                action=qpos_from_obs(sent if sent is not None else action),
                timestamp=time.time(),
                images={name: rgb_from_obs(obs, name) for name in names},
                depths=depths,
            )
            from so101.live_viz import publish_live_joints

            publish_live_joints(
                qpos_from_obs(obs),
                qpos_from_obs(sent if sent is not None else action),
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
                writer = EpisodeWriter(dataset, index, tags=tags)
                print(f"Redo episode {index:06d}", flush=True)
            elif key == "q":
                writer.abort()
                print("Quit without saving the current episode", flush=True)
                break
        else:
            if key == "n":
                index = dataset.next_episode_index()
                writer = EpisodeWriter(dataset, index, tags=tags)
                phase = "record"
                print(f"Recording episode {index:06d}  n=save  r=redo  q=quit", flush=True)
            elif key == "q":
                print("Stopped after reset", flush=True)
                break
        _busy_wait(time.perf_counter() - t0, fps, sleep)

    return saved


def record_from_env(
    *,
    dataset_name: str = "so101_bag_clip_bowl",
    num_episodes: int = 50,
    task: str = "Place bag clip in bowl",
    confirm: bool = True,
) -> int:
    from so101.config import load_cameras, load_ports, project_root
    from so101.dataset import DatasetMeta, compute_stats, write_stats
    from so101.hardware import make_follower, make_leader
    from so101.home import safe_power_down
    from so101.keys import OperatorKeys

    ports = load_ports()
    cameras = load_cameras()
    cam0 = cameras[0]
    names = [c.name for c in cameras]
    use_depth = cam0.use_depth and cam0.backend == "zmq"
    depth_names = [n for n in names if n in {"wrist", "base"}] if use_depth else []
    if use_depth and not depth_names:
        depth_names = [names[0]]
    depth_name = depth_names[0] if depth_names else "wrist"
    root = project_root() / "data" / "datasets" / dataset_name
    fresh_meta = DatasetMeta(
        name=dataset_name,
        fps=cam0.fps,
        joint_names=list(JOINT_NAMES),
        camera_name=names[0],
        camera_names=tuple(names),
        camera_width=cam0.width,
        camera_height=cam0.height,
        task=task,
        episode_count=0,
        use_depth=use_depth,
        depth_camera_name=depth_name,
        depth_camera_names=tuple(depth_names),
    )
    if (root / "meta.json").is_file():
        dataset = SO101Dataset(root)
        if dataset.meta.episode_count == 0:
            dataset = SO101Dataset.create(root, fresh_meta)
            print(f"Reset empty dataset meta at {root} → cameras={list(names)} depth={use_depth}", flush=True)
        else:
            stored = list(dataset.meta.names())
            if stored != names or bool(dataset.meta.use_depth) != bool(use_depth):
                raise ValueError(
                    f"Dataset at {root} has cameras={stored} use_depth={dataset.meta.use_depth} "
                    f"but camera.env has cameras={names} use_depth={use_depth}. "
                    "Delete the dataset folder or pick a new --dataset name before changing cameras."
                )
            use_depth = dataset.meta.use_depth
            depth_name = dataset.meta.depth_camera_name
            depth_names = list(dataset.meta.depth_names())
            names = stored
            print(f"Resuming {root} with {dataset.meta.episode_count} episodes", flush=True)
    else:
        dataset = SO101Dataset.create(root, fresh_meta)
    remaining = num_episodes - dataset.meta.episode_count
    if remaining < 1:
        print(f"Already have {dataset.meta.episode_count} episodes (target {num_episodes})")
        return 0

    print("BOTH ARMS: follower + leader USB and 12V ON. Omen publisher must be running.")
    print(f"Dataset {root}  this_run={remaining}  target={num_episodes}  task={task!r}")
    print(f"Cameras: {', '.join(names)}  {cam0.width}x{cam0.height}@{cam0.fps}")
    if use_depth:
        print(f"Depth: {depth_names} uint16 from ZMQ (Omen publisher must not use --no-depth)")
    print("Live preview: dashboard http://127.0.0.1:8080  (cameras, depth, joint graphs)", flush=True)
    print("Keys n/r/q work in this terminal or on the dashboard Keys panel.", flush=True)
    if confirm:
        input("Confirm hardware, then press Enter (Ctrl-C to abort) ")

    home_qpos = _resolve_record_home(dataset)
    if home_qpos is not None:
        print(f"Stale-cam home pose ready ({len(home_qpos)} joints)", flush=True)

    follower = make_follower(ports, cameras, with_cameras=True)
    leader = make_leader(ports)
    follower.connect()
    leader.connect()
    depth_source = None
    if use_depth:
        from so101.zmq_rgbd import LatestZMQRGBD

        depth_source = LatestZMQRGBD(cam0.zmq_host, cam0.zmq_port)
        for dname in depth_names:
            print(f"Waiting for depth {dname!r} from {cam0.zmq_host}:{cam0.zmq_port} ...", flush=True)
            depth_source.wait_for_depth(dname, timeout_s=20.0)
        print("Depth stream OK", flush=True)
    try:
        with OperatorKeys() as keys:
            saved = run_record_loop(
                dataset=dataset,
                follower=follower,
                leader=leader,
                camera_names=names,
                fps=cam0.fps,
                keys=keys,
                num_episodes=remaining,
                depth_source=depth_source,
                depth_camera_name=depth_name,
                depth_camera_names=depth_names,
                home_qpos=home_qpos,
            )
    finally:
        if depth_source is not None:
            depth_source.close()
        leader.disconnect()
        safe_power_down(follower, home_qpos, sleep=time.sleep)
        follower.disconnect()

    if dataset.meta.episode_count > 0:
        write_stats(root, compute_stats(SO101Dataset(root)))
        print(f"Wrote stats.json  episodes={dataset.meta.episode_count}")
    return saved
