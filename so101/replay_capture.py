"""Replay saved actions while capturing live multi-camera (+ depth) into a new dataset."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from so101.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset, compute_stats, write_stats
from so101.hardware import action_dict_from_qpos, qpos_from_obs, rgbs_from_obs


def _wait_rgbs(
    depth_source: Any,
    names: Sequence[str],
    *,
    timeout_s: float = 2.0,
) -> dict[str, np.ndarray]:
    """Prefer ZMQ RGB (all cams including base); fall back not used here."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        latest = getattr(depth_source, "latest_images", None)
        if callable(latest):
            images = latest()
            if images and all(n in images for n in names):
                return {n: images[n] for n in names}
        # LatestZMQRGBD keeps jpegs; decode via public helpers if present.
        get_rgb = getattr(depth_source, "latest_rgb", None)
        if callable(get_rgb):
            images = {}
            ok = True
            for n in names:
                frame = get_rgb(n)
                if frame is None:
                    ok = False
                    break
                images[n] = frame
            if ok:
                return images
        time.sleep(0.01)
    raise RuntimeError(f"Timed out waiting for RGB cameras {list(names)}")


def replay_capture_episode(
    *,
    follower: Any,
    src_episode: Any,
    writer: EpisodeWriter,
    camera_names: list[str],
    depth_names: list[str],
    fps: int,
    depth_source: Any,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.perf_counter,
) -> int:
    """Play ``src_episode.action`` and record live cameras/depth into ``writer``."""
    n = int(src_episode.num_frames)
    for i in range(n):
        t0 = now()
        action = np.asarray(src_episode.action[i], dtype=np.float32)
        follower.send_action(action_dict_from_qpos(action))
        obs = follower.get_observation()
        state = qpos_from_obs(obs)
        # RGB: ZMQ first (base/overhead/wrist), else follower OpenCV/ZMQ cams.
        try:
            images = _wait_rgbs(depth_source, camera_names, timeout_s=1.5)
        except RuntimeError:
            try:
                frames = rgbs_from_obs(obs, camera_names)
                images = {n: frames[i] for i, n in enumerate(camera_names)}
            except Exception as exc:
                raise RuntimeError(
                    f"Missing live cameras {list(camera_names)}. "
                    f"Is the Omen publishing CAMERA_NAMES={camera_names}?"
                ) from exc
        depths: dict[str, np.ndarray] = {}
        for name in depth_names:
            z = depth_source.latest_depth(name)
            if z is None:
                z = depth_source.wait_for_depth(name, timeout_s=2.0)
            if z is None:
                raise RuntimeError(f"No depth frame for {name!r}")
            depths[name] = z
        writer.add_frame(
            state,
            action,
            timestamp=float(i) / float(fps),
            images=images,
            depths=depths if depths else None,
        )
        leftover = (1.0 / fps) - (now() - t0)
        if leftover > 0:
            sleep(leftover)
    return writer.num_frames


def replay_capture_from_env(
    *,
    source_dataset: str = "so101_d405_demo",
    dest_dataset: str = "so101_keys_in_cup",
    task: str = "keys in the cup",
    episode: int | None = None,
    max_episodes: int | None = None,
    confirm: bool = True,
) -> int:
    """Replay source episode(s) and write a new multi-cam dataset."""
    from so101.config import load_cameras, load_ports, project_root
    from so101.hardware import make_follower
    from so101.home import interpolate_to_qpos, safe_power_down
    from so101.zmq_rgbd import LatestZMQRGBD

    root = project_root()
    ports = load_ports()
    cameras = load_cameras()
    cam0 = cameras[0]
    names = [c.name for c in cameras]
    if cam0.backend != "zmq":
        raise RuntimeError("replay-capture requires CAMERA_BACKEND=zmq (Omen publisher)")
    use_depth = bool(cam0.use_depth)
    # Dual D405 depths when base is in CAMERA_NAMES.
    depth_names = [n for n in names if n in {"wrist", "base"}] if use_depth else []
    if use_depth and not depth_names:
        depth_names = ["wrist"]

    src_root = root / "data" / "datasets" / source_dataset
    dest_root = root / "data" / "datasets" / dest_dataset
    src = SO101Dataset(src_root)
    if src.meta.episode_count < 1:
        raise ValueError(f"Source dataset {src_root} has no episodes")

    if dest_root.is_dir() and (dest_root / "meta.json").is_file():
        dest = SO101Dataset(dest_root)
        if list(dest.meta.names()) != names:
            raise ValueError(
                f"Dest dataset cameras={list(dest.meta.names())} != camera.env {names}. "
                f"Use a fresh dest name or delete {dest_root}."
            )
    else:
        dest = SO101Dataset.create(
            dest_root,
            DatasetMeta(
                name=dest_dataset,
                fps=cam0.fps,
                joint_names=list(JOINT_NAMES),
                camera_name=names[0],
                camera_names=tuple(names),
                camera_width=cam0.width,
                camera_height=cam0.height,
                task=task,
                episode_count=0,
                use_depth=use_depth,
                depth_camera_name=depth_names[0] if depth_names else "wrist",
                depth_camera_names=tuple(depth_names),
            ),
        )

    if episode is not None:
        indices = [int(episode)]
    else:
        n = src.meta.episode_count
        if max_episodes is not None:
            n = min(n, int(max_episodes))
        indices = list(range(n))

    print("REPLAY-CAPTURE: FOLLOWER USB + 12V ON. Leader OFF. Omen must stream all cams + depth.")
    print(f"Source: {src_root}  episodes={indices}")
    print(f"Dest:   {dest_root}  task={task!r}  cameras={names}  depths={depth_names}")
    if confirm:
        input("Clear workspace, then press Enter (Ctrl-C to abort) ")

    follower = make_follower(ports, cameras, with_cameras=True)
    follower.connect()
    depth_source = LatestZMQRGBD(cam0.zmq_host, cam0.zmq_port)
    rest_pose = None
    saved = 0
    try:
        for name in depth_names:
            print(f"Waiting for depth {name!r} …", flush=True)
            depth_source.wait_for_depth(name, timeout_s=20.0)
        for ep_i in indices:
            ep = src.load_episode(ep_i)
            rest_pose = ep.state[0]
            print(f"Homing to episode {ep_i} start …", flush=True)
            interpolate_to_qpos(follower, ep.state[0], n_steps=60, dt=0.04, sleep=time.sleep)
            out_i = dest.next_episode_index()
            from so101.dataset.tags import replay_capture_tags

            writer = EpisodeWriter(
                dest,
                out_i,
                source="replay_capture",
                intervention=False,
                tags=replay_capture_tags(),
            )
            print(f"Replaying src={ep_i} → dest={out_i} frames={ep.num_frames}", flush=True)
            try:
                n = replay_capture_episode(
                    follower=follower,
                    src_episode=ep,
                    writer=writer,
                    camera_names=names,
                    depth_names=depth_names,
                    fps=cam0.fps,
                    depth_source=depth_source,
                )
                writer.save()
                saved += 1
                print(f"Saved {out_i:06d}  frames={n}", flush=True)
            except Exception:
                writer.abort()
                raise
    except KeyboardInterrupt:
        print("\nCtrl-C — parking before disconnect.", flush=True)
        raise SystemExit(130) from None
    finally:
        try:
            safe_power_down(follower, rest_pose, sleep=time.sleep)
        finally:
            depth_source.close()
            try:
                follower.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"disconnect: {exc}", flush=True)

    if dest.meta.episode_count > 0:
        write_stats(dest_root, compute_stats(SO101Dataset(dest_root)))
        print(f"Wrote {dest_root}/stats.json  episodes={dest.meta.episode_count}")
    return saved
