"""Human-gated DAgger on the SO-101 follower.

Policy drives until Pause. Pause moves the leader onto the follower, then
releases torque. Correction frames (leader actions on follower-visited states)
are the only frames written. Autonomous policy frames are not treated as expert
actions.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from so101.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset
from so101.hardware import action_dict_from_qpos, qpos_from_obs, rgb_from_obs
from so101.keys import KeySource
from so101.leader_sync import sync_leader_to_follower


def _busy_wait(elapsed: float, fps: int, sleep: Callable[[float], None]) -> None:
    leftover = (1.0 / fps) - elapsed
    if leftover > 0:
        sleep(leftover)


def _normalize_key(key: str | None) -> str | None:
    if key == " ":
        return "p"
    return key


def run_dagger_loop(
    *,
    dataset: SO101Dataset,
    follower: Any,
    leader: Any,
    policy: Any,
    camera_names: list[str],
    fps: int,
    keys: KeySource,
    num_episodes: int | None = None,
    depth_source: Any | None = None,
    depth_camera_name: str = "wrist",
    sync_steps: int = 40,
    sync_dt: float = 0.04,
    sleep: Callable[[float], None] = time.sleep,
    episode_tags: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Collect correction episodes. Returns how many were saved.

    Keys: p pause/sync or resume policy, c start recording the correction,
    n save, r discard the open correction, q quit.
    """
    names = list(camera_names)
    phase = "policy"
    saved = 0
    writer: EpisodeWriter | None = None
    hold_q: np.ndarray | None = None
    tags = list(episode_tags or ())
    policy.reset()
    print(
        "HG-DAgger  p=pause (leader moves to follower)  c=record correction  "
        "p=resume policy  n=save  r=discard  q=quit",
        flush=True,
    )
    print("Policy is driving. Leader stays where it is until you press p.", flush=True)

    while True:
        t0 = time.perf_counter()
        key = _normalize_key(keys.poll())
        if key == "q":
            if writer is not None and not writer.is_saved:
                writer.abort()
            print("Quit", flush=True)
            break
        if key == "r":
            if writer is not None:
                writer.abort()
                writer = None
                print("Discarded the open correction", flush=True)
            phase = "policy"
            hold_q = None
            policy.reset()
            print("Policy is driving.", flush=True)
        elif key == "n":
            if writer is None or writer.num_frames < 1:
                print("Nothing to save — press p, then c, and move the leader first", flush=True)
            else:
                writer.save()
                saved += 1
                print(f"Saved correction {writer.index:06d} ({saved} this session)", flush=True)
                writer = None
                phase = "policy"
                hold_q = None
                policy.reset()
                if num_episodes is not None and saved >= num_episodes:
                    break
                print("Policy is driving. Press p to correct again.", flush=True)
        elif key == "p" and phase == "policy":
            print("Pause: holding follower and moving leader to that pose…", flush=True)
            hold_q = sync_leader_to_follower(
                leader, follower, n_steps=sync_steps, dt=sync_dt, sleep=sleep
            )
            phase = "handoff"
            print(
                "Leader matches the follower and torque is off. Grab it, press c to record, "
                "or p to give control back to the policy.",
                flush=True,
            )
        elif key == "p" and phase in {"handoff", "correct"}:
            phase = "policy"
            hold_q = None
            policy.reset()
            print("Policy resumed. Unsaved correction frames stay until n or r.", flush=True)
        elif key == "c" and phase == "handoff":
            if writer is None:
                writer = EpisodeWriter(
                    dataset,
                    dataset.next_episode_index(),
                    source="dagger",
                    intervention=True,
                    tags=tags,
                )
            phase = "correct"
            print("Recording correction. n=save  r=discard  p=resume policy", flush=True)
        elif key == "c" and phase == "policy":
            print("Press p first so the leader moves to the follower before you correct.", flush=True)

        if phase == "policy":
            obs = follower.get_observation()
            images = [rgb_from_obs(obs, name) for name in names]
            image = images[0] if len(images) == 1 else images
            depth_u16 = None
            use_depth_policy = bool(getattr(getattr(policy, "cfg", None), "use_depth", False))
            if use_depth_policy:
                if depth_source is None:
                    raise RuntimeError("depth_source required when policy uses depth")
                depth_u16 = depth_source.latest_depth(depth_camera_name)
                if depth_u16 is None:
                    raise RuntimeError(f"No depth frame yet for {depth_camera_name!r}")
            qpos = qpos_from_obs(obs)
            action = policy.select_action(image, qpos, depth_u16=depth_u16)
            follower.send_action(action_dict_from_qpos(np.asarray(action, dtype=np.float32)))
            try:
                from so101.live_viz import publish_live_joints

                publish_live_joints(qpos, action)
            except Exception:
                pass
        elif phase == "handoff":
            if hold_q is None:
                raise RuntimeError("handoff without a held pose")
            follower.send_action(action_dict_from_qpos(hold_q))
        else:
            if writer is None:
                raise RuntimeError("correct phase without a writer")
            obs = follower.get_observation()
            action = leader.get_action()
            sent = follower.send_action(action)
            depth_u16 = None
            if dataset.meta.use_depth:
                if depth_source is None:
                    raise RuntimeError("depth_source required when recording with use_depth")
                depth_u16 = depth_source.latest_depth(depth_camera_name)
                if depth_u16 is None:
                    depth_u16 = depth_source.wait_for_depth(depth_camera_name, timeout_s=2.0)
                if depth_u16 is None:
                    raise RuntimeError(f"No depth frame yet for {depth_camera_name!r}")
            state = qpos_from_obs(obs)
            act = qpos_from_obs(sent if sent is not None else action)
            writer.add_frame(
                state=state,
                action=act,
                timestamp=time.time(),
                images={name: rgb_from_obs(obs, name) for name in names},
                depth_u16=depth_u16,
            )
            try:
                from so101.live_viz import publish_live_joints

                publish_live_joints(state, act)
            except Exception:
                pass
        _busy_wait(time.perf_counter() - t0, fps, sleep)

    return saved


def _open_corrections_dataset(
    *,
    base: SO101Dataset,
    root: Path,
    name: str,
) -> SO101Dataset:
    fresh = DatasetMeta(
        name=name,
        fps=base.meta.fps,
        joint_names=list(JOINT_NAMES),
        camera_name=base.meta.camera_name,
        camera_names=base.meta.names(),
        camera_width=base.meta.camera_width,
        camera_height=base.meta.camera_height,
        task=base.meta.task,
        episode_count=0,
        use_depth=base.meta.use_depth,
        depth_camera_name=base.meta.depth_camera_name,
    )
    if (root / "meta.json").is_file():
        dataset = SO101Dataset(root)
        if dataset.meta.episode_count == 0:
            return SO101Dataset.create(root, fresh)
        if dataset.meta.names() != fresh.names() or bool(dataset.meta.use_depth) != bool(fresh.use_depth):
            raise ValueError(
                f"Corrections at {root} cameras={list(dataset.meta.names())} "
                f"use_depth={dataset.meta.use_depth} do not match the base dataset."
            )
        print(f"Resuming corrections {root} with {dataset.meta.episode_count} episodes", flush=True)
        return dataset
    return SO101Dataset.create(root, fresh)


def dagger_from_env(
    *,
    dataset_name: str = "so101_d405_demo",
    corrections_name: str = "",
    checkpoint: str = "",
    num_episodes: int = 10,
    confirm: bool = True,
    policy: str = "act",
    policy_base: str = "smolvla",
    grasp_config: str = "",
    task: str = "",
    sync_steps: int = 40,
    sync_dt: float = 0.04,
) -> int:
    from so101.config import load_cameras, load_ports, project_root
    from so101.dataset import compute_stats, write_stats
    from so101.eval import _dataset_home_qpos, _load_eval_policy
    from so101.hardware import make_follower, make_leader
    from so101.home import interpolate_to_qpos, safe_power_down
    from so101.keys import OperatorKeys
    from so101.train import detect_device

    root = project_root()
    ports = load_ports()
    cameras = load_cameras()
    cam0 = cameras[0]
    names = [c.name for c in cameras]
    base_root = root / "data" / "datasets" / dataset_name
    base = SO101Dataset(base_root)
    corr_name = corrections_name.strip() or f"{dataset_name}_dagger"
    corr_root = root / "data" / "datasets" / corr_name
    dataset = _open_corrections_dataset(base=base, root=corr_root, name=corr_name)

    device = detect_device()
    task_text = task.strip() or base.meta.task
    policy_obj = _load_eval_policy(
        policy_kind=policy,
        policy_base=policy_base,
        checkpoint=checkpoint,
        dataset_name=dataset_name,
        root=root,
        device=device,
        task=task_text,
        camera_names=names,
        image_height=cam0.height,
        image_width=cam0.width,
        grasp_config=grasp_config,
    )
    use_depth = bool(dataset.meta.use_depth) or bool(
        getattr(getattr(policy_obj, "cfg", None), "use_depth", False)
    )
    depth_name = dataset.meta.depth_camera_name
    print("HG-DAgger: BOTH arms USB + 12V ON. Omen publisher must be streaming.")
    print(f"Base demos: {base_root}  corrections: {corr_root}")
    print(f"Policy: {policy}" + (f" base={policy_base}" if policy == "hybrid" else ""))
    print("Leader does not move until you press p.")
    if confirm:
        input("Confirm hardware, then press Enter (Ctrl-C to abort) ")

    follower = make_follower(ports, cameras, with_cameras=True)
    leader = make_leader(ports)
    follower.connect()
    leader.connect()
    depth_source = None
    rest_pose = None
    if use_depth:
        from so101.zmq_rgbd import LatestZMQRGBD

        depth_source = LatestZMQRGBD(cam0.zmq_host, cam0.zmq_port)
        depth_source.wait_for_depth(depth_name, timeout_s=20.0)
    try:
        home = _dataset_home_qpos(root, dataset_name)
        rest_pose = home
        print(f"Moving follower to demo start pose {list(home)}", flush=True)
        interpolate_to_qpos(follower, home, n_steps=80, dt=0.04, sleep=time.sleep)
        from so101.dataset.tags import dagger_tags

        with OperatorKeys(allowed={"p", "c", "n", "r", "q", " "}) as op_keys:
            saved = run_dagger_loop(
                dataset=dataset,
                follower=follower,
                leader=leader,
                policy=policy_obj,
                camera_names=names,
                fps=cam0.fps,
                keys=op_keys,
                num_episodes=num_episodes,
                depth_source=depth_source,
                depth_camera_name=depth_name,
                sync_steps=sync_steps,
                sync_dt=sync_dt,
                episode_tags=dagger_tags(policy, policy_base),
            )
    except KeyboardInterrupt:
        print("\nCtrl-C — parking follower before disconnect.", flush=True)
        raise SystemExit(130) from None
    finally:
        try:
            safe_power_down(follower, rest_pose, sleep=time.sleep)
        finally:
            if depth_source is not None:
                depth_source.close()
            try:
                leader.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"leader disconnect: {exc}", flush=True)
            try:
                follower.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"follower disconnect: {exc}", flush=True)

    if dataset.meta.episode_count > 0:
        write_stats(corr_root, compute_stats(SO101Dataset(corr_root)))
        print(f"Wrote {corr_root}/stats.json  episodes={dataset.meta.episode_count}")
    return saved
