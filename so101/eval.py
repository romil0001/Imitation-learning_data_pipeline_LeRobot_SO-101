"""Run a trained ACT policy on the follower. Omen D405 publisher must be up."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from so101.hardware import action_dict_from_qpos, qpos_from_obs, rgbs_from_obs


def _dataset_home_qpos(root: Path, dataset_name: str) -> tuple[float, ...]:
    from so101.dataset import SO101Dataset
    from so101.home import mean_start_qpos

    ds = SO101Dataset(root / "data" / "datasets" / dataset_name)
    return tuple(float(x) for x in mean_start_qpos(ds))


def run_eval_loop(
    *,
    follower: Any,
    policy: Any,
    camera_name: str | None = None,
    camera_names: list[str] | None = None,
    fps: int,
    duration_s: float,
    max_steps: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.perf_counter,
    should_stop: Callable[[], bool] | None = None,
    depth_source: Any | None = None,
    depth_camera_name: str = "wrist",
) -> int:
    names = list(camera_names) if camera_names else [str(camera_name)]
    policy.reset()
    t0 = now()
    steps = 0
    use_depth = bool(getattr(getattr(policy, "cfg", None), "use_depth", False))
    while True:
        if max_steps is not None and steps >= max_steps:
            break
        if now() - t0 >= duration_s:
            break
        if should_stop is not None and should_stop():
            break
        loop_t = now()
        obs = follower.get_observation()
        images = rgbs_from_obs(obs, names)
        image = images[0] if len(images) == 1 else images
        depth_u16 = None
        if use_depth:
            if depth_source is None:
                raise RuntimeError("depth_source required when policy uses depth")
            depth_u16 = depth_source.latest_depth(depth_camera_name)
            if depth_u16 is None:
                raise RuntimeError(f"No depth frame yet for {depth_camera_name!r}")
        action = policy.select_action(image, qpos_from_obs(obs), depth_u16=depth_u16)
        follower.send_action(action_dict_from_qpos(action))
        steps += 1
        leftover = (1.0 / fps) - (now() - loop_t)
        if leftover > 0:
            sleep(leftover)
    return steps


def eval_from_env(
    *,
    checkpoint: str = "",
    dataset_name: str = "so101_d405_demo",
    duration_s: float = 60.0,
    confirm: bool = True,
    protocol: str = "",
    n_trials: int | None = None,
    keys: Any | None = None,
    reset_confirm: Callable[..., None] | None = None,
    output_dir: Path | None = None,
) -> int:
    from dataclasses import replace

    from so101.config import load_cameras, load_ports, project_root
    from so101.hardware import make_follower
    from so101.home import interpolate_to_qpos
    from so101.keys import StdinKeys
    from so101.models.act import ACTPolicy, find_policy_weights
    from so101.train import detect_device

    root = project_root()
    ckpt = Path(checkpoint) if checkpoint else root / "outputs" / "train" / f"act_{dataset_name}"
    try:
        find_policy_weights(ckpt)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No ACT checkpoint at {ckpt}. Train first: python -m so101 train") from exc
    ports = load_ports()
    cameras = load_cameras()
    cam0 = cameras[0]
    names = [c.name for c in cameras]
    device = detect_device()
    policy = ACTPolicy.load(ckpt, device=device)
    use_depth = bool(policy.cfg.use_depth)
    depth_name = policy.cfg.depth_camera_name
    print("EVAL: follower USB + 12V ON. Leader can be OFF. Omen publisher must be streaming.")
    if use_depth:
        print(f"Depth: {depth_name} uint16 from ZMQ (Omen publisher must not use --no-depth)")
    proto_path = protocol.strip()
    cam_label = ",".join(names)
    if proto_path:
        print(f"Policy {ckpt}  camera={cam_label}  {cam0.width}x{cam0.height}@{cam0.fps}  protocol={proto_path}")
    else:
        print(f"Policy {ckpt}  camera={cam_label}  {cam0.width}x{cam0.height}@{cam0.fps}  duration={duration_s}s")
    if confirm:
        input("Confirm hardware, then press Enter (Ctrl-C to abort) ")
    follower = make_follower(ports, cameras, with_cameras=True)
    follower.connect()
    depth_source = None
    if use_depth:
        from so101.zmq_rgbd import LatestZMQRGBD

        depth_source = LatestZMQRGBD(cam0.zmq_host, cam0.zmq_port)
    try:
        home = _dataset_home_qpos(root, dataset_name)
        if proto_path:
            from so101.eval_protocol import load_protocol, run_protocol

            loaded = load_protocol(proto_path)
            if loaded.home_qpos is None:
                loaded = replace(loaded, home_qpos=home)
            print(f"Home qpos={list(loaded.home_qpos)}  steps={loaded.home_steps}  dt={loaded.home_dt}s")
            run_id = time.strftime("%Y%m%d_%H%M%S")
            run_dir = Path(output_dir) if output_dir else root / "outputs" / "eval" / ckpt.name / run_id
            kw = dict(
                protocol=loaded,
                follower=follower,
                policy=policy,
                camera_names=names,
                fps=cam0.fps,
                run_dir=run_dir,
                n_trials=n_trials,
                reset_confirm=reset_confirm,
                depth_source=depth_source,
                depth_camera_name=depth_name,
            )
            if keys is not None:
                summary = run_protocol(keys=keys, **kw)
            else:
                with StdinKeys(allowed={"y", "n", "s", "f"}) as stdin_keys:
                    summary = run_protocol(keys=stdin_keys, **kw)
            return int(summary["n_success"])
        print(f"Home qpos={list(home)}")
        interpolate_to_qpos(follower, home, n_steps=80, dt=0.04, sleep=time.sleep)
        return run_eval_loop(
            follower=follower,
            policy=policy,
            camera_names=names,
            fps=cam0.fps,
            duration_s=duration_s,
            depth_source=depth_source,
            depth_camera_name=depth_name,
        )
    finally:
        if depth_source is not None:
            depth_source.close()
        follower.disconnect()
