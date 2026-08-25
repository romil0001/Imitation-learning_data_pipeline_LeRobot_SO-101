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


def _resolve_smolvla_checkpoint(root: Path, checkpoint: str, dataset_name: str) -> Path:
    if checkpoint:
        return Path(checkpoint)
    candidates = [
        root / "outputs" / "train" / f"smolvla_{dataset_name}",
        root / "outputs" / "train" / f"smolvla_{dataset_name}" / "checkpoints" / "last" / "pretrained_model",
    ]
    for c in candidates:
        if c.is_dir() and (
            (c / "config.json").is_file()
            or (c / "pretrained_model" / "config.json").is_file()
            or any(c.glob("**/config.json"))
        ):
            # Prefer nested pretrained_model when present (lerobot-train layout).
            nested = c / "pretrained_model"
            if nested.is_dir():
                return nested
            ckpts = sorted(c.glob("checkpoints/*/pretrained_model"))
            if ckpts:
                return ckpts[-1]
            return c
    raise FileNotFoundError(
        f"No SmolVLA checkpoint under outputs/train/smolvla_{dataset_name}. "
        "Convert + train first: bash scripts/13_convert_lerobot.sh && bash scripts/14_train_smolvla.sh"
    )


def _load_eval_policy(
    *,
    policy_kind: str,
    policy_base: str,
    checkpoint: str,
    dataset_name: str,
    root: Path,
    device: str,
    task: str,
    camera_names: list[str],
    image_height: int,
    image_width: int,
    grasp_config: str,
) -> Any:
    from so101.grasp_config import load_grasp_config
    from so101.hybrid_policy import HybridGraspPolicy
    from so101.models.act import ACTPolicy, find_policy_weights
    from so101.models.smolvla_wrap import SmolVLAPolicyWrap

    kind = policy_kind.strip().lower()
    base_kind = policy_base.strip().lower()

    def load_act() -> Any:
        ckpt = Path(checkpoint) if checkpoint else root / "outputs" / "train" / f"act_{dataset_name}"
        find_policy_weights(ckpt)
        return ACTPolicy.load(ckpt, device=device)

    def load_smolvla() -> Any:
        ckpt = _resolve_smolvla_checkpoint(root, checkpoint, dataset_name)
        return SmolVLAPolicyWrap.load(
            ckpt,
            device=device,
            task=task,
            camera_names=camera_names,
            image_height=image_height,
            image_width=image_width,
        )

    if kind == "act":
        return load_act()
    if kind == "smolvla":
        return load_smolvla()
    if kind == "hybrid":
        if base_kind == "act":
            base = load_act()
        elif base_kind == "smolvla":
            base = load_smolvla()
        else:
            raise ValueError(f"Unknown --policy-base {policy_base!r} (use act or smolvla)")
        return HybridGraspPolicy(base, grasp_cfg=load_grasp_config(grasp_config or None))
    raise ValueError(f"Unknown --policy {policy_kind!r} (use act, smolvla, or hybrid)")


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
    policy: str = "act",
    policy_base: str = "smolvla",
    grasp_config: str = "",
    task: str = "",
) -> int:
    from dataclasses import replace

    from so101.config import load_cameras, load_ports, project_root
    from so101.hardware import make_follower
    from so101.home import interpolate_to_qpos
    from so101.keys import StdinKeys
    from so101.train import detect_device

    root = project_root()
    ports = load_ports()
    cameras = load_cameras()
    cam0 = cameras[0]
    names = [c.name for c in cameras]
    device = detect_device()
    proto_path = protocol.strip()
    task_text = task.strip()
    if not task_text and proto_path:
        from so101.eval_protocol import load_protocol

        task_text = load_protocol(proto_path).task
    if not task_text:
        task_text = "Place keys into metal cup"

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
    use_depth = bool(getattr(getattr(policy_obj, "cfg", None), "use_depth", False))
    depth_name = getattr(getattr(policy_obj, "cfg", None), "depth_camera_name", "wrist")
    ckpt_label = checkpoint or f"{policy}/{policy_base if policy == 'hybrid' else policy}"
    print("EVAL: follower USB + 12V ON. Leader can be OFF. Omen publisher must be streaming.")
    print(f"Policy kind: {policy}" + (f"  base={policy_base}" if policy == "hybrid" else ""))
    if use_depth:
        print(f"Depth: {depth_name} uint16 from ZMQ (Omen publisher must not use --no-depth)")
        if policy == "hybrid":
            print(f"Depth grasp: {grasp_config or 'defaults (config/hybrid_grasp.yaml recommended)'}")
    cam_label = ",".join(names)
    if proto_path:
        print(f"Policy {ckpt_label}  camera={cam_label}  {cam0.width}x{cam0.height}@{cam0.fps}  protocol={proto_path}")
    else:
        print(f"Policy {ckpt_label}  camera={cam_label}  {cam0.width}x{cam0.height}@{cam0.fps}  duration={duration_s}s")
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
            safe_name = Path(str(ckpt_label).replace("/", "_")).name
            run_dir = Path(output_dir) if output_dir else root / "outputs" / "eval" / safe_name / run_id
            kw = dict(
                protocol=loaded,
                follower=follower,
                policy=policy_obj,
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
            policy=policy_obj,
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
