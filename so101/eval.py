"""Run a trained ACT policy on the follower. Omen D405 publisher must be up."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from so101.hardware import action_dict_from_qpos, qpos_from_obs, rgbs_from_obs


def open_success_dataset(
    *,
    base_dataset_name: str,
    success_dataset_name: str = "",
    root: Path | None = None,
) -> Any:
    """Create or resume ``{base}_eval`` for autonomous success trajectories."""
    from so101.config import project_root
    from so101.dataset import JOINT_NAMES, DatasetMeta, SO101Dataset

    proj = Path(root) if root is not None else project_root()
    base = SO101Dataset(proj / "data" / "datasets" / base_dataset_name)
    name = (success_dataset_name or "").strip() or f"{base_dataset_name}_eval"
    dest = proj / "data" / "datasets" / name
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
        depth_camera_names=base.meta.depth_names(),
    )
    if (dest / "meta.json").is_file():
        dataset = SO101Dataset(dest)
        if dataset.meta.episode_count == 0:
            return SO101Dataset.create(dest, fresh)
        if dataset.meta.names() != fresh.names() or bool(dataset.meta.use_depth) != bool(fresh.use_depth):
            raise ValueError(
                f"Success dataset at {dest} cameras={list(dataset.meta.names())} "
                f"use_depth={dataset.meta.use_depth} does not match {base_dataset_name}."
            )
        if (
            dataset.meta.camera_width != fresh.camera_width
            or dataset.meta.camera_height != fresh.camera_height
        ):
            raise ValueError(
                f"Success dataset at {dest} resolution "
                f"{dataset.meta.camera_width}x{dataset.meta.camera_height} "
                f"!= {fresh.camera_width}x{fresh.camera_height}"
            )
        print(f"Resuming success demos {dest} with {dataset.meta.episode_count} episodes", flush=True)
        return dataset
    print(f"Creating success demo dataset {dest}", flush=True)
    return SO101Dataset.create(dest, fresh)


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
    writer: Any | None = None,
) -> int:
    names = list(camera_names) if camera_names else [str(camera_name)]
    policy.reset()
    t0 = now()
    steps = 0
    use_depth = bool(getattr(getattr(policy, "cfg", None), "use_depth", False))
    record_depth_names: tuple[str, ...] = ()
    if writer is not None:
        record_depth_names = tuple(writer.dataset.meta.depth_names())
        # Writer cameras are the source of truth for saved frames.
        names = list(writer.dataset.meta.names())
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
        depths: dict[str, Any] | None = None
        if use_depth or record_depth_names:
            if depth_source is None:
                raise RuntimeError("depth_source required when policy/dataset uses depth")
            if record_depth_names:
                depths = {}
                for dname in record_depth_names:
                    z = depth_source.latest_depth(dname)
                    if z is None:
                        raise RuntimeError(f"No depth frame yet for {dname!r}")
                    depths[dname] = z
                depth_u16 = depths.get(depth_camera_name)
                if depth_u16 is None:
                    depth_u16 = next(iter(depths.values()))
            else:
                depth_u16 = depth_source.latest_depth(depth_camera_name)
                if depth_u16 is None:
                    raise RuntimeError(f"No depth frame yet for {depth_camera_name!r}")
        qpos = qpos_from_obs(obs)
        action = policy.select_action(image, qpos, depth_u16=depth_u16)
        follower.send_action(action_dict_from_qpos(action))
        if writer is not None:
            writer.add_frame(
                state=qpos,
                action=np.asarray(action, dtype=np.float32).reshape(-1),
                timestamp=time.time(),
                images={name: img for name, img in zip(names, images, strict=True)},
                depths=depths,
                depth_u16=None if depths is not None else depth_u16,
            )
        try:
            from so101.live_viz import publish_live_joints

            publish_live_joints(qpos, action)
        except Exception:
            pass
        steps += 1
        leftover = (1.0 / fps) - (now() - loop_t)
        if leftover > 0:
            sleep(leftover)
    return steps


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
    recipe: str = "",
    n_action_steps: int | None = None,
    async_replan: bool | None = None,
    replan_threshold: int | None = None,
    low_pass_alpha: float | None = None,
    freespace_scale: float | None = None,
) -> Any:
    from so101.algorithms import LoadContext, load_policy
    from so101.recipe import default_gemfuse_recipe_path

    recipe_path = (recipe or "").strip()
    if not recipe_path and policy_kind.strip().lower() == "gemfuse":
        default = default_gemfuse_recipe_path(root)
        if default.is_file():
            recipe_path = str(default)

    ctx = LoadContext(
        checkpoint=checkpoint,
        dataset_name=dataset_name,
        root=root,
        device=device,
        task=task,
        camera_names=tuple(camera_names),
        image_height=image_height,
        image_width=image_width,
        grasp_config=grasp_config,
        policy_base=policy_base,
        recipe=recipe_path,
        n_action_steps=n_action_steps,
        async_replan=async_replan,
        replan_threshold=replan_threshold,
        low_pass_alpha=low_pass_alpha,
        freespace_scale=freespace_scale,
    )
    return load_policy(policy_kind, ctx)


def eval_from_env(
    *,
    checkpoint: str = "",
    dataset_name: str = "so101_bag_clip_bowl",
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
    recipe: str = "",
    n_action_steps: int | None = None,
    async_replan: bool | None = None,
    replan_threshold: int | None = None,
    low_pass_alpha: float | None = None,
    freespace_scale: float | None = None,
    save_successes: bool = True,
    success_dataset: str = "",
) -> int:
    from dataclasses import replace

    from so101.config import load_cameras, load_ports, project_root
    from so101.hardware import make_follower
    from so101.home import interpolate_to_qpos, safe_power_down
    from so101.keys import OperatorKeys
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
        task_text = "Place bag clip in bowl"

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
        recipe=recipe,
        n_action_steps=n_action_steps,
        async_replan=async_replan,
        replan_threshold=replan_threshold,
        low_pass_alpha=low_pass_alpha,
        freespace_scale=freespace_scale,
    )
    use_depth = bool(getattr(getattr(policy_obj, "cfg", None), "use_depth", False))
    depth_name = getattr(getattr(policy_obj, "cfg", None), "depth_camera_name", "wrist")
    ckpt_label = checkpoint or f"{policy}/{policy_base if policy == 'hybrid' else policy}"
    print("EVAL: follower USB + 12V ON. Leader can be OFF. Omen publisher must be streaming.")
    print(f"Policy kind: {policy}" + (f"  base={policy_base}" if policy == "hybrid" else ""))
    if recipe.strip() or any(
        v is not None
        for v in (n_action_steps, async_replan, replan_threshold, low_pass_alpha, freespace_scale)
    ):
        print(
            "Recipe:"
            + (f" path={recipe}" if recipe.strip() else " (CLI overrides)")
            + (f" n_action_steps={n_action_steps}" if n_action_steps is not None else "")
            + (f" async_replan={async_replan}" if async_replan is not None else "")
            + (f" replan_threshold={replan_threshold}" if replan_threshold is not None else "")
            + (f" low_pass={low_pass_alpha}" if low_pass_alpha is not None else "")
            + (f" freespace={freespace_scale}" if freespace_scale is not None else "")
        )
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
    rest_pose = None
    if use_depth:
        from so101.zmq_rgbd import LatestZMQRGBD

        depth_source = LatestZMQRGBD(cam0.zmq_host, cam0.zmq_port)
    try:
        home = _dataset_home_qpos(root, dataset_name)
        rest_pose = home
        if proto_path:
            from so101.eval_protocol import load_protocol, run_protocol

            loaded = load_protocol(proto_path)
            if loaded.home_qpos is None:
                loaded = replace(loaded, home_qpos=home)
            rest_pose = loaded.home_qpos
            print(f"Home qpos={list(loaded.home_qpos)}  steps={loaded.home_steps}  dt={loaded.home_dt}s")
            run_id = time.strftime("%Y%m%d_%H%M%S")
            safe_name = Path(str(ckpt_label).replace("/", "_")).name
            run_dir = Path(output_dir) if output_dir else root / "outputs" / "eval" / safe_name / run_id
            success_ds = None
            if save_successes:
                success_ds = open_success_dataset(
                    base_dataset_name=dataset_name,
                    success_dataset_name=success_dataset,
                    root=root,
                )
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
                success_dataset=success_ds,
                policy_tag=policy if policy != "hybrid" else f"hybrid-{policy_base}",
            )
            if keys is not None:
                summary = run_protocol(keys=keys, **kw)
            else:
                with OperatorKeys(allowed={"y", "n", "s", "f"}) as op_keys:
                    summary = run_protocol(keys=op_keys, **kw)
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
    except KeyboardInterrupt:
        print("\nCtrl-C — parking arm before disconnect.", flush=True)
        raise SystemExit(130) from None
    finally:
        try:
            safe_power_down(follower, rest_pose, sleep=time.sleep)
        finally:
            if depth_source is not None:
                depth_source.close()
            try:
                follower.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"disconnect: {exc}", flush=True)
