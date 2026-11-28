"""Online residual SAC on the follower: ACT base + dense depth reward (+ phase-mix)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from so101.hardware import action_dict_from_qpos, qpos_from_obs, rgbs_from_obs
from so101.live_viz import publish_live_joints
from so101.rl.dense_reward import DenseDepthReward, DenseDepthRewardConfig
from so101.rl.residual_policy import ResidualRLPolicy
from so101.rl.residual_sac import ResidualSAC, ResidualSACConfig


def _load_rl_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return raw if isinstance(raw, dict) else {}


def _should_learn(policy: Any, prev_phase: str | None) -> bool:
    """Phase-mix: only mid-band steps enter the SAC buffer. Plain residual: always."""
    if not hasattr(policy, "last_phase"):
        return True
    return prev_phase == "mid"


def run_residual_rl_loop(
    *,
    follower: Any,
    policy: Any,
    agent: ResidualSAC,
    reward_fn: DenseDepthReward,
    camera_names: list[str],
    fps: float,
    depth_source: Any,
    depth_camera_name: str = "wrist",
    keys: Any | None = None,
    num_episodes: int = 20,
    episode_duration_s: float = 45.0,
    output_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
    home_fn: Callable[[], None] | None = None,
) -> int:
    """Collect online transitions and update residual SAC. Returns episodes run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    dt = 1.0 / max(1e-3, float(fps))
    completed = 0
    phase_mix = hasattr(policy, "last_phase")

    print(
        f"{'Phase-mix' if phase_mix else 'Residual'} RL: {num_episodes} episodes × "
        f"{episode_duration_s:.0f}s  max_delta=±{agent.cfg.max_delta_deg}°  "
        f"buffer→{agent.cfg.buffer_size}",
        flush=True,
    )
    if phase_mix:
        print(
            f"Learn mid-band only  far>{getattr(policy, 'far_enter_mm', '?')}mm  "
            f"near≤{getattr(policy, 'near_enter_mm', '?')}mm",
            flush=True,
        )
    print("Keys: e/space=end episode  q=quit", flush=True)

    for ep in range(num_episodes):
        if home_fn is not None:
            home_fn()
        policy.reset()
        ep_ret = 0.0
        steps = 0
        mid_steps = 0
        t0 = time.perf_counter()
        prev_feat: np.ndarray | None = None
        prev_res: np.ndarray | None = None
        prev_phase: str | None = None
        print(f"Episode {ep + 1}/{num_episodes} — exploring", flush=True)

        while time.perf_counter() - t0 < episode_duration_s:
            key = keys.poll() if keys is not None else None
            if key == "q":
                print("Quit", flush=True)
                agent.save(output_dir)
                return completed
            if key in {"e", " "}:
                break

            loop_t = time.perf_counter()
            obs = follower.get_observation()
            images = rgbs_from_obs(obs, camera_names)
            image = images[0] if len(images) == 1 else images
            depth_u16 = depth_source.latest_depth(depth_camera_name)
            if depth_u16 is None:
                raise RuntimeError(f"No depth frame for {depth_camera_name!r}")
            qpos = qpos_from_obs(obs)
            action = policy.select_action(image, qpos, depth_u16=depth_u16)
            follower.send_action(action_dict_from_qpos(action))
            try:
                publish_live_joints(qpos, action)
            except Exception:  # noqa: BLE001
                pass

            rs = reward_fn.step(qpos, depth_u16, done=False)
            r = float(rs.reward)
            if prev_feat is not None and prev_res is not None and _should_learn(policy, prev_phase):
                agent.push(prev_feat, prev_res, r, rs.features, done=False)
                if agent._steps % max(1, agent.cfg.update_every) == 0:
                    stats = agent.update()
                    if stats and steps % 25 == 0:
                        phase_s = f" phase={getattr(policy, 'last_phase', '-')}"
                        print(
                            f"  step={steps} R_ep={ep_ret:.2f} "
                            f"r={r:.3f} d={rs.depth_mm or float('nan'):.0f} "
                            f"q={stats.get('q_loss', float('nan')):.3f}{phase_s}",
                            flush=True,
                        )
            ep_ret += r
            prev_feat = rs.features
            prev_res = (
                policy.last_residual
                if policy.last_residual is not None
                else np.zeros(agent.cfg.act_dim, dtype=np.float32)
            )
            prev_phase = getattr(policy, "last_phase", None)
            if prev_phase == "mid":
                mid_steps += 1
            steps += 1
            leftover = dt - (time.perf_counter() - loop_t)
            if leftover > 0:
                sleep(leftover)

        # bootstrap terminal transition
        if prev_feat is not None and prev_res is not None and _should_learn(policy, prev_phase):
            obs = follower.get_observation()
            qpos = qpos_from_obs(obs)
            depth_u16 = depth_source.latest_depth(depth_camera_name)
            rs = reward_fn.step(qpos, depth_u16, done=True)
            agent.push(prev_feat, prev_res, rs.reward, rs.features, done=True)
            agent.update()
            ep_ret += float(rs.reward)

        completed += 1
        row = {
            "episode": ep,
            "return": ep_ret,
            "steps": steps,
            "mid_steps": mid_steps if phase_mix else steps,
            "buffer": agent.buffer.size,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        mid_msg = f"  mid={mid_steps}" if phase_mix else ""
        print(
            f"Episode {ep + 1} done  return={ep_ret:.2f}  steps={steps}{mid_msg}",
            flush=True,
        )
        if (ep + 1) % 2 == 0:
            agent.save(output_dir)
            print(f"  checkpoint → {output_dir}", flush=True)

    agent.save(output_dir)
    print(f"Saved residual SAC → {output_dir}", flush=True)
    return completed


def residual_rl_from_env(
    *,
    dataset_name: str = "so101_bag_clip_bowl",
    checkpoint: str = "",
    residual_checkpoint: str = "",
    output_dir: str = "",
    config_path: str = "",
    num_episodes: int = 20,
    episode_duration_s: float = 45.0,
    policy: str = "act",
    confirm: bool = True,
    explore: bool = True,
    device: str = "",
) -> int:
    """Hardware entry: load ACT, optional residual warm-start, online SAC."""
    from so101.config import load_cameras, load_ports, project_root
    from so101.eval import _dataset_home_qpos, _load_eval_policy
    from so101.hardware import make_follower
    from so101.home import interpolate_to_qpos, safe_power_down
    from so101.keys import OperatorKeys
    from so101.train import detect_device
    from so101.zmq_rgbd import LatestZMQRGBD

    root = project_root()
    ports = load_ports()
    cameras = load_cameras()
    cam0 = cameras[0]
    names = [c.name for c in cameras]
    cfg_path = Path(config_path) if config_path.strip() else root / "config" / "phase_mix.yaml"
    raw = _load_rl_config(cfg_path)
    reward_cfg = DenseDepthRewardConfig.from_dict(raw.get("reward") or {})
    sac_cfg = ResidualSACConfig.from_dict(raw.get("sac") or {})
    ctrl = raw.get("control") or {}
    phases = raw.get("phases") or {}
    use_phase_mix = bool(phases) or "phase_mix" in cfg_path.name
    base_scale = float(ctrl.get("base_scale", 1.0))
    residual_scale = float(ctrl.get("residual_scale", 1.0))
    low_pass_alpha = float(ctrl.get("low_pass_alpha", 0.0))
    auto_close = bool(ctrl.get("auto_close", True))
    auto_close_tol_mm = float(ctrl.get("auto_close_tol_mm", 40.0))
    gripper_closed = float(
        ctrl.get("gripper_closed", (raw.get("reward") or {}).get("gripper_closed", 2.0))
    )
    far_enter_mm = float(phases.get("far_enter_mm", 280.0))
    near_enter_mm = float(phases.get("near_enter_mm", 180.0))
    dev = device.strip() or detect_device()
    sac_cfg.device = "mps" if dev == "mps" else ("cuda" if dev == "cuda" else "cpu")

    default_out = "phase_mix_act" if use_phase_mix else "residual_rl_act"
    out = Path(output_dir) if output_dir.strip() else root / "outputs" / "train" / default_out
    ckpt = checkpoint.strip()
    if not ckpt:
        v2 = root / "outputs" / "train" / f"act_{dataset_name}_v2"
        ckpt = str(v2 if v2.is_dir() else root / "outputs" / "train" / f"act_{dataset_name}")

    base_policy = _load_eval_policy(
        policy_kind=policy,
        policy_base="act",
        checkpoint=ckpt,
        dataset_name=dataset_name,
        root=root,
        device=dev,
        task="",
        camera_names=names,
        image_height=cam0.height,
        image_width=cam0.width,
        grasp_config="",
        n_action_steps=5,
        recipe="",
        low_pass_alpha=0.3,
    )

    if residual_checkpoint.strip():
        agent = ResidualSAC.load(residual_checkpoint.strip(), device=sac_cfg.device)
        # keep buffer empty for new run but reuse weights
        agent.buffer = agent.buffer.__class__(
            agent.cfg.buffer_size, agent.cfg.obs_dim, agent.cfg.act_dim
        )
        # Prefer YAML max_delta for safety even when warm-starting
        agent.cfg.max_delta_deg = sac_cfg.max_delta_deg
        agent.cfg.alpha = sac_cfg.alpha
    else:
        agent = ResidualSAC(sac_cfg)

    reward_fn = DenseDepthReward(reward_cfg)
    if use_phase_mix:
        from so101.grasp_config import load_grasp_config
        from so101.rl.phase_mix_policy import PhaseMixPolicy

        grasp_path = str(raw.get("grasp_config") or "")
        if grasp_path and not Path(grasp_path).is_file():
            alt = root / grasp_path
            if alt.is_file():
                grasp_path = str(alt)
        policy_obj = PhaseMixPolicy(
            base_policy,
            agent,
            reward_fn,
            grasp_cfg=load_grasp_config(grasp_path or None),
            explore=explore,
            far_enter_mm=far_enter_mm,
            near_enter_mm=near_enter_mm,
            base_scale=base_scale,
            residual_scale=residual_scale,
            low_pass_alpha=low_pass_alpha,
            approach_dampen_mm=float(ctrl.get("approach_dampen_mm", 260.0)),
            approach_dampen_scale=float(ctrl.get("approach_dampen_scale", 0.35)),
            auto_close=auto_close,
            auto_close_tol_mm=auto_close_tol_mm,
            gripper_closed=gripper_closed,
        )
    else:
        policy_obj = ResidualRLPolicy(
            base_policy,
            agent,
            reward_fn,
            explore=explore,
            base_scale=base_scale,
            residual_scale=residual_scale,
            low_pass_alpha=low_pass_alpha,
            auto_close=auto_close,
            auto_close_tol_mm=auto_close_tol_mm,
            gripper_closed=gripper_closed,
        )
    print(
        f"Control: base_scale={base_scale} residual_scale={residual_scale} "
        f"low_pass={low_pass_alpha} auto_close={auto_close}@{auto_close_tol_mm:.0f}mm "
        f"max_delta=±{agent.cfg.max_delta_deg}°",
        flush=True,
    )
    if use_phase_mix:
        print(
            f"Phases: far d>{far_enter_mm:.0f}mm  mid  near d≤{near_enter_mm:.0f}mm",
            flush=True,
        )

    label = "Phase-mix" if use_phase_mix else "Residual RL"
    print(f"{label} — FOLLOWER + Omen RGB-D (leader OFF).")
    print(f"Base ACT: {ckpt}")
    print(f"Config:   {cfg_path}")
    print(f"Output:   {out}")
    if confirm:
        input("Confirm hardware, then press Enter (Ctrl-C to abort) ")

    follower = make_follower(ports, cameras, with_cameras=True)
    follower.connect()
    depth_source = LatestZMQRGBD(cam0.zmq_host, cam0.zmq_port)
    depth_source.wait_for_depth("wrist", timeout_s=20.0)
    home = _dataset_home_qpos(root, dataset_name)
    rest = home

    def _go_home() -> None:
        print(f"Home {list(home)}", flush=True)
        interpolate_to_qpos(follower, home, n_steps=80, dt=0.04, sleep=time.sleep)

    try:
        _go_home()
        with OperatorKeys(allowed={"q", "e", " "}) as op_keys:
            return run_residual_rl_loop(
                follower=follower,
                policy=policy_obj,
                agent=agent,
                reward_fn=reward_fn,
                camera_names=names,
                fps=cam0.fps,
                depth_source=depth_source,
                depth_camera_name="wrist",
                keys=op_keys,
                num_episodes=num_episodes,
                episode_duration_s=episode_duration_s,
                output_dir=out,
                home_fn=_go_home,
            )
    except KeyboardInterrupt:
        print("\nCtrl-C — parking.", flush=True)
        raise SystemExit(130) from None
    finally:
        try:
            safe_power_down(follower, rest, sleep=time.sleep)
        finally:
            depth_source.close()
            try:
                follower.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"follower disconnect: {exc}", flush=True)
