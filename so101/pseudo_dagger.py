"""Depth pseudo-DAgger: automatic corrections from wrist depth (no leader arm).

ACT (or another base policy) proposes an action. A depth blob servo nudges
shoulder_pan / lift joints so the nearest wrist-depth cluster moves toward
image center and a target range. Frames where the nudge exceeds a threshold
are logged as expert corrections for fine-tuning via ``--corrections``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from so101.dataset import JOINT_NAMES, DatasetMeta, EpisodeWriter, SO101Dataset
from so101.depth_grasp import nearest_depth_blob


@dataclass
class PseudoDaggerConfig:
    """Approach-band depth servo + logging threshold."""

    enabled: bool = True
    approach_min_mm: float = 80.0
    approach_max_mm: float = 520.0
    target_mm: float = 120.0
    tol_mm: float = 20.0
    min_mm: float = 40.0
    max_mm: float = 2000.0
    track_roi_frac: float = 0.75
    track_band_mm: float = 50.0
    gain_deg_per_norm: float = 10.0
    lateral_joints: tuple[str, ...] = ("shoulder_pan",)
    lateral_signs: dict[str, float] = field(default_factory=lambda: {"shoulder_pan": 1.0})
    gain_deg_per_mm: float = 0.06
    vertical_joints: tuple[str, ...] = ("shoulder_lift", "elbow_flex")
    joint_signs: dict[str, float] = field(
        default_factory=lambda: {"shoulder_lift": 1.0, "elbow_flex": 1.0}
    )
    max_delta_deg: float = 5.0
    log_delta_deg: float = 0.35
    blend: float = 1.0
    close_gripper: bool = False
    gripper_closed: float = 2.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PseudoDaggerConfig:
        fields = cls.__dataclass_fields__
        kwargs: dict[str, Any] = {}
        for k, v in data.items():
            if k not in fields:
                continue
            if k in {"vertical_joints", "lateral_joints"}:
                kwargs[k] = tuple(str(x) for x in v)
            elif k in {"joint_signs", "lateral_signs"}:
                kwargs[k] = {str(a): float(b) for a, b in dict(v).items()}
            else:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PseudoDaggerConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Pseudo-DAgger config must be a mapping: {path}")
        return cls.from_dict(raw)


@dataclass
class PseudoDaggerStepInfo:
    active: bool
    logged: bool
    depth_mm: float | None
    offset_xy: tuple[float, float] | None
    delta_inf: float
    action_base: np.ndarray
    action_out: np.ndarray


class PseudoDaggerServo:
    """Proportional depth centering in the approach band."""

    def __init__(self, cfg: PseudoDaggerConfig | None = None) -> None:
        self.cfg = cfg or PseudoDaggerConfig()
        self.last_info: PseudoDaggerStepInfo | None = None

    def reset(self) -> None:
        self.last_info = None

    def correct(
        self,
        action: np.ndarray,
        state: np.ndarray,
        depth_u16: np.ndarray | None,
    ) -> tuple[np.ndarray, PseudoDaggerStepInfo]:
        base = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        out = base.copy()
        info = PseudoDaggerStepInfo(
            active=False,
            logged=False,
            depth_mm=None,
            offset_xy=None,
            delta_inf=0.0,
            action_base=base,
            action_out=out,
        )
        if not self.cfg.enabled or depth_u16 is None:
            self.last_info = info
            return out, info

        blob = nearest_depth_blob(
            depth_u16,
            roi_frac=self.cfg.track_roi_frac,
            min_mm=self.cfg.min_mm,
            max_mm=self.cfg.max_mm,
            band_mm=self.cfg.track_band_mm,
        )
        if blob is None:
            self.last_info = info
            return out, info

        ox, oy, med = blob
        info.depth_mm = med
        info.offset_xy = (ox, oy)
        if not (self.cfg.approach_min_mm <= med <= self.cfg.approach_max_mm):
            self.last_info = info
            return out, info

        info.active = True
        blend = float(np.clip(self.cfg.blend, 0.0, 1.0))
        state_f = np.asarray(state, dtype=np.float32).reshape(-1)
        # Magnitude of the intentional servo nudge (not ACT↔state disagreement).
        nudge_inf = 0.0

        # Lateral: steer blob toward image center.
        for name in self.cfg.lateral_joints:
            if name not in JOINT_NAMES:
                continue
            i = JOINT_NAMES.index(name)
            sign = float(self.cfg.lateral_signs.get(name, 1.0))
            delta = sign * self.cfg.gain_deg_per_norm * float(ox)
            delta = float(np.clip(delta, -self.cfg.max_delta_deg, self.cfg.max_delta_deg))
            nudge_inf = max(nudge_inf, abs(delta))
            # Nudge the policy command (keeps ACT trajectory, adds centering).
            out[i] = float(base[i]) + blend * delta

        # Vertical: regulate wrist-object depth toward target_mm.
        error = float(med) - float(self.cfg.target_mm)
        for name in self.cfg.vertical_joints:
            if name not in JOINT_NAMES:
                continue
            i = JOINT_NAMES.index(name)
            sign = float(self.cfg.joint_signs.get(name, 1.0))
            delta = sign * self.cfg.gain_deg_per_mm * error
            delta = float(np.clip(delta, -self.cfg.max_delta_deg, self.cfg.max_delta_deg))
            nudge_inf = max(nudge_inf, abs(delta))
            out[i] = float(base[i]) + blend * delta

        if self.cfg.close_gripper and abs(error) <= self.cfg.tol_mm:
            gi = JOINT_NAMES.index("gripper")
            out[gi] = float(self.cfg.gripper_closed)

        info.delta_inf = float(nudge_inf)
        info.logged = info.delta_inf >= float(self.cfg.log_delta_deg)
        info.action_out = out
        self.last_info = info
        return out, info


class PseudoDaggerPolicy:
    """Base policy + depth pseudo-expert. ``cfg.use_depth`` always True."""

    def __init__(self, base: Any, servo: PseudoDaggerServo | None = None) -> None:
        self.base = base
        self.servo = servo or PseudoDaggerServo()
        self.cfg = type("Cfg", (), {"use_depth": True, "depth_camera_name": "wrist"})()
        base_cfg = getattr(base, "cfg", None)
        if base_cfg is not None and hasattr(base_cfg, "depth_camera_name"):
            self.cfg.depth_camera_name = base_cfg.depth_camera_name
        self.last_info: PseudoDaggerStepInfo | None = None

    def reset(self) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset()
        self.servo.reset()
        self.last_info = None

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        try:
            raw = self.base.select_action(image_rgb, state, depth_u16=depth_u16)
        except TypeError:
            raw = self.base.select_action(image_rgb, state)
        out, info = self.servo.correct(np.asarray(raw, dtype=np.float32), state, depth_u16)
        self.last_info = info
        return out


def _open_corrections_dataset(*, base: SO101Dataset, root: Path, name: str) -> SO101Dataset:
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
    if (root / "meta.json").is_file():
        dataset = SO101Dataset(root)
        if dataset.meta.episode_count == 0:
            return SO101Dataset.create(root, fresh)
        if dataset.meta.names() != fresh.names() or bool(dataset.meta.use_depth) != bool(fresh.use_depth):
            raise ValueError(
                f"Corrections at {root} cameras={list(dataset.meta.names())} "
                f"use_depth={dataset.meta.use_depth} do not match the base dataset."
            )
        print(f"Resuming pseudo-DAgger corrections {root} with {dataset.meta.episode_count} episodes", flush=True)
        return dataset
    return SO101Dataset.create(root, fresh)


def run_pseudo_dagger_loop(
    *,
    dataset: SO101Dataset,
    follower: Any,
    policy: PseudoDaggerPolicy,
    camera_names: list[str],
    fps: int,
    keys: Any,
    num_episodes: int | None = None,
    depth_source: Any,
    depth_camera_name: str = "wrist",
    episode_duration_s: float = 45.0,
    min_log_frames: int = 8,
    save_all_frames: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    episode_tags: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Roll out policy+servo. Auto-buffers correction frames; n saves, r discards, q quits.

    Keys: n=save buffer as episode, r=discard buffer / skip, space or e=end episode
    early and auto-save if enough frames, q=quit.
    """
    from so101.hardware import action_dict_from_qpos, qpos_from_obs, rgbs_from_obs

    names = list(camera_names)
    depth_names = list(dataset.meta.depth_names()) or [depth_camera_name]
    tags = list(episode_tags or ())
    saved = 0
    attempts = 0
    max_attempts = None if num_episodes is None else max(1, int(num_episodes) * 3)
    print(
        "Pseudo-DAgger (depth)  e/space=end episode (auto-save if enough frames)  "
        "n=force save  r=discard  q=quit",
        flush=True,
    )
    print(
        f"Logging frames where depth nudge >= {policy.servo.cfg.log_delta_deg} deg "
        f"(save_all={save_all_frames}). Approach band "
        f"{policy.servo.cfg.approach_min_mm:.0f}-{policy.servo.cfg.approach_max_mm:.0f} mm.",
        flush=True,
    )

    while True:
        if num_episodes is not None and saved >= num_episodes:
            break
        if max_attempts is not None and attempts >= max_attempts:
            print(
                f"Stopping after {attempts} attempts with {saved} saved "
                f"(wanted {num_episodes}). Get closer to the clip or raise approach_max_mm.",
                flush=True,
            )
            break
        attempts += 1
        policy.reset()
        writer = EpisodeWriter(
            dataset,
            dataset.meta.episode_count,
            source="pseudo_dagger",
            intervention=True,
            tags=tags,
        )
        logged = 0
        active_steps = 0
        steps = 0
        t_ep = time.perf_counter()
        print(
            f"Episode open (attempt {attempts}"
            + (f"/{max_attempts}" if max_attempts else "")
            + f", saved {saved}"
            + (f"/{num_episodes}" if num_episodes is not None else "")
            + f") → corrections buffer. Drive until timeout "
            f"({episode_duration_s:.0f}s) or press e.",
            flush=True,
        )
        abort_episode = False
        force_save = False
        while True:
            key = keys.poll() if keys is not None else None
            if key == "q":
                writer.abort()
                print("Quit", flush=True)
                return saved
            if key in {"r"}:
                writer.abort()
                abort_episode = True
                print("Discarded episode buffer", flush=True)
                break
            if key in {"n"}:
                force_save = True
                break
            if key in {"e", " "}:
                break
            if time.perf_counter() - t_ep >= episode_duration_s:
                break

            loop_t = time.perf_counter()
            obs = follower.get_observation()
            images = rgbs_from_obs(obs, names)
            image = images[0] if len(images) == 1 else images
            depths: dict[str, np.ndarray] = {}
            for dname in depth_names:
                z = depth_source.latest_depth(dname)
                if z is None:
                    raise RuntimeError(f"No depth frame yet for {dname!r}")
                depths[dname] = z
            depth_u16 = depths.get(depth_camera_name)
            if depth_u16 is None:
                depth_u16 = next(iter(depths.values()))
            qpos = qpos_from_obs(obs)
            action = policy.select_action(image, qpos, depth_u16=depth_u16)
            follower.send_action(action_dict_from_qpos(action))
            info = policy.last_info
            if info is not None and info.active:
                active_steps += 1
            should_log = save_all_frames or (info is not None and info.logged)
            if should_log:
                writer.add_frame(
                    state=qpos,
                    action=np.asarray(action, dtype=np.float32).reshape(-1),
                    timestamp=time.time(),
                    images={name: img for name, img in zip(names, images, strict=True)},
                    depths=depths,
                )
                logged += 1
            steps += 1
            if steps == 1 or steps % 50 == 0:
                d = info.depth_mm if info is not None else None
                ox = info.offset_xy[0] if info is not None and info.offset_xy else None
                print(
                    f"  step={steps} active={active_steps} logged={logged} "
                    f"depth={d if d is not None else float('nan'):.0f} "
                    f"ox={ox if ox is not None else float('nan'):+.2f} "
                    f"dInf={info.delta_inf if info else 0:.2f}",
                    flush=True,
                )
            leftover = (1.0 / fps) - (time.perf_counter() - loop_t)
            if leftover > 0:
                sleep(leftover)

        if abort_episode:
            continue
        if logged < min_log_frames and not force_save:
            writer.abort()
            print(
                f"Skipped save (logged {logged} < min {min_log_frames}). "
                "Press n next time to force, or get closer so the depth servo engages.",
                flush=True,
            )
            continue
        if logged < 1:
            writer.abort()
            print("Nothing logged — no save", flush=True)
            continue
        writer.save()
        saved += 1
        print(f"Saved pseudo-correction {writer.index:06d}  frames={logged}  ({saved} this session)", flush=True)

    return saved


def pseudo_dagger_from_env(
    *,
    dataset_name: str = "so101_bag_clip_bowl",
    corrections_name: str = "",
    checkpoint: str = "",
    num_episodes: int = 10,
    confirm: bool = True,
    policy: str = "act",
    config_path: str = "",
    task: str = "",
    episode_duration_s: float = 45.0,
    min_log_frames: int = 8,
    save_all_frames: bool = False,
    n_action_steps: int | None = 5,
    recipe: str = "",
) -> int:
    """Collect depth pseudo-corrections on the follower (leader not required)."""
    from so101.config import load_cameras, load_ports, project_root
    from so101.dataset import compute_stats, write_stats
    from so101.dataset.tags import pseudo_dagger_tags
    from so101.eval import _dataset_home_qpos, _load_eval_policy
    from so101.hardware import make_follower
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
    corr_name = corrections_name.strip() or f"{dataset_name}_pseudo_dagger"
    corr_root = root / "data" / "datasets" / corr_name
    dataset = _open_corrections_dataset(base=base, root=corr_root, name=corr_name)

    cfg_path = Path(config_path) if config_path.strip() else root / "config" / "pseudo_dagger.yaml"
    servo_cfg = PseudoDaggerConfig.from_yaml(cfg_path) if cfg_path.is_file() else PseudoDaggerConfig()

    device = detect_device()
    task_text = task.strip() or base.meta.task
    base_policy = _load_eval_policy(
        policy_kind=policy,
        policy_base="act",
        checkpoint=checkpoint,
        dataset_name=dataset_name,
        root=root,
        device=device,
        task=task_text,
        camera_names=names,
        image_height=cam0.height,
        image_width=cam0.width,
        grasp_config="",
        n_action_steps=n_action_steps,
        recipe=(recipe or "").strip(),
        low_pass_alpha=0.3 if not (recipe or "").strip() else None,
    )
    policy_obj = PseudoDaggerPolicy(base_policy, PseudoDaggerServo(servo_cfg))

    print("Pseudo-DAgger: FOLLOWER USB + 12V ON (leader optional/off). Omen must stream RGB+depth.")
    print(f"Base demos: {base_root}")
    print(f"Corrections: {corr_root}")
    print(f"Policy: {policy}  checkpoint={checkpoint or '(default)'}  config={cfg_path}")
    print(
        f"Approach {servo_cfg.approach_min_mm:.0f}-{servo_cfg.approach_max_mm:.0f} mm  "
        f"target={servo_cfg.target_mm:.0f}  log_delta>={servo_cfg.log_delta_deg}°",
        flush=True,
    )
    if confirm:
        input("Confirm hardware, then press Enter (Ctrl-C to abort) ")

    follower = make_follower(ports, cameras, with_cameras=True)
    follower.connect()
    from so101.zmq_rgbd import LatestZMQRGBD

    depth_source = LatestZMQRGBD(cam0.zmq_host, cam0.zmq_port)
    depth_name = dataset.meta.depth_camera_name or "wrist"
    depth_source.wait_for_depth(depth_name, timeout_s=20.0)
    rest_pose = None
    try:
        home = _dataset_home_qpos(root, dataset_name)
        rest_pose = home
        print(f"Moving follower to demo start pose {list(home)}", flush=True)
        interpolate_to_qpos(follower, home, n_steps=80, dt=0.04, sleep=time.sleep)
        with OperatorKeys(allowed={"n", "r", "q", "e", " "}) as op_keys:
            saved = run_pseudo_dagger_loop(
                dataset=dataset,
                follower=follower,
                policy=policy_obj,
                camera_names=names,
                fps=cam0.fps,
                keys=op_keys,
                num_episodes=num_episodes,
                depth_source=depth_source,
                depth_camera_name=depth_name,
                episode_duration_s=episode_duration_s,
                min_log_frames=min_log_frames,
                save_all_frames=save_all_frames,
                episode_tags=pseudo_dagger_tags(policy),
            )
    except KeyboardInterrupt:
        print("\nCtrl-C — parking follower before disconnect.", flush=True)
        raise SystemExit(130) from None
    finally:
        try:
            safe_power_down(follower, rest_pose, sleep=time.sleep)
        finally:
            depth_source.close()
            try:
                follower.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"follower disconnect: {exc}", flush=True)

    if dataset.meta.episode_count > 0:
        write_stats(corr_root, compute_stats(SO101Dataset(corr_root)))
        print(f"Wrote {corr_root}/stats.json  episodes={dataset.meta.episode_count}")
    print(
        f"Next: CORRECTIONS_NAME={corr_name} INIT_CHECKPOINT={checkpoint or f'outputs/train/act_{dataset_name}'} "
        f"bash scripts/19_train_dagger.sh",
        flush=True,
    )
    return saved
