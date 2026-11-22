"""LeRobot ACT adapter with the same select_action(images, state) interface as the local ACT."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from so101.dataset.schema import JOINT_NAMES


@dataclass
class LeRobotACTWrapConfig:
    use_depth: bool = False
    depth_camera_name: str = "wrist"
    camera_names: tuple[str, ...] = ("wrist", "overhead", "base")
    n_action_steps: int = 100
    chunk_size: int = 100
    ensemble_temp: float | None = None
    robot_type: str = "so101_follower"
    task: str = "Place bag clip into bowl"
    image_height: int = 480
    image_width: int = 640


def resolve_lerobot_act_dir(path: Path) -> Path | None:
    """Return a directory that contains LeRobot ACT ``config.json`` and ``model.safetensors``."""
    path = Path(path)
    candidates = [
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate.resolve()
    return None


def _filtered_checkpoint_dir(path: Path) -> Path:
    """Copy the checkpoint with config keys this LeRobot build rejects removed.

    The training checkout may save fields (for example ``dtype``) that the conda
    LeRobot install does not know. Weights and processor files stay as symlinks.
    """
    from lerobot.policies.act.configuration_act import ACTConfig

    raw = json.loads((path / "config.json").read_text())
    dropped: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="so101-act-"))
    # Retry until ACTConfig accepts the file. Each failure names the unknown fields.
    for _ in range(8):
        (tmp / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
        try:
            ACTConfig.from_pretrained(tmp)
        except Exception as exc:
            names = re.findall(r"`([^`]+)`", str(exc))
            unknown = [name for name in names if name in raw and name != "ACTConfig"]
            if not unknown:
                raise
            for name in unknown:
                raw.pop(name, None)
                dropped.append(name)
            continue
        break
    else:
        raise RuntimeError(f"Could not adapt {path / 'config.json'} to this LeRobot install")
    for name in os.listdir(path):
        if name == "config.json":
            continue
        os.symlink(path / name, tmp / name)
    if dropped:
        print(f"Ignored checkpoint config fields not in this LeRobot: {', '.join(dropped)}", flush=True)
    return tmp


class LeRobotACTPolicyWrap:
    """Thin wrapper: numpy images/state → LeRobot ACT → joint qpos."""

    def __init__(
        self,
        policy: Any,
        preprocess: Any,
        postprocess: Any,
        ds_features: dict[str, Any],
        *,
        device: str = "cpu",
        cfg: LeRobotACTWrapConfig | None = None,
    ) -> None:
        self.policy = policy
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.ds_features = ds_features
        self.device = torch.device(device)
        self.cfg = cfg or LeRobotACTWrapConfig()
        self._synced_n: int | None = None

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        device: str | None = None,
        task: str = "Place bag clip into bowl",
        camera_names: tuple[str, ...] | list[str] = ("wrist", "overhead", "base"),
        image_height: int = 480,
        image_width: int = 640,
    ) -> LeRobotACTPolicyWrap:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.feature_utils import hw_to_dataset_features

        from so101.train import detect_device

        src = resolve_lerobot_act_dir(Path(checkpoint))
        if src is None:
            raise FileNotFoundError(f"No LeRobot ACT checkpoint at {checkpoint}")
        device = device or detect_device()
        try:
            model = ACTPolicy.from_pretrained(str(src))
            load_path = str(src)
        except Exception as exc:
            if "not valid for" not in str(exc):
                raise
            load_path = str(_filtered_checkpoint_dir(src))
            model = ACTPolicy.from_pretrained(load_path)
        model.to(device)
        model.eval()
        cam_names = tuple(str(n) for n in camera_names)
        img_keys = [str(key).split(".")[-1] for key in model.config.image_features]
        if img_keys and img_keys != list(cam_names):
            raise ValueError(
                f"Checkpoint cameras {img_keys} do not match eval cameras {list(cam_names)}. "
                "Use the same wrist,overhead,base order as training."
            )
        preprocess, postprocess = make_pre_post_processors(
            model.config,
            load_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        joint = {f"{name}.pos": float for name in JOINT_NAMES}
        obs: dict[str, Any] = dict(joint)
        for name in cam_names:
            obs[name] = (int(image_height), int(image_width), 3)
        ds_features = {
            **hw_to_dataset_features(joint, "action"),
            **hw_to_dataset_features(obs, "observation"),
        }
        cfg = LeRobotACTWrapConfig(
            camera_names=cam_names,
            n_action_steps=int(model.config.n_action_steps),
            chunk_size=int(model.config.chunk_size),
            task=task,
            image_height=image_height,
            image_width=image_width,
        )
        wrap = cls(model, preprocess, postprocess, ds_features, device=device, cfg=cfg)
        wrap.reset()
        return wrap

    def _sync_horizon(self) -> None:
        n = int(self.cfg.n_action_steps)
        if self._synced_n == n:
            return
        self.policy.config.n_action_steps = n
        self.policy.reset()
        self._synced_n = n

    def reset(self) -> None:
        self._synced_n = None
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        del depth_u16
        from lerobot.policies.utils import build_inference_frame, make_robot_action

        self._sync_horizon()
        if isinstance(image_rgb, (list, tuple)):
            images = list(image_rgb)
        else:
            images = [image_rgb]
        if len(images) != len(self.cfg.camera_names):
            raise ValueError(f"Got {len(images)} images but camera_names={self.cfg.camera_names}")
        observation: dict[str, Any] = {
            f"{name}.pos": float(state[i]) for i, name in enumerate(JOINT_NAMES)
        }
        for name, img in zip(self.cfg.camera_names, images, strict=True):
            observation[name] = np.asarray(img)
        frame = build_inference_frame(
            observation=observation,
            device=self.device,
            ds_features=self.ds_features,
            task=self.cfg.task,
            robot_type=self.cfg.robot_type,
        )
        batch = self.preprocess(frame)
        with torch.inference_mode():
            action_t = self.policy.select_action(batch)
        action_t = self.postprocess(action_t)
        robot_action = make_robot_action(action_t, self.ds_features)
        return np.array(
            [float(robot_action[f"{name}.pos"]) for name in JOINT_NAMES],
            dtype=np.float32,
        )
