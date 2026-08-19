from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from so101.dataset.reader import SO101Dataset, episode_dir
from so101.dataset.schema import DatasetMeta, EpisodeMeta, N_JOINTS
from so101.dataset.video import write_mp4


class EpisodeWriter:
    def __init__(self, dataset: SO101Dataset, index: int) -> None:
        self.dataset = dataset
        self.index = index
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._frames: list[np.ndarray] = []
        self._aborted = False
        self._saved = False

    def add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        timestamp: float,
        image_rgb: np.ndarray,
    ) -> None:
        if self._aborted or self._saved:
            raise RuntimeError("Cannot add frames after save/abort")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.shape[0] != N_JOINTS or action.shape[0] != N_JOINTS:
            raise ValueError(f"state and action must have length {N_JOINTS}")
        img = np.asarray(image_rgb)
        meta = self.dataset.meta
        if img.shape[0] != meta.camera_height or img.shape[1] != meta.camera_width:
            raise ValueError(
                f"image shape {img.shape[:2]} != {(meta.camera_height, meta.camera_width)}"
            )
        if img.dtype != np.uint8:
            raise ValueError("image_rgb must be uint8")
        self._states.append(state.copy())
        self._actions.append(action.copy())
        self._timestamps.append(float(timestamp))
        self._frames.append(img.copy())

    def save(self) -> Path:
        if self._aborted:
            raise RuntimeError("Episode was aborted")
        if self._saved:
            return episode_dir(self.dataset.root, self.index)
        if not self._frames:
            raise ValueError("Cannot save an empty episode")
        folder = episode_dir(self.dataset.root, self.index)
        folder.mkdir(parents=True, exist_ok=True)
        state = np.stack(self._states, axis=0)
        action = np.stack(self._actions, axis=0)
        timestamps = np.asarray(self._timestamps, dtype=np.float64)
        np.save(folder / "state.npy", state)
        np.save(folder / "action.npy", action)
        np.save(folder / "timestamps.npy", timestamps)
        write_mp4(folder / f"{self.dataset.meta.camera_name}.mp4", self._frames, self.dataset.meta.fps)
        emeta = EpisodeMeta(index=self.index, num_frames=len(self._frames), task=self.dataset.meta.task)
        (folder / "episode.json").write_text(json.dumps(emeta.to_dict(), indent=2) + "\n")
        new_count = max(self.dataset.meta.episode_count, self.index + 1)
        self.dataset.meta = DatasetMeta(
            name=self.dataset.meta.name,
            fps=self.dataset.meta.fps,
            joint_names=list(self.dataset.meta.joint_names),
            camera_name=self.dataset.meta.camera_name,
            camera_width=self.dataset.meta.camera_width,
            camera_height=self.dataset.meta.camera_height,
            task=self.dataset.meta.task,
            episode_count=new_count,
        )
        self.dataset.save_meta()
        self._saved = True
        self._frames.clear()
        return folder

    def abort(self) -> None:
        self._aborted = True
        self._frames.clear()
        folder = episode_dir(self.dataset.root, self.index)
        if folder.exists():
            shutil.rmtree(folder)
