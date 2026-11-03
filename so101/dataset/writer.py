from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from so101.dataset.reader import SO101Dataset, episode_dir
from so101.dataset.schema import EpisodeMeta, N_JOINTS
from so101.dataset.video import write_mp4


class EpisodeWriter:
    def __init__(
        self,
        dataset: SO101Dataset,
        index: int,
        *,
        source: str = "teleop",
        intervention: bool = False,
        tags: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.dataset = dataset
        self.index = index
        self.source = source
        self.intervention = intervention
        self.tags = tuple(str(t) for t in (tags or ()) if str(t).strip())
        self._names = self.dataset.meta.names()
        self._depth_names = self.dataset.meta.depth_names()
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._frames: dict[str, list[np.ndarray]] = {name: [] for name in self._names}
        self._depth_frames: dict[str, list[np.ndarray]] = {name: [] for name in self._depth_names}
        self._aborted = False
        self._saved = False

    def add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        timestamp: float,
        image_rgb: np.ndarray | None = None,
        images: dict[str, np.ndarray] | None = None,
        depth_u16: np.ndarray | None = None,
        depths: dict[str, np.ndarray] | None = None,
    ) -> None:
        if self._aborted or self._saved:
            raise RuntimeError("Cannot add frames after save/abort")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.shape[0] != N_JOINTS or action.shape[0] != N_JOINTS:
            raise ValueError(f"state and action must have length {N_JOINTS}")
        if images is None:
            if image_rgb is None:
                raise ValueError("Provide image_rgb or images")
            if len(self._names) != 1:
                raise ValueError("multi-camera episode requires images={name: rgb, ...}")
            images = {self._names[0]: image_rgb}
        missing = [name for name in self._names if name not in images]
        extra = [name for name in images if name not in self._frames]
        if missing or extra:
            raise ValueError(f"images keys {sorted(images)} != cameras {list(self._names)}")
        meta = self.dataset.meta
        checked: dict[str, np.ndarray] = {}
        for name in self._names:
            img = np.asarray(images[name])
            if img.shape[0] != meta.camera_height or img.shape[1] != meta.camera_width:
                raise ValueError(
                    f"{name} image shape {img.shape[:2]} != {(meta.camera_height, meta.camera_width)}"
                )
            if img.dtype != np.uint8:
                raise ValueError("image_rgb must be uint8")
            checked[name] = img.copy()
        self._states.append(state.copy())
        self._actions.append(action.copy())
        self._timestamps.append(float(timestamp))
        for name, img in checked.items():
            self._frames[name].append(img)

        if self._depth_names:
            if depths is None:
                if depth_u16 is None:
                    raise ValueError("depths= or depth_u16 required when dataset use_depth is true")
                if len(self._depth_names) != 1:
                    raise ValueError("multi-depth episode requires depths={name: u16, ...}")
                depths = {self._depth_names[0]: depth_u16}
            missing_d = [name for name in self._depth_names if name not in depths]
            extra_d = [name for name in depths if name not in self._depth_frames]
            if missing_d or extra_d:
                raise ValueError(
                    f"depths keys {sorted(depths)} != depth cams {list(self._depth_names)}"
                )
            for name in self._depth_names:
                z = np.asarray(depths[name], dtype=np.uint16)
                if z.shape[0] != meta.camera_height or z.shape[1] != meta.camera_width:
                    raise ValueError(
                        f"depth {name} shape {z.shape[:2]} != {(meta.camera_height, meta.camera_width)}"
                    )
                self._depth_frames[name].append(z.copy())
        elif depth_u16 is not None or depths is not None:
            raise ValueError("depth provided but dataset meta.use_depth is false")

    def save(self) -> Path:
        if self._aborted:
            raise RuntimeError("Episode was aborted")
        if self._saved:
            return episode_dir(self.dataset.root, self.index)
        first = self._frames[self._names[0]]
        if not first:
            raise ValueError("Cannot save an empty episode")
        folder = episode_dir(self.dataset.root, self.index)
        folder.mkdir(parents=True, exist_ok=True)
        state = np.stack(self._states, axis=0)
        action = np.stack(self._actions, axis=0)
        timestamps = np.asarray(self._timestamps, dtype=np.float64)
        np.save(folder / "state.npy", state)
        np.save(folder / "action.npy", action)
        np.save(folder / "timestamps.npy", timestamps)
        for name in self._names:
            write_mp4(folder / f"{name}.mp4", self._frames[name], self.dataset.meta.fps)
        depth_path: Path | None = None
        if self._depth_names:
            for name in self._depth_names:
                if len(self._depth_frames[name]) != len(first):
                    raise RuntimeError(f"depth frame count mismatch for {name}")
                path = folder / f"{name}_depth.npy"
                np.save(path, np.stack(self._depth_frames[name], axis=0))
            primary = self.dataset.meta.depth_camera_name
            depth_path = folder / f"{primary}_depth.npy"
        emeta = EpisodeMeta(
            index=self.index,
            num_frames=len(first),
            task=self.dataset.meta.task,
            source=self.source,
            intervention=self.intervention,
            tags=self.tags,
        )
        (folder / "episode.json").write_text(json.dumps(emeta.to_dict(), indent=2) + "\n")
        new_count = max(self.dataset.meta.episode_count, self.index + 1)
        self.dataset.meta = replace(self.dataset.meta, episode_count=new_count)
        self.dataset.save_meta()
        self._saved = True
        for frames in self._frames.values():
            frames.clear()
        for frames in self._depth_frames.values():
            frames.clear()
        return folder

    @property
    def num_frames(self) -> int:
        return len(self._states)

    @property
    def is_saved(self) -> bool:
        return self._saved

    def abort(self) -> None:
        self._aborted = True
        for frames in self._frames.values():
            frames.clear()
        for frames in self._depth_frames.values():
            frames.clear()
        folder = episode_dir(self.dataset.root, self.index)
        if folder.exists():
            shutil.rmtree(folder)
