from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

N_JOINTS = len(JOINT_NAMES)


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    fps: int
    joint_names: list[str]
    camera_name: str
    camera_width: int
    camera_height: int
    task: str
    episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMeta:
        return cls(
            name=str(data["name"]),
            fps=int(data["fps"]),
            joint_names=list(data["joint_names"]),
            camera_name=str(data["camera_name"]),
            camera_width=int(data["camera_width"]),
            camera_height=int(data["camera_height"]),
            task=str(data["task"]),
            episode_count=int(data["episode_count"]),
        )


@dataclass(frozen=True)
class EpisodeMeta:
    index: int
    num_frames: int
    task: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeMeta:
        return cls(index=int(data["index"]), num_frames=int(data["num_frames"]), task=str(data["task"]))


@dataclass
class Episode:
    index: int
    num_frames: int
    task: str
    state: Any
    action: Any
    timestamps: Any
    video_path: Path
    camera_name: str

    def load_frames(self) -> Any:
        from so101.dataset.video import read_mp4

        return read_mp4(self.video_path)
