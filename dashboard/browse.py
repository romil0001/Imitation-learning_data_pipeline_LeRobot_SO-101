"""Dataset browse helpers for the operator dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from so101.dataset.reader import SO101Dataset, episode_dir
from so101.dataset.schema import EpisodeMeta
from so101.dataset.tags import effective_tags


MAX_JOINT_POINTS = 2000


def datasets_root(project_root: Path) -> Path:
    return (project_root / "data" / "datasets").resolve()


def resolve_dataset_dir(project_root: Path, name: str) -> Path:
    """Resolve a dataset folder under data/datasets; reject path traversal."""
    root = datasets_root(project_root)
    if not name or name != Path(name).name or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"invalid dataset name {name!r}")
    path = (root / name).resolve()
    if not str(path).startswith(str(root)):
        raise ValueError(f"dataset path escapes root: {name!r}")
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"dataset not found: {name}")
    return path


def list_datasets(project_root: Path) -> list[dict[str, Any]]:
    root = datasets_root(project_root)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        meta_path = child / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            raw = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append(
            {
                "name": str(raw.get("name") or child.name),
                "path_name": child.name,
                "task": str(raw.get("task", "")),
                "episode_count": int(raw.get("episode_count", 0)),
                "fps": int(raw.get("fps", 0)),
                "camera_names": list(raw.get("camera_names") or [raw.get("camera_name", "camera")]),
                "use_depth": bool(raw.get("use_depth", False)),
                "depth_camera_names": list(raw.get("depth_camera_names") or []),
            }
        )
    return out


def _episode_folder(ds: SO101Dataset, index: int) -> Path:
    folder = episode_dir(ds.root, index)
    if not folder.is_dir():
        raise FileNotFoundError(f"episode folder missing: {folder}")
    return folder


def _load_episode_meta(folder: Path) -> EpisodeMeta:
    return EpisodeMeta.from_dict(json.loads((folder / "episode.json").read_text()))


def episode_file_inventory(folder: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not folder.is_dir():
        return files
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        files.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "suffix": path.suffix.lower(),
            }
        )
    return files


def list_episodes(ds: SO101Dataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fps = max(1, int(ds.meta.fps))
    for i in range(ds.meta.episode_count):
        folder = episode_dir(ds.root, i)
        if not (folder / "episode.json").is_file():
            continue
        emeta = _load_episode_meta(folder)
        tags = effective_tags(emeta.tags, source=emeta.source, intervention=emeta.intervention)
        rows.append(
            {
                "index": emeta.index,
                "num_frames": emeta.num_frames,
                "duration_s": round(emeta.num_frames / fps, 3),
                "task": emeta.task,
                "source": emeta.source,
                "intervention": emeta.intervention,
                "tags": tags,
                "files": [f["name"] for f in episode_file_inventory(folder)],
            }
        )
    return rows


def dataset_detail(ds: SO101Dataset) -> dict[str, Any]:
    return {
        "meta": ds.meta.to_dict(),
        "episodes": list_episodes(ds),
    }


def episode_detail(ds: SO101Dataset, index: int, *, dataset_name: str | None = None) -> dict[str, Any]:
    folder = _episode_folder(ds, index)
    emeta = _load_episode_meta(folder)
    tags = effective_tags(emeta.tags, source=emeta.source, intervention=emeta.intervention)
    url_name = dataset_name or ds.root.name
    arrays: dict[str, Any] = {}
    for name in ("state.npy", "action.npy", "timestamps.npy"):
        path = folder / name
        if not path.is_file():
            continue
        arr = np.load(path, mmap_mode="r")
        arrays[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nbytes": int(arr.nbytes),
        }
    for name in ds.meta.depth_names():
        path = folder / f"{name}_depth.npy"
        if path.is_file():
            arr = np.load(path, mmap_mode="r")
            arrays[path.name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": int(arr.nbytes),
            }
    return {
        "index": emeta.index,
        "num_frames": emeta.num_frames,
        "task": emeta.task,
        "source": emeta.source,
        "intervention": emeta.intervention,
        "tags": tags,
        "fps": ds.meta.fps,
        "camera_names": list(ds.meta.names()),
        "depth_camera_names": list(ds.meta.depth_names()),
        "arrays": arrays,
        "files": episode_file_inventory(folder),
        "video_urls": {
            name: f"/api/datasets/{url_name}/episodes/{index}/video/{name}"
            for name in ds.meta.names()
            if (folder / f"{name}.mp4").is_file()
        },
        "depth_cameras": [
            name
            for name in ds.meta.depth_names()
            if (folder / f"{name}_depth.npy").is_file()
        ],
    }


def downsample_joints(
    state: np.ndarray,
    action: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_points: int = MAX_JOINT_POINTS,
) -> dict[str, Any]:
    n = int(state.shape[0])
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    return {
        "num_frames": n,
        "sampled": int(idx.shape[0]),
        "indices": idx.tolist(),
        "timestamps": np.asarray(timestamps, dtype=np.float64)[idx].tolist(),
        "state": np.asarray(state, dtype=np.float32)[idx].tolist(),
        "action": np.asarray(action, dtype=np.float32)[idx].tolist(),
    }


def episode_joints(ds: SO101Dataset, index: int) -> dict[str, Any]:
    folder = _episode_folder(ds, index)
    state = np.load(folder / "state.npy")
    action = np.load(folder / "action.npy")
    timestamps = np.load(folder / "timestamps.npy")
    payload = downsample_joints(state, action, timestamps)
    payload["joint_names"] = list(ds.meta.joint_names)
    payload["fps"] = ds.meta.fps
    payload["index"] = index
    return payload


def video_path(ds: SO101Dataset, index: int, camera_name: str) -> Path:
    folder = _episode_folder(ds, index)
    if camera_name not in ds.meta.names():
        raise FileNotFoundError(f"unknown camera {camera_name!r}")
    path = folder / f"{camera_name}.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"missing video {path.name}")
    return path


def depth_frame_png(ds: SO101Dataset, index: int, camera_name: str, frame: int) -> bytes:
    folder = _episode_folder(ds, index)
    path = folder / f"{camera_name}_depth.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing depth {path.name}")
    arr = np.load(path, mmap_mode="r")
    if frame < 0 or frame >= arr.shape[0]:
        raise IndexError(f"frame {frame} out of range (0..{arr.shape[0] - 1})")
    depth = np.asarray(arr[frame], dtype=np.float32)
    valid = depth > 0
    if not np.any(valid):
        gray = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo = float(np.percentile(depth[valid], 5))
        hi = float(np.percentile(depth[valid], 95))
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
        gray[~valid] = 0
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    ok, buf = cv2.imencode(".png", colored)
    if not ok:
        raise RuntimeError("failed to encode depth PNG")
    return buf.tobytes()
