"""Dataset browse APIs on the operator dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import app as dash  # noqa: E402
from dashboard.browse import (  # noqa: E402
    depth_frame_png,
    list_datasets,
    resolve_dataset_dir,
)
from so101.dataset import (  # noqa: E402
    JOINT_NAMES,
    DatasetMeta,
    EpisodeWriter,
    SO101Dataset,
    collection_tags,
)


def _rgb(h: int, w: int, v: int) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (v, 20, 40)
    return frame


def _make_dataset(root: Path, *, n_frames: int = 4, with_depth: bool = True) -> SO101Dataset:
    meta = DatasetMeta(
        name=root.name,
        fps=15,
        joint_names=list(JOINT_NAMES),
        camera_name="wrist",
        camera_names=("wrist", "overhead"),
        camera_width=32,
        camera_height=24,
        task="Browse test",
        episode_count=0,
        use_depth=with_depth,
        depth_camera_name="wrist",
        depth_camera_names=("wrist",) if with_depth else (),
    )
    ds = SO101Dataset.create(root, meta)
    writer = EpisodeWriter(ds, index=0, tags=collection_tags())
    for t in range(n_frames):
        depth = None
        depths = None
        if with_depth:
            depths = {"wrist": np.full((24, 32), 500 + t, dtype=np.uint16)}
        writer.add_frame(
            state=np.linspace(t, t + 5, 6, dtype=np.float32),
            action=np.linspace(t, t + 5, 6, dtype=np.float32) + 1,
            timestamp=t / 15.0,
            images={"wrist": _rgb(24, 32, t * 10), "overhead": _rgb(24, 32, 80 + t)},
            depths=depths,
            depth_u16=depth,
        )
    writer.save()
    return SO101Dataset(root)


@pytest.fixture()
def browse_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    data_root = tmp_path / "data" / "datasets"
    data_root.mkdir(parents=True)
    _make_dataset(data_root / "unit_browse")
    monkeypatch.setattr(dash, "ROOT", tmp_path)
    return TestClient(dash.app)


def test_resolve_dataset_dir_rejects_traversal(tmp_path: Path) -> None:
    (tmp_path / "data" / "datasets").mkdir(parents=True)
    with pytest.raises(ValueError):
        resolve_dataset_dir(tmp_path, "../secrets")
    with pytest.raises(ValueError):
        resolve_dataset_dir(tmp_path, "foo/bar")
    with pytest.raises(FileNotFoundError):
        resolve_dataset_dir(tmp_path, "missing_ds")


def test_list_datasets_finds_meta(tmp_path: Path) -> None:
    root = tmp_path / "data" / "datasets" / "demo"
    _make_dataset(root, with_depth=False)
    rows = list_datasets(tmp_path)
    assert len(rows) == 1
    assert rows[0]["path_name"] == "demo"
    assert rows[0]["episode_count"] == 1


def test_api_datasets_and_episode(browse_client: TestClient) -> None:
    r = browse_client.get("/api/datasets")
    assert r.status_code == 200
    names = [d["path_name"] for d in r.json()["datasets"]]
    assert "unit_browse" in names

    r = browse_client.get("/api/datasets/unit_browse")
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["name"] == "unit_browse"
    assert body["episodes"][0]["tags"]
    assert "success_collection" in body["episodes"][0]["tags"]

    r = browse_client.get("/api/datasets/unit_browse/episodes/0")
    assert r.status_code == 200
    detail = r.json()
    assert detail["num_frames"] == 4
    assert "state.npy" in detail["arrays"]
    assert "wrist" in detail["video_urls"]
    assert "wrist" in detail["depth_cameras"]

    r = browse_client.get("/api/datasets/unit_browse/episodes/0/joints")
    assert r.status_code == 200
    joints = r.json()
    assert joints["num_frames"] == 4
    assert len(joints["state"]) == 4
    assert len(joints["state"][0]) == 6

    r = browse_client.get("/api/datasets/unit_browse/episodes/0/files")
    assert r.status_code == 200
    names = {f["name"] for f in r.json()["files"]}
    assert "state.npy" in names
    assert "wrist.mp4" in names


def test_api_path_traversal_rejected(browse_client: TestClient) -> None:
    r = browse_client.get("/api/datasets/..%2Fsecrets")
    assert r.status_code in {400, 404}


def test_api_video_and_depth_frame(browse_client: TestClient) -> None:
    r = browse_client.get("/api/datasets/unit_browse/episodes/0/video/wrist")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("video/")
    assert len(r.content) > 0

    r = browse_client.get("/api/datasets/unit_browse/episodes/0/depth/wrist/frame/1")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    r = browse_client.get("/api/datasets/unit_browse/episodes/0/depth/wrist/frame/99")
    assert r.status_code == 400


def test_depth_frame_png_helper(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path / "depth_only")
    png = depth_frame_png(ds, 0, "wrist", 0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
