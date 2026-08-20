"""Tests for ZMQ depth encode/decode."""

from __future__ import annotations

import numpy as np

from so101.zmq_rgbd import decode_depth_png_b64, depth_colormap_bgr, depth_png_b64


def test_depth_png_roundtrip() -> None:
    z = np.array([[0, 500], [1200, 2000]], dtype=np.uint16)
    out = decode_depth_png_b64(depth_png_b64(z))
    assert out.dtype == np.uint16
    assert out.shape == z.shape
    np.testing.assert_array_equal(out, z)


def test_depth_colormap_shape() -> None:
    z = np.zeros((24, 32), dtype=np.uint16)
    z[8:16, 10:20] = 800
    vis = depth_colormap_bgr(z)
    assert vis.shape == (24, 32, 3)
    assert vis.dtype == np.uint8
