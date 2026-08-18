from __future__ import annotations

import json
import unittest

import cv2
import numpy as np

from publish_d405 import bgr_to_jpeg_b64, zmq_payload


class PayloadTests(unittest.TestCase):
    def test_zmq_payload_includes_both_keys(self) -> None:
        raw = zmq_payload({"front": "aaa", "birdseye": "bbb"}, 1.5)
        data = json.loads(raw)
        self.assertEqual(data["images"]["front"], "aaa")
        self.assertEqual(data["images"]["birdseye"], "bbb")
        self.assertEqual(data["ts"], 1.5)

    def test_bgr_to_jpeg_b64_roundtrip(self) -> None:
        bgr = np.zeros((16, 24, 3), dtype=np.uint8)
        bgr[:] = (0, 255, 0)
        b64 = bgr_to_jpeg_b64(bgr, 70)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 10)
        buf = np.frombuffer(__import__("base64").b64decode(b64), dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
