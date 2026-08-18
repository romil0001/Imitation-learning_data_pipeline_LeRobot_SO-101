from __future__ import annotations

import base64
import unittest

from zmq_jpeg import jpegs_from_zmq_payload


class ZmqJpegTests(unittest.TestCase):
    def test_decodes_front_and_birdseye(self) -> None:
        payload = {
            "images": {
                "front": base64.b64encode(b"FRONTJPEG").decode("ascii"),
                "birdseye": base64.b64encode(b"BIRDJPEG").decode("ascii"),
            },
            "ts": 1.0,
        }
        out = jpegs_from_zmq_payload(payload)
        self.assertEqual(out["front"], b"FRONTJPEG")
        self.assertEqual(out["birdseye"], b"BIRDJPEG")

    def test_skips_empty_and_missing_images(self) -> None:
        self.assertEqual(jpegs_from_zmq_payload({}), {})
        self.assertEqual(jpegs_from_zmq_payload({"images": {"front": ""}}), {})


if __name__ == "__main__":
    unittest.main()
