"""Copy this file to ``<name>.py`` (no leading underscore) to try a new policy on the follower.

Eval calls, once per control cycle:

    policy.reset()                          # at the start of each episode
    action = policy.select_action(images, state, depth_u16=depth)

Contract:
    images     uint8 HxWx3, or a list in config/camera.env order (wrist, then overhead)
    state      float32 [6], same units as state.npy
    depth_u16  uint16 millimeters, or None. Set cfg.use_depth = True to receive frames.
    action     float32 [6], same units and order as action.npy

Joint order (index 0..5):
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

Register from ``so101.algorithms.registry`` (not ``so101.algorithms``) so import
stays acyclic. Do not call list_algorithms() at import time.

The body below holds the current joint pose. Replace ``select_action`` with your
model, point ``--checkpoint`` at its weights inside ``_load``, and uncomment
``register``. Then:

    python -m so101 algorithms
    python -m so101 eval --policy my_policy --checkpoint outputs/train/my_policy
"""

from __future__ import annotations

from typing import Any

import numpy as np

from so101.algorithms.registry import Algorithm, LoadContext, register


class HoldPolicy:
    """Smoke-test policy: command the joints to stay where they are."""

    def __init__(self) -> None:
        self.cfg = type("Cfg", (), {"use_depth": False, "depth_camera_name": "wrist"})()

    def reset(self) -> None:
        return None

    def select_action(
        self,
        image_rgb: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        state: np.ndarray,
        depth_u16: np.ndarray | None = None,
    ) -> np.ndarray:
        del image_rgb, depth_u16
        return np.asarray(state, dtype=np.float32).reshape(6).copy()


def _load(ctx: LoadContext) -> HoldPolicy:
    del ctx  # read ctx.checkpoint, ctx.camera_names, ctx.device, ctx.task here
    return HoldPolicy()


def _train(ctx: Any) -> None:
    del ctx
    raise RuntimeError(
        "Train my_policy in this function, or leave train=None and document the command in Algorithm.run."
    )


# register(
#     Algorithm(
#         name="my_policy",
#         family="ml",
#         summary="One line: what the model learns.",
#         run=(
#             "train: python -m so101 train --algo my_policy\n"
#             "eval:  python -m so101 eval --policy my_policy"
#         ),
#         module="so101.algorithms.plugins.my_policy",
#         load=_load,
#         train=_train,
#     )
# )
