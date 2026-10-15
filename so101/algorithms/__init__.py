"""Setup algorithms (the robot cell) and ML algorithms (learned policies).

``python -m so101 algorithms`` prints both lists. A new learned policy is a file
in ``so101/algorithms/plugins/`` that implements ``select_action`` and registers
itself. Eval and the depth-grasp hybrid wrapper then run it with no other edits.
"""

from __future__ import annotations

from so101.algorithms.registry import (
    Algorithm,
    LoadContext,
    TrainContext,
    ensure_registered,
    format_catalog,
    get,
    list_algorithms,
    load_policy,
    names,
    register,
    runnable_names,
    train_algorithm,
    unregister,
)

ensure_registered()

__all__ = [
    "Algorithm",
    "LoadContext",
    "TrainContext",
    "ensure_registered",
    "format_catalog",
    "get",
    "list_algorithms",
    "load_policy",
    "names",
    "register",
    "runnable_names",
    "train_algorithm",
    "unregister",
]
