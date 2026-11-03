"""Provenance tags stamped by record / dagger / eval / replay-capture scripts."""

from __future__ import annotations


def _policy_framework_tags(policy: str, policy_base: str = "") -> list[str]:
    kind = (policy or "").strip().lower() or "unknown"
    base = (policy_base or "").strip().lower()
    tags = [f"policy_{kind}"]
    if kind == "hybrid" and base:
        tags.append(f"framework_{base}")
    else:
        tags.append(f"framework_{kind}")
    return tags


def collection_tags() -> list[str]:
    return ["success_collection", "method_teleop", "framework_so101"]


def dagger_tags(policy: str, policy_base: str = "") -> list[str]:
    return [
        "method_dagger",
        "intervention",
        "framework_so101",
        *_policy_framework_tags(policy, policy_base),
    ]


def eval_tags(policy: str, policy_base: str = "") -> list[str]:
    return [
        "success_eval",
        "framework_so101",
        *_policy_framework_tags(policy, policy_base),
    ]


def replay_capture_tags() -> list[str]:
    return ["method_replay_capture", "framework_so101"]


def derived_tags(source: str, intervention: bool) -> list[str]:
    """Fallback badges for episodes recorded before tags existed."""
    tags: list[str] = ["framework_so101"]
    src = (source or "teleop").strip().lower()
    if src == "teleop":
        tags.extend(["success_collection", "method_teleop"])
    elif src == "dagger":
        tags.extend(["method_dagger", "intervention"])
    elif src == "replay_capture":
        tags.append("method_replay_capture")
    else:
        tags.append(f"method_{src}")
    if intervention and "intervention" not in tags:
        tags.append("intervention")
    return tags


def effective_tags(
    tags: list[str] | tuple[str, ...] | None,
    *,
    source: str = "teleop",
    intervention: bool = False,
) -> list[str]:
    if tags:
        return list(dict.fromkeys(str(t) for t in tags if str(t).strip()))
    return derived_tags(source, intervention)
