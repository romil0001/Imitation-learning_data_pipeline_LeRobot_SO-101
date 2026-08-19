"""GDM-style scored real-robot eval: conditions × trials, binary y/n success."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from so101.eval import run_eval_loop
from so101.keys import KeySource

SCORE_SUCCESS = frozenset({"y", "s"})
SCORE_FAIL = frozenset({"n", "f"})


@dataclass(frozen=True)
class Condition:
    id: str
    pen: str
    lighting: str = "overhead"
    distractors: str = "none"
    split: str = "id"


@dataclass(frozen=True)
class Protocol:
    task: str
    timeout_s: float
    n_trials_per_condition: int
    success_criterion: str
    conditions: list[Condition]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "timeout_s": self.timeout_s,
            "n_trials_per_condition": self.n_trials_per_condition,
            "success_criterion": self.success_criterion,
            "conditions": [asdict(c) for c in self.conditions],
        }


@dataclass(frozen=True)
class TrialResult:
    trial_index: int
    condition_id: str
    success: bool
    steps: int
    early_stop: bool
    timeout_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_protocol(path: str | Path) -> Protocol:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Protocol {path} must be a mapping")
    conditions = [Condition(**_condition_fields(raw)) for raw in data.get("conditions") or []]
    if not conditions:
        raise ValueError(f"Protocol {path} needs at least one condition")
    criterion = str(data.get("success_criterion") or "").strip()
    return Protocol(
        task=str(data["task"]),
        timeout_s=float(data["timeout_s"]),
        n_trials_per_condition=int(data["n_trials_per_condition"]),
        success_criterion=criterion,
        conditions=conditions,
    )


def _condition_fields(raw: dict[str, Any]) -> dict[str, Any]:
    cid = str(raw.get("id") or f"pen={raw.get('pen', 'unknown')}")
    return {
        "id": cid,
        "pen": str(raw.get("pen", "")),
        "lighting": str(raw.get("lighting", "overhead")),
        "distractors": str(raw.get("distractors", "none")),
        "split": str(raw.get("split", "id")),
    }


def score_from_key(ch: str | None) -> bool | None:
    if ch is None:
        return None
    c = ch.strip().lower()
    if c in SCORE_SUCCESS:
        return True
    if c in SCORE_FAIL:
        return False
    return None


def _pc_success(n_success: int, n: int) -> float:
    if n <= 0:
        return 0.0
    return 100.0 * n_success / n


def _default_reset_confirm(condition: Condition, trial_i: int, n_trials: int) -> None:
    print(
        f"Reset scene for {condition.id}  trial {trial_i + 1}/{n_trials}  "
        f"(pen={condition.pen}, lighting={condition.lighting}, distractors={condition.distractors})",
        flush=True,
    )
    print("Place the pen on the taped mark. Clear the gripper. Ready the keys zone.", flush=True)
    input("Press Enter when the scene is reset (Ctrl-C to abort) ")


def _wait_for_score(keys: KeySource, sleep: Callable[[float], None]) -> bool:
    print("Score this trial: y=success  n=fail  (s/f aliases)", flush=True)
    while True:
        verdict = score_from_key(keys.poll())
        if verdict is not None:
            return verdict
        sleep(0.05)


def summarize(results: list[TrialResult], condition_ids: list[str]) -> dict[str, Any]:
    n_trials = len(results)
    n_success = sum(1 for r in results if r.success)
    per_condition: dict[str, dict[str, float | int]] = {}
    for cid in condition_ids:
        rows = [r for r in results if r.condition_id == cid]
        n = len(rows)
        ns = sum(1 for r in rows if r.success)
        per_condition[cid] = {"n": n, "n_success": ns, "pc_success": _pc_success(ns, n)}
    return {
        "pc_success": _pc_success(n_success, n_trials),
        "n_trials": n_trials,
        "n_success": n_success,
        "per_condition": per_condition,
    }


def run_protocol(
    *,
    protocol: Protocol,
    follower: Any,
    policy: Any,
    camera_name: str,
    fps: int,
    run_dir: Path,
    keys: KeySource,
    n_trials: int | None = None,
    reset_confirm: Callable[[Condition, int, int], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.perf_counter,
    max_steps: int | None = None,
) -> dict[str, Any]:
    n = protocol.n_trials_per_condition if n_trials is None else int(n_trials)
    if n < 1:
        raise ValueError("n_trials must be >= 1")
    effective = replace(protocol, n_trials_per_condition=n)
    confirm = reset_confirm if reset_confirm is not None else _default_reset_confirm
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol.json").write_text(json.dumps(effective.to_dict(), indent=2) + "\n")
    print(f"Task: {effective.task}", flush=True)
    print(f"Success: {effective.success_criterion}", flush=True)
    print(f"Timeout {effective.timeout_s}s  {n} trials × {len(effective.conditions)} conditions", flush=True)

    results: list[TrialResult] = []
    trial_index = 0
    for condition in effective.conditions:
        for trial_i in range(n):
            trial_index += 1
            confirm(condition, trial_i, n)
            scored: list[bool] = []

            def should_stop() -> bool:
                verdict = score_from_key(keys.poll())
                if verdict is None:
                    return False
                scored.append(verdict)
                return True

            steps = run_eval_loop(
                follower=follower,
                policy=policy,
                camera_name=camera_name,
                fps=fps,
                duration_s=effective.timeout_s,
                max_steps=max_steps,
                sleep=sleep,
                now=now,
                should_stop=should_stop,
            )
            early_stop = bool(scored)
            success = scored[0] if scored else _wait_for_score(keys, sleep)
            result = TrialResult(
                trial_index=trial_index,
                condition_id=condition.id,
                success=success,
                steps=steps,
                early_stop=early_stop,
                timeout_s=effective.timeout_s,
            )
            results.append(result)
            trial_dir = run_dir / "trials" / f"{trial_index:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "condition.json").write_text(json.dumps(asdict(condition), indent=2) + "\n")
            (trial_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
            mark = "SUCCESS" if success else "FAIL"
            print(
                f"Trial {trial_index:04d} {condition.id} {mark}  steps={steps} early_stop={early_stop}",
                flush=True,
            )

    summary = summarize(results, [c.id for c in effective.conditions])
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"pc_success={summary['pc_success']}  {summary['n_success']}/{summary['n_trials']}  logs={run_dir}",
        flush=True,
    )
    return summary
