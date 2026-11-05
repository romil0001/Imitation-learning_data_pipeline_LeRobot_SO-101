"""List checkpoints / protocols and launch so101 eval from the dashboard."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

POLICIES = ("act", "act-jax", "smolvla", "hybrid", "gemfuse")
POLICY_BASES = ("smolvla", "act", "act-jax")


@dataclass
class EvalProcess:
    pid: int
    cmd: list[str]
    started_at: float
    policy: str
    checkpoint: str
    mode: str
    popen: subprocess.Popen | None = field(default=None, repr=False)


_lock = threading.Lock()
_current: EvalProcess | None = None


def _checkpoint_kind(run: Path) -> str:
    if (run / "best_val" / "model.pt").is_file() or (run / "best_val" / "model.msgpack").is_file():
        return "best_val"
    if (run / "last" / "model.pt").is_file() or (run / "last" / "model.msgpack").is_file():
        return "last"
    if (run / "model.pt").is_file() or (run / "model.msgpack").is_file():
        return "legacy"
    return "none"


def list_checkpoints(root: Path) -> list[dict]:
    train = root / "outputs" / "train"
    if not train.is_dir():
        return []
    out: list[dict] = []
    for run in sorted(p for p in train.iterdir() if p.is_dir()):
        kind = _checkpoint_kind(run)
        if kind == "none":
            continue
        weights = []
        if (run / "best_val").is_dir():
            weights.append("best_val")
        if (run / "last").is_dir():
            weights.append("last")
        if kind == "legacy":
            weights.append("legacy")
        out.append(
            {
                "name": run.name,
                "path": str(run),
                "kind": kind,
                "weights": weights,
                "label": f"{run.name} [{kind}]",
            }
        )
    return out


def list_protocols(root: Path) -> list[dict]:
    proto_dir = root / "config" / "eval"
    if not proto_dir.is_dir():
        return []
    return [
        {"name": p.name, "path": str(p)}
        for p in sorted(proto_dir.glob("*.yaml"))
        if p.is_file()
    ]


def resolve_checkpoint(run_path: str, weights: str = "best_val") -> Path:
    run = Path(run_path).expanduser().resolve()
    if not run.exists():
        raise FileNotFoundError(f"checkpoint not found: {run}")
    if run.is_file():
        return run
    if weights == "last" and (run / "last").is_dir():
        return run / "last"
    if weights == "best_val" and (run / "best_val").is_dir():
        # Keep run root so so101 prefers best_val; also accept explicit best_val/
        return run
    if weights == "legacy":
        return run
    return run


def _status_unlocked() -> dict:
    global _current
    cur = _current
    if cur is None:
        return {"running": _pgrep_eval(), "pid": None, "info": None}
    alive = cur.popen is not None and cur.popen.poll() is None
    if not alive:
        _current = None
        return {"running": _pgrep_eval(), "pid": None, "info": None}
    return {
        "running": True,
        "pid": cur.pid,
        "info": {
            "policy": cur.policy,
            "checkpoint": cur.checkpoint,
            "mode": cur.mode,
            "started_at": cur.started_at,
            "elapsed_s": round(time.time() - cur.started_at, 1),
            "cmd": cur.cmd,
        },
    }


def eval_status() -> dict:
    with _lock:
        return _status_unlocked()


def _pgrep_eval() -> bool:
    try:
        r = subprocess.run(
            ["pgrep", "-f", "so101 eval"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return bool(r.stdout.strip())


def start_eval(
    root: Path,
    *,
    policy: str = "act",
    checkpoint: str,
    weights: str = "best_val",
    dataset: str = "so101_bag_clip_bowl",
    mode: str = "protocol",
    duration: float = 60.0,
    protocol: str = "",
    policy_base: str = "smolvla",
    grasp_config: str = "",
    task: str = "",
    n_trials: int | None = None,
    n_action_steps: int | None = 25,
    recipe: str = "",
) -> dict:
    global _current

    policy = (policy or "act").strip().lower()
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy {policy!r}; choose from {POLICIES}")

    ckpt = resolve_checkpoint(checkpoint, weights=weights)
    mode = (mode or "protocol").strip().lower()
    if mode not in {"protocol", "timed"}:
        raise ValueError("mode must be 'protocol' or 'timed'")

    with _lock:
        st = _status_unlocked()
        if st.get("running") and _current is not None and _current.popen and _current.popen.poll() is None:
            raise RuntimeError(f"eval already running (pid={_current.pid})")
        if _pgrep_eval():
            raise RuntimeError("an external so101 eval process is already running")

        py = sys.executable
        cmd = [
            py,
            "-m",
            "so101",
            "eval",
            f"--dataset={dataset}",
            f"--checkpoint={ckpt}",
            f"--policy={policy}",
            "--yes",
        ]
        if policy == "hybrid":
            cmd.append(f"--policy-base={policy_base or 'smolvla'}")
            grasp = grasp_config or str(root / "config" / "hybrid_grasp.yaml")
            cmd.append(f"--grasp-config={grasp}")
        if task:
            cmd.append(f"--task={task}")
        # Default closed-loop: replan every step (rectify mistakes immediately).
        if n_action_steps is not None:
            cmd.append(f"--n-action-steps={int(n_action_steps)}")
        recipe_path = (recipe or "").strip()
        if not recipe_path and n_action_steps == 1:
            candidate = root / "config" / "recipes" / "reactive_act1.yaml"
            if candidate.is_file():
                recipe_path = str(candidate)
        elif not recipe_path and n_action_steps == 25:
            candidate = root / "config" / "recipes" / "act25.yaml"
            if candidate.is_file():
                recipe_path = str(candidate)
        elif not recipe_path and n_action_steps == 200:
            candidate = root / "config" / "recipes" / "act200.yaml"
            if candidate.is_file():
                recipe_path = str(candidate)
        if recipe_path:
            cmd.append(f"--recipe={recipe_path}")

        if mode == "protocol":
            proto = protocol or str(root / "config" / "eval" / "bag_clip_in_bowl.yaml")
            if not Path(proto).is_file():
                raise FileNotFoundError(f"protocol not found: {proto}")
            cmd.append(f"--protocol={proto}")
            if n_trials is not None:
                cmd.append(f"--n-trials={int(n_trials)}")
            mode_label = f"protocol:{proto}"
        else:
            cmd.append(f"--duration={float(duration)}")
            mode_label = f"timed:{duration}s"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        session_log = root / "outputs" / "session.log"
        session_log.parent.mkdir(parents=True, exist_ok=True)

        log_f = open(session_log, "a", encoding="utf-8")  # noqa: SIM115 — kept open for subprocess lifetime
        log_f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} eval =====\n")
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.flush()

        popen = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _current = EvalProcess(
            pid=popen.pid,
            cmd=cmd,
            started_at=time.time(),
            policy=policy,
            checkpoint=str(ckpt),
            mode=mode_label,
            popen=popen,
        )
        threading.Thread(
            target=_wait_and_close,
            args=(popen, log_f),
            daemon=True,
        ).start()
        return {
            "ok": True,
            "pid": popen.pid,
            "cmd": cmd,
            "checkpoint": str(ckpt),
            "policy": policy,
            "mode": mode_label,
        }


def _wait_and_close(popen: subprocess.Popen, log_f) -> None:
    global _current
    try:
        popen.wait()
    finally:
        try:
            log_f.close()
        except OSError:
            pass
        with _lock:
            if _current is not None and _current.pid == popen.pid:
                _current = None


def stop_eval() -> dict:
    global _current
    with _lock:
        cur = _current
        if cur is None or cur.popen is None:
            if _pgrep_eval():
                raise RuntimeError("eval is running outside the dashboard; stop it from that terminal")
            return {"ok": True, "stopped": False, "hint": "no dashboard-launched eval"}
        try:
            os.killpg(cur.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            cur.popen.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(cur.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _current = None
        return {"ok": True, "stopped": True, "pid": cur.pid}


def options_payload(root: Path) -> dict:
    default_ds = "so101_bag_clip_bowl"
    default_proto = root / "config" / "eval" / "bag_clip_in_bowl.yaml"
    return {
        "policies": list(POLICIES),
        "policy_bases": list(POLICY_BASES),
        "checkpoints": list_checkpoints(root),
        "protocols": list_protocols(root),
        "defaults": {
            "policy": "act",
            "dataset": default_ds,
            "mode": "protocol",
            "duration": 60,
            "protocol": str(default_proto) if default_proto.is_file() else "",
            "weights": "best_val",
            "policy_base": "smolvla",
            "n_action_steps": 25,
            "control_mode": "act25",
        },
        "status": eval_status(),
    }
