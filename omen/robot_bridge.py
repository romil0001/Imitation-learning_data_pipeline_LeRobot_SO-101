"""Local follower bridge on the Omen: publish joint state, apply Mac actions.

Mac teleop/eval keeps the leader USB; the follower USB plugs into this PC.

  PUB tcp://0.0.0.0:5556  {"state": [6 floats], "ts": ...}
  SUB tcp://0.0.0.0:5557  {"action": [6 floats]} or {"name.pos": float, ...}

Usage (Anaconda Prompt, omen-d405 or lerobot env with feetech):
  python omen/robot_bridge.py --port COM5
  python omen/robot_bridge.py --port COM5 --state-port 5556 --cmd-port 5557
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-101 follower ZMQ bridge (Omen)")
    p.add_argument("--port", required=True, help="Follower serial port (e.g. COM5)")
    p.add_argument("--robot-id", default="my_follower_arm")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--state-port", type=int, default=5556)
    p.add_argument("--cmd-port", type=int, default=5557)
    p.add_argument("--hz", type=float, default=30.0, help="State publish / control rate")
    p.add_argument("--p", type=int, default=10)
    p.add_argument("--i", type=int, default=0)
    p.add_argument("--d", type=int, default=48)
    p.add_argument("--retries", type=int, default=8)
    return p.parse_args()


def qpos_from_obs(obs: dict[str, Any]) -> list[float]:
    return [float(obs[f"{name}.pos"]) for name in JOINT_NAMES]


def action_from_msg(data: dict[str, Any]) -> dict[str, float] | None:
    if "action" in data:
        q = np.asarray(data["action"], dtype=np.float32).reshape(-1)
        if q.shape[0] != len(JOINT_NAMES):
            return None
        return {f"{name}.pos": float(q[i]) for i, name in enumerate(JOINT_NAMES)}
    if all(f"{name}.pos" in data for name in JOINT_NAMES):
        return {f"{name}.pos": float(data[f"{name}.pos"]) for name in JOINT_NAMES}
    return None


def make_local_follower(args: argparse.Namespace) -> Any:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cfg = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        cameras={},
        position_p_coefficient=args.p,
        position_i_coefficient=args.i,
        position_d_coefficient=args.d,
        num_read_retries=args.retries,
    )
    return SO101Follower(cfg)


def main() -> int:
    import zmq

    args = parse_args()
    dt = 1.0 / max(args.hz, 1.0)
    print(f"Follower bridge  port={args.port}  id={args.robot_id}", flush=True)
    print(f"PUB state tcp://{args.bind}:{args.state_port}", flush=True)
    print(f"SUB cmds  tcp://{args.bind}:{args.cmd_port}", flush=True)
    print("Mac: FOLLOWER_BACKEND=remote  ROBOT_HOST=<this LAN IP>", flush=True)

    try:
        follower = make_local_follower(args)
    except Exception as e:
        print(f"Failed to create follower: {e}", file=sys.stderr)
        print("Install lerobot[feetech] in this env and check the COM port.", file=sys.stderr)
        return 1

    # Skip interactive LeRobot calibrate() prompt on the Omen bridge.
    # Run a full calibrate once on this machine if joint zeros look wrong.
    follower.connect(calibrate=False)
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{args.bind}:{args.state_port}")
    sub = ctx.socket(zmq.SUB)
    sub.bind(f"tcp://{args.bind}:{args.cmd_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, 0)
    sub.setsockopt(zmq.CONFLATE, True)

    n = 0
    t0 = time.time()
    try:
        while True:
            loop_t = time.perf_counter()
            try:
                raw = sub.recv_string(zmq.NOBLOCK)
                msg = json.loads(raw)
                action = action_from_msg(msg)
                if action is not None:
                    follower.send_action(action)
            except zmq.Again:
                pass
            except Exception as exc:
                if n % 50 == 0:
                    print(f"cmd decode/send: {exc}", flush=True)

            obs = follower.get_observation()
            state = qpos_from_obs(obs)
            pub.send_string(json.dumps({"state": state, "ts": time.time()}))
            n += 1
            if n % 60 == 0:
                dt_s = time.time() - t0
                print(f"bridge {n} ticks ({n / max(dt_s, 1e-3):.1f} Hz) state[0]={state[0]:.2f}", flush=True)
            leftover = dt - (time.perf_counter() - loop_t)
            if leftover > 0:
                time.sleep(leftover)
    except KeyboardInterrupt:
        print("\nBridge stopped — holding last pose then disconnect.", flush=True)
    finally:
        try:
            obs = follower.get_observation()
            follower.send_action({f"{n}.pos": float(obs[f"{n}.pos"]) for n in JOINT_NAMES})
            time.sleep(0.2)
        except Exception:
            pass
        try:
            follower.disconnect()
        except Exception as exc:
            print(f"disconnect: {exc}", flush=True)
        pub.close(0)
        sub.close(0)
        ctx.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
