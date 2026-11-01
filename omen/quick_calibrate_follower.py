"""Non-interactive follower calibration for the Omen bridge.

Assumes the arm is already near mid-range (safe rest pose). Sets half-turn
homings from the current pose and uses full STS3215 ranges (0–4095), with
wrist_roll as continuous.

Usage (12V ON, clear workspace):
  conda activate omen-d405
  python omen/quick_calibrate_follower.py --port COM3
"""

from __future__ import annotations

import argparse

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick SO-101 follower calibration (mid-pose)")
    p.add_argument("--port", default="COM3")
    p.add_argument("--robot-id", default="my_follower_arm")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SO101FollowerConfig(port=args.port, id=args.robot_id, cameras={})
    robot = SO101Follower(cfg)
    robot.bus.connect()
    print(f"Connected {args.port} id={args.robot_id}", flush=True)
    print("Using current pose as mid-range. Disable torque and set homings…", flush=True)

    robot.bus.disable_torque()
    for motor in robot.bus.motors:
        robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    homing_offsets = robot.bus.set_half_turn_homings()
    range_mins = {m: 0 for m in robot.bus.motors}
    range_maxes = {m: 4095 for m in robot.bus.motors}

    robot.calibration = {}
    for motor, m in robot.bus.motors.items():
        robot.calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=homing_offsets[motor],
            range_min=range_mins[motor],
            range_max=range_maxes[motor],
        )

    robot.bus.write_calibration(robot.calibration)
    robot._save_calibration()
    print(f"Calibration saved to {robot.calibration_fpath}", flush=True)
    robot.bus.disconnect(disable_torque=True)
    print("Done. Re-run omen\\run_robot_bridge.bat", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
