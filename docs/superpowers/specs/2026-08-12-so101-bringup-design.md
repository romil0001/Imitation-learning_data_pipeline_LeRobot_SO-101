# SO-101 Leader/Follower Bring-up Design

**Date:** 2026-08-12  
**Status:** Approved for implementation planning  
**Hardware:** 2× bus drivers, follower 6× STS3215 (12V / 30 kg), leader 6× STS3215 (12V / 30 kg)  
**Host:** macOS  
**Scope:** Bring-up only (install → ports → motor IDs → calibrate → teleop)

## Goal

Create a local Mac project that follows the official Hugging Face [SO-101 guide](https://huggingface.co/docs/lerobot/en/so101) so a first-time user can bring up leader and follower arms without mixing ports, power, or calibration IDs.

## Constraints

- Both arms use **12V** STS3215 servos — power each MotorsBus with a **12V** supply only.
- Hardware is not connected yet; software and scripts must be ready before first plug-in.
- Leader and follower must stay clearly separated at every step (labels, script names, banners, env vars, robot/teleop ids).
- Cameras, dataset recording, training, and evaluation are **out of scope** for this pass.

## Approach

Local conda environment + Hugging Face `lerobot[feetech]` + numbered shell scripts that wrap the official CLI, with `config/ports.env` storing distinct leader/follower serial ports.

## Project layout

```
Lerobot setup/
  README.md
  .gitignore
  env/
    setup_env.sh
  config/
    ports.env.example
    ports.env                 # gitignored; filled after find-port
  scripts/
    01_find_port.sh
    02_setup_motors_follower.sh
    03_setup_motors_leader.sh
    04_calibrate_follower.sh
    05_calibrate_leader.sh
    06_teleoperate.sh
    _common.sh                # load ports.env, validate, print banners
  third_party/
    lerobot/                  # cloned by setup_env.sh
  docs/
    checklist.md
    superpowers/specs/
      2026-08-12-so101-bringup-design.md
```

## Power and identity rules

| Step | Follower bus + 12V | Leader bus + 12V | Notes |
|------|--------------------|------------------|-------|
| Find follower port | ON | OFF / unplugged | Save `FOLLOWER_PORT` |
| Find leader port | OFF / unplugged | ON | Save `LEADER_PORT` |
| Setup motors follower | ON | OFF | `so101_follower` |
| Setup motors leader | OFF | ON | `so101_leader` |
| Calibrate follower | ON | OFF | `--robot.id=my_follower_arm` |
| Calibrate leader | OFF | ON | `--teleop.id=my_leader_arm` |
| Teleoperate | ON | ON | Both ports from `ports.env` |

Physical stickers: label each board and PSU **FOLLOWER** or **LEADER**.

## Bring-up flow

1. **Environment** — `env/setup_env.sh` creates conda env `lerobot` (Python 3.12), clones Hugging Face `lerobot` tag `v0.6.1` into `third_party/lerobot` (gitignored), and `pip install -e ".[feetech,hardware]"` from that checkout.
2. **Find ports** — `lerobot-find-port` via `01_find_port.sh`, run twice (once per arm). Write `FOLLOWER_PORT` and `LEADER_PORT` into `config/ports.env` (Mac paths like `/dev/tty.usbmodem…`).
3. **Motor IDs** — one motor daisy-chain connection at a time per official guide; follower script then leader script; never both boards powered.
4. **Calibrate** — mid-range pose, then full joint sweep; separate calibration files via distinct ids.
5. **Teleop** — both arms powered; leader motion mirrors on follower.

## Script behavior

- `_common.sh` sources `ports.env`, requires both ports set and **not equal**, activates conda env when needed.
- Follower/leader scripts print a clear banner stating which arm must be powered and which must be unplugged before continuing.
- Scripts wrap official commands (`lerobot-find-port`, `lerobot-setup-motors`, `lerobot-calibrate`, `lerobot-teleoperate`) rather than reimplementing Feetech protocol.

## Success criteria

- Conda env installs on this Mac.
- Leader and follower ports are saved separately and validated.
- Motor IDs configured on both arms.
- Both arms calibrated with distinct ids.
- Teleop: moving the leader moves the follower.

## Out of scope

- Cameras / OpenCV viewing
- `lerobot-record` / datasets
- Policy training and evaluation
- Linux udev rules
- Changing gear-ratio assumptions in LeRobot (user has uniform 12V STS3215 on both arms)

## References

- https://huggingface.co/docs/lerobot/en/so101
- https://github.com/huggingface/lerobot
