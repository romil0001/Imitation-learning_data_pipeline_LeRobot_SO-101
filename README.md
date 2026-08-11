# LeRobot SO-101 Bring-up (Mac)

Leader + follower SO-101 with **12V STS3215** servos (6 each) and 2 bus drivers.

Official guide: https://huggingface.co/docs/lerobot/en/so101

## Danger
- Use **12V** power supplies only for these servos.
- During motor ID and calibration, power **one arm at a time**.
- Teleop is the only step that needs **both** arms powered.

## Quick start
See `docs/checklist.md`, then run scripts in numeric order under `scripts/`.

1. `bash env/setup_env.sh`
2. Label boards/PSUs FOLLOWER vs LEADER
3. `bash scripts/01_find_port.sh` (saves `config/ports.env`)
4. Follower motors only → `bash scripts/02_setup_motors_follower.sh`
5. Leader motors only → `bash scripts/03_setup_motors_leader.sh`
6. Calibrate each arm (`04` then `05`)
7. Both powered → `bash scripts/06_teleoperate.sh`
