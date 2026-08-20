# SO-101 Physical Checklist

## Stickers
- [ ] Label bus board A **FOLLOWER**
- [ ] Label bus board B **LEADER**
- [ ] Label 12V PSU A **FOLLOWER**
- [ ] Label 12V PSU B **LEADER**

## Power rules (12V STS3215 only — never 5V)
| Step | Follower USB+12V | Leader USB+12V |
|------|------------------|----------------|
| Find follower port | ON | OFF |
| Find leader port | OFF | ON |
| Setup motors follower | ON | OFF |
| Setup motors leader | OFF | ON |
| Calibrate follower | ON | OFF |
| Calibrate leader | OFF | ON |
| Teleoperate / record | ON | ON |
| Train ACT | OFF (no robot) | OFF |
| Eval ACT | ON | OFF |

## Two machines
- [ ] **Mac:** leader + follower USB (motor boards can share USB 2)
- [ ] **HP Omen:** D405 (wrist) on USB 3; USB webcam (overhead)
- [ ] Same Wi-Fi/LAN; Omen `ipconfig` IPv4 written into Mac `config/camera.env` as `ZMQ_HOST`
- [ ] Mac `config/camera.env` has `CAMERA_NAMES=wrist,overhead` and `CAMERA_USE_DEPTH=true`
- [ ] Windows Firewall allows inbound TCP **5555**
- [ ] Do **not** train on the 4 GB Omen (train ACT on the Mac with MPS)

## Mac software order
1. `bash env/setup_env.sh`
2. `bash scripts/01_find_port.sh`
3. `bash scripts/02_setup_motors_follower.sh`
4. `bash scripts/03_setup_motors_leader.sh`
5. `bash scripts/04_calibrate_follower.sh`
6. `bash scripts/05_calibrate_leader.sh`
7. Copy `config/camera.env.example` → `config/camera.env`, set `ZMQ_HOST`

## Omen software order
1. Miniconda installed
2. `omen\setup_omen.bat`
3. D405 plugged into USB 3 (wrist); USB webcam plugged in (overhead)
4. `python omen\publish_d405.py --list-webcams` then `omen\run_publish.bat` (leave it running)

## Session
1. Mac: `bash scripts/09_dashboard.sh` → http://127.0.0.1:8080 (operator site; **D** = status check)
2. Mac: `bash scripts/06_teleoperate.sh`
3. Mac: `bash scripts/08_record.sh` (~50 episodes for ACT; same cameras/task; data in `data/datasets/`)
4. Mac: `bash scripts/10_train_act.sh` (MPS, hours — not the Omen; holds out episodes, writes `best_val`; depth used automatically if recorded; optional `--backend jax`)
5. Mac: follower + Omen publisher → `bash scripts/11_eval_act.sh` (loads `best_val`; wrist depth fed to policy when trained with depth; scored pen-to-keys grid; reset scene, then **y**/**n** each trial; train loss is not the score; logs in `outputs/eval/`)
