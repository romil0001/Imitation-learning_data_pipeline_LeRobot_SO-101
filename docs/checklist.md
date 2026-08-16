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

## Two machines
- [ ] **Mac:** leader + follower USB (motor boards can share USB 2)
- [ ] **HP Omen:** D405 on a USB 3 SuperSpeed port (not a USB 2 hub)
- [ ] Same Wi-Fi/LAN; Omen `ipconfig` IPv4 written into Mac `config/camera.env` as `ZMQ_HOST`
- [ ] Windows Firewall allows inbound TCP **5555**
- [ ] Do **not** train or run full LeRobot on the 4 GB Omen

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
3. D405 plugged into USB 3; RealSense Viewer optional sanity check
4. `omen\run_publish.bat` (leave it running)

## Session
1. Mac: `bash scripts/09_dashboard.sh` → http://127.0.0.1:8080
2. Mac: `bash scripts/06_teleoperate.sh`
3. Mac: `bash scripts/08_record.sh` (RGB demos; optional Omen `--dump-dir` for depth PNGs)
