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
| Teleoperate | ON | ON |

## Software order
1. `bash env/setup_env.sh`
2. `bash scripts/01_find_port.sh`
3. `bash scripts/02_setup_motors_follower.sh`
4. `bash scripts/03_setup_motors_leader.sh`
5. `bash scripts/04_calibrate_follower.sh`
6. `bash scripts/05_calibrate_leader.sh`
7. `bash scripts/06_teleoperate.sh`
