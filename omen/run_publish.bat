@echo off
cd /d "%~dp0\.."
call conda activate omen-d405
python omen\publish_d405.py --camera-name wrist --overhead-name overhead --webcam-index 0 --no-depth %*
