@echo off
cd /d "%~dp0\.."
call conda activate omen-d405
REM XWF-1080P is usually index 2 on this Omen (0=laptop, 1=empty). Depth enabled by default.
python omen\publish_d405.py --camera-name wrist --overhead-name overhead --webcam-index 2 %*
