@echo off
cd /d "%~dp0\.."
call conda activate omen-d405
REM Single D405 (wrist) + USB overhead (legacy):
REM   python omen\publish_d405.py --camera-name wrist --overhead-name overhead --webcam-index 2
REM Dual D405 (wrist+base RGB+depth) + overhead — set serials from --list-realsense:
REM   python omen\publish_d405.py --list-realsense
REM   python omen\publish_d405.py --wrist-serial SERIAL1 --base-serial SERIAL2 --webcam-index 2
REM XWF-1080P is usually index 2 on this Omen (0=laptop, 1=empty). Depth enabled by default.
if "%WRIST_SERIAL%"=="" (
  python omen\publish_d405.py --camera-name wrist --overhead-name overhead --webcam-index 2 %*
) else (
  python omen\publish_d405.py --wrist-serial %WRIST_SERIAL% --base-serial %BASE_SERIAL% --webcam-index 2 %*
)
