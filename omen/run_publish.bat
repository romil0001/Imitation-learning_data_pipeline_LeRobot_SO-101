@echo off
cd /d "%~dp0\.."
call conda activate omen-d405
REM Loads omen\camera_serials.env when present (WRIST_SERIAL / BASE_SERIAL / WEBCAM_INDEX).
if exist "%~dp0camera_serials.env" call "%~dp0camera_serials.env"
REM Dual D405 (wrist+base RGB+depth) + USB overhead. XWF-1080P is usually index 2.
if "%WEBCAM_INDEX%"=="" set WEBCAM_INDEX=2
REM --base-no-depth keeps wrist metric depth while fitting a 2nd D405 on the same USB root.
if "%WRIST_SERIAL%"=="" (
  python omen\publish_d405.py --camera-name wrist --overhead-name overhead --webcam-index %WEBCAM_INDEX% %*
) else (
  python omen\publish_d405.py --wrist-serial %WRIST_SERIAL% --base-serial %BASE_SERIAL% --webcam-index %WEBCAM_INDEX% --base-no-depth %*
)
