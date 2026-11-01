@echo off
cd /d "%~dp0\.."
call conda activate omen-d405
REM Prefer .bat (cmd call); fall back to legacy .env
if exist "%~dp0camera_serials.bat" (
  call "%~dp0camera_serials.bat"
) else if exist "%~dp0camera_serials.env" (
  call "%~dp0camera_serials.env"
)
if "%WEBCAM_INDEX%"=="" set WEBCAM_INDEX=2
REM --base-no-depth keeps wrist metric depth while fitting a 2nd D405 on the same USB root.
if "%WRIST_SERIAL%"=="" (
  echo WARNING: WRIST_SERIAL unset — using first RealSense. Create omen\camera_serials.bat
  python omen\publish_d405.py --camera-name wrist --overhead-name overhead --webcam-index %WEBCAM_INDEX% %*
) else (
  echo Using wrist=%WRIST_SERIAL% base=%BASE_SERIAL% webcam=%WEBCAM_INDEX%
  python omen\publish_d405.py --wrist-serial %WRIST_SERIAL% --base-serial %BASE_SERIAL% --webcam-index %WEBCAM_INDEX% --base-no-depth %*
)
