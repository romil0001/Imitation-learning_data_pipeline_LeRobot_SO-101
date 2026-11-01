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
REM Dual D405 depth on. If USB bandwidth drops frames, set BASE_NO_DEPTH=1 before launch.
set "BASE_DEPTH_FLAG="
if /I "%BASE_NO_DEPTH%"=="1" set "BASE_DEPTH_FLAG=--base-no-depth"
if /I "%BASE_NO_DEPTH%"=="true" set "BASE_DEPTH_FLAG=--base-no-depth"
if "%WRIST_SERIAL%"=="" (
  echo WARNING: WRIST_SERIAL unset — using first RealSense. Create omen\camera_serials.bat
  python omen\publish_d405.py --camera-name wrist --overhead-name overhead --webcam-index %WEBCAM_INDEX% %*
) else (
  if defined BASE_DEPTH_FLAG (set "BASE_DEPTH_MSG=off") else (set "BASE_DEPTH_MSG=on")
  echo Using wrist=%WRIST_SERIAL% base=%BASE_SERIAL% webcam=%WEBCAM_INDEX% base_depth=%BASE_DEPTH_MSG%
  python omen\publish_d405.py --wrist-serial %WRIST_SERIAL% --base-serial %BASE_SERIAL% --webcam-index %WEBCAM_INDEX% %BASE_DEPTH_FLAG% %*
)
