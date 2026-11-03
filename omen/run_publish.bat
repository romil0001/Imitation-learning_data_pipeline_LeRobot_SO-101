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
  echo ERROR: WRIST_SERIAL unset. Edit omen\camera_serials.bat ^(python omen\publish_d405.py --list-realsense^).
  exit /b 1
)
if "%BASE_SERIAL%"=="" (
  echo ERROR: BASE_SERIAL unset. Dashboard needs a base stream. Edit omen\camera_serials.bat.
  echo If wrist/base views are swapped, swap the two serial values and restart.
  exit /b 1
)
if /I "%WRIST_SERIAL%"=="%BASE_SERIAL%" (
  echo ERROR: WRIST_SERIAL and BASE_SERIAL are the same ^(%WRIST_SERIAL%^). They must differ.
  exit /b 1
)
if defined BASE_DEPTH_FLAG (set "BASE_DEPTH_MSG=off") else (set "BASE_DEPTH_MSG=on")
echo Using wrist=%WRIST_SERIAL% base=%BASE_SERIAL% webcam=%WEBCAM_INDEX% base_depth=%BASE_DEPTH_MSG%
echo Resilient PUB: if one camera drops, others keep streaming; reconnect is automatic.
python omen\publish_d405.py --wrist-serial %WRIST_SERIAL% --base-serial %BASE_SERIAL% --webcam-index %WEBCAM_INDEX% %BASE_DEPTH_FLAG% %*
