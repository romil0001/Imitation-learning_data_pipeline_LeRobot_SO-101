@echo off
cd /d "%~dp0\.."
call conda activate omen-d405
REM Loads FOLLOWER_COM from omen\camera_serials.env when present.
if exist "%~dp0camera_serials.env" call "%~dp0camera_serials.env"
REM Follower USB must be on THIS PC (Device Manager → Ports).
if "%FOLLOWER_COM%"=="" (
  echo Set FOLLOWER_COM to the Windows COM port, e.g. set FOLLOWER_COM=COM3
  echo Or add it to omen\camera_serials.env, then re-run omen\run_robot_bridge.bat
  exit /b 1
)
python omen\robot_bridge.py --port %FOLLOWER_COM% %*
