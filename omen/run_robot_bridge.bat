@echo off
cd /d "%~dp0\.."
call conda activate omen-d405
REM Follower USB must be on THIS PC. Find COM port in Device Manager.
REM Example:
REM   set FOLLOWER_COM=COM5
REM   omen\run_robot_bridge.bat
if "%FOLLOWER_COM%"=="" (
  echo Set FOLLOWER_COM to the Windows COM port, e.g. set FOLLOWER_COM=COM5
  echo Then re-run omen\run_robot_bridge.bat
  exit /b 1
)
python omen\robot_bridge.py --port %FOLLOWER_COM% %*
