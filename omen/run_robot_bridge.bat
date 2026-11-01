@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

call conda activate omen-d405

REM Load local COM / camera IDs (must be a .bat — cmd does not reliably call .env)
if exist "%~dp0camera_serials.bat" call "%~dp0camera_serials.bat"

REM Default for this Omen if serials file missing
if not defined FOLLOWER_COM set "FOLLOWER_COM=COM3"

echo Follower bridge on %FOLLOWER_COM%
python omen\robot_bridge.py --port %FOLLOWER_COM% %*
exit /b %ERRORLEVEL%
