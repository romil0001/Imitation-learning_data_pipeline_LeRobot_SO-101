@echo off
setlocal
cd /d "%~dp0"
echo Enabling OpenSSH Server (needs Admin).
echo If you cannot elevate, use the no-admin path instead:
echo   1^) On Mac:  python scripts/omen_broker.py
echo   2^) On Omen: omen\start_remote_agent.bat
echo   3^) On Mac:  python scripts/omen_run.py "git status"
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_openssh.ps1"
if errorlevel 1 (
  echo.
  echo Admin install failed — use omen\start_remote_agent.bat ^(no Admin^).
  exit /b 1
)
pause
