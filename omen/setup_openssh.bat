@echo off
setlocal
cd /d "%~dp0"
echo Enabling OpenSSH Server (may prompt for Admin)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_openssh.ps1"
if errorlevel 1 (
  echo.
  echo If install failed, right-click this bat -^> Run as administrator.
  exit /b 1
)
pause
