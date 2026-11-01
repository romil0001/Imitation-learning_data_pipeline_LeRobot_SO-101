@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

where conda >nul 2>&1
if errorlevel 1 (
  echo conda not on PATH. Open Anaconda Prompt or re-run omen\setup_omen.bat
  exit /b 1
)

REM XWF-1080P is usually index 2 on this Omen (0=laptop, 1=empty). Depth on by default.
REM --no-capture-output keeps the live fps lines visible in this window.
call conda run -n omen-d405 --no-capture-output python omen\publish_d405.py --camera-name wrist --overhead-name overhead --webcam-index 2 %*
exit /b %ERRORLEVEL%
