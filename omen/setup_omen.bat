@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

where conda >nul 2>&1
if errorlevel 1 (
  echo Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html
  echo Then open "Anaconda Prompt" ^(or a new cmd^) so conda is on PATH, and re-run:
  echo   omen\setup_omen.bat
  exit /b 1
)

call conda.bat env list 2>nul | findstr /i /c:"omen-d405" >nul
if errorlevel 1 (
  echo Creating conda env omen-d405 ...
  call conda create -y -n omen-d405 python=3.12
  if errorlevel 1 (
    echo conda create failed.
    exit /b 1
  )
) else (
  echo Conda env omen-d405 already exists.
)

echo Installing packages into omen-d405 ...
call conda run -n omen-d405 python -m pip install -r omen\requirements.txt
if errorlevel 1 (
  echo pip install failed.
  exit /b 1
)

echo.
echo Verifying Python imports ...
call conda run -n omen-d405 python -c "import cv2, numpy, zmq, pyrealsense2 as rs; print('imports ok', 'cv2', cv2.__version__)"
if errorlevel 1 (
  echo.
  echo Import check failed. On Windows also install Intel RealSense SDK USB drivers:
  echo   https://github.com/IntelRealSense/librealsense/releases
  echo Install Intel.RealSense.SDK-*.exe, reboot if prompted, then re-run this script.
  exit /b 1
)

echo.
echo === LAN IPv4 ^(put one in Mac config/camera.env as ZMQ_HOST^) ===
call "%~dp0show_lan_ip.bat"

echo.
echo Running hardware self-check ^(D405 + webcams + ZMQ port^) ...
call conda run -n omen-d405 --no-capture-output python omen\publish_d405.py --self-check
set CHECK_ERR=%ERRORLEVEL%

echo.
echo --------------------------------------------------------------------
echo Setup packages: OK
if %CHECK_ERR% NEQ 0 (
  echo Hardware self-check: FAILED — plug D405 USB 3 + overhead webcam, then:
  echo   conda run -n omen-d405 --no-capture-output python omen\publish_d405.py --self-check
) else (
  echo Hardware self-check: OK
)
echo.
echo Next steps:
echo   1. Admin once:  omen\open_firewall.bat
echo   2. Cameras:     python omen\publish_d405.py --list-webcams
echo   3. Stream:      omen\run_publish.bat
echo   4. Mac:         set ZMQ_HOST to an IPv4 printed above in config/camera.env
echo --------------------------------------------------------------------
exit /b %CHECK_ERR%
