@echo off
setlocal
cd /d "%~dp0\.."

where conda >nul 2>&1
if errorlevel 1 (
  echo Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html
  exit /b 1
)

call conda.bat env list | findstr /r /c:"^omen-d405 " >nul
if errorlevel 1 (
  echo Creating conda env omen-d405 ...
  call conda create -y -n omen-d405 python=3.12
)

call conda activate omen-d405
pip install -r omen\requirements.txt

echo.
echo "Setup complete. Plug D405 (wrist) into USB 3 and the USB webcam (overhead), then:"
echo "  conda activate omen-d405"
echo "  python omen\publish_d405.py --list-webcams"
echo "  omen\run_publish.bat"
echo "If overhead is the laptop camera, rerun with --webcam-index 2 (skip the 640x480 D405 index)"
echo.
echo Allow port 5555 in Windows Firewall if the Mac cannot connect.
echo Set ZMQ_HOST on the Mac to this PC's IPv4 address (ipconfig).
