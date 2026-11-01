@echo off
setlocal EnableExtensions
REM Print IPv4 addresses for Mac config/camera.env ZMQ_HOST.

echo IPv4 addresses ^(skip 127.0.0.1 and virtual adapters if unsure^):
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
  for /f "tokens=* delims= " %%b in ("%%a") do echo   %%b
)
echo.
echo On the Mac: copy config/camera.env.example → config/camera.env
echo and set ZMQ_HOST to the LAN address that is on the same Wi-Fi as the Mac.
