@echo off
setlocal EnableExtensions
REM Allow Mac subscribers to reach the ZMQ publisher. Run as Administrator once.

net session >nul 2>&1
if errorlevel 1 (
  echo Run this script as Administrator ^(right-click → Run as administrator^).
  exit /b 1
)

netsh advfirewall firewall show rule name="SO101 Omen ZMQ 5555" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  echo Firewall rule "SO101 Omen ZMQ 5555" already exists.
  exit /b 0
)

netsh advfirewall firewall add rule name="SO101 Omen ZMQ 5555" dir=in action=allow protocol=TCP localport=5555 profile=any
if errorlevel 1 (
  echo Failed to add firewall rule.
  exit /b 1
)
echo Added inbound TCP 5555 rule: SO101 Omen ZMQ 5555
echo Mac config/camera.env ZMQ_HOST must be this PC's LAN IPv4 ^(omen\show_lan_ip.bat^).
exit /b 0
