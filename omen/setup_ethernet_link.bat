@echo off
setlocal EnableExtensions
REM Direct USB-Ethernet to the Mac. Pair with scripts\21_use_ethernet_link.sh
REM Mac = 192.168.2.1   Omen = 192.168.2.2   ZMQ = TCP 5555

set "OMEN_IP=192.168.2.2"
set "MASK=255.255.255.0"
set "IFACE=%ETH_IFACE%"
if "%IFACE%"=="" set "IFACE=Ethernet"

echo SO101 Ethernet link: this PC -^> %OMEN_IP%  (adapter "%IFACE%")
echo If the name is wrong, run:  netsh interface show interface
echo Then:  set ETH_IFACE=Your Name ^& omen\setup_ethernet_link.bat
echo.

net session >nul 2>&1
if errorlevel 1 (
  echo ERROR: Run this .bat as Administrator ^(right-click ^> Run as administrator^).
  exit /b 1
)

echo Setting static IPv4 on "%IFACE%" ...
netsh interface ip set address name="%IFACE%" static %OMEN_IP% %MASK%
if errorlevel 1 (
  echo netsh failed. List adapters with: netsh interface show interface
  exit /b 1
)

echo Allowing inbound TCP 5555 ^(ZMQ publisher^) ...
netsh advfirewall firewall delete rule name="SO101 ZMQ 5555" >nul 2>&1
netsh advfirewall firewall add rule name="SO101 ZMQ 5555" dir=in action=allow protocol=TCP localport=5555 profile=any

echo.
echo Done. On the Mac:
echo   1. bash scripts/21_use_ethernet_link.sh
echo   2. config\camera.env should have ZMQ_HOST=%OMEN_IP%
echo   3. Restart omen\run_publish.bat then Mac dashboard/record
echo.
ipconfig | findstr /i "IPv4 Ethernet"
endlocal
