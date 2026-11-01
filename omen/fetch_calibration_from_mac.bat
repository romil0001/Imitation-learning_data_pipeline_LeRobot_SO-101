@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Mac LAN IP serving omen/calibration over HTTP (started from the Mac agent).
if not defined MAC_CAL_URL set "MAC_CAL_URL=http://192.168.0.120:8765"

set "TMP=%TEMP%\so101_calibration"
mkdir "%TMP%\robots\so_follower" 2>nul
mkdir "%TMP%\teleoperators\so_leader" 2>nul

echo Downloading calibrations from %MAC_CAL_URL% ...
curl -fsSL "%MAC_CAL_URL%/robots/so_follower/my_follower_arm.json" -o "%TMP%\robots\so_follower\my_follower_arm.json"
if errorlevel 1 (
  echo Download failed. Is the Mac serving on 192.168.0.120:8765 ?
  exit /b 1
)
curl -fsSL "%MAC_CAL_URL%/teleoperators/so_leader/my_leader_arm.json" -o "%TMP%\teleoperators\so_leader\my_leader_arm.json"
if errorlevel 1 (
  echo Download failed for leader calibration.
  exit /b 1
)

set "DST=%USERPROFILE%\.cache\huggingface\lerobot\calibration"
mkdir "%DST%\robots\so_follower" 2>nul
mkdir "%DST%\teleoperators\so_leader" 2>nul
copy /Y "%TMP%\robots\so_follower\my_follower_arm.json" "%DST%\robots\so_follower\my_follower_arm.json" >nul
copy /Y "%TMP%\teleoperators\so_leader\my_leader_arm.json" "%DST%\teleoperators\so_leader\my_leader_arm.json" >nul

echo Installed follower -^> %DST%\robots\so_follower\my_follower_arm.json
echo Installed leader   -^> %DST%\teleoperators\so_leader\my_leader_arm.json
echo Restart robot_bridge after this if it is already running.
exit /b 0
