@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "SRC=%~dp0calibration"
set "DST=%USERPROFILE%\.cache\huggingface\lerobot\calibration"

if not exist "%SRC%\robots\so_follower\my_follower_arm.json" (
  echo Missing %SRC%\robots\so_follower\my_follower_arm.json
  echo Run this from the repo copy that has omen\calibration\, or use fetch_calibration_from_mac.bat
  exit /b 1
)
if not exist "%SRC%\teleoperators\so_leader\my_leader_arm.json" (
  echo Missing %SRC%\teleoperators\so_leader\my_leader_arm.json
  exit /b 1
)

mkdir "%DST%\robots\so_follower" 2>nul
mkdir "%DST%\teleoperators\so_leader" 2>nul

copy /Y "%SRC%\robots\so_follower\my_follower_arm.json" "%DST%\robots\so_follower\my_follower_arm.json" >nul
copy /Y "%SRC%\teleoperators\so_leader\my_leader_arm.json" "%DST%\teleoperators\so_leader\my_leader_arm.json" >nul

echo Installed follower -^> %DST%\robots\so_follower\my_follower_arm.json
echo Installed leader   -^> %DST%\teleoperators\so_leader\my_leader_arm.json
echo Restart robot_bridge after this if it is already running.
exit /b 0
