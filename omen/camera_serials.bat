@echo off
REM Local Omen camera IDs — called by run_publish.bat
REM List devices: python omen\publish_d405.py --list-realsense
REM If dashboard wrist tile shows the table cam (or base shows the gripper), swap the two serials.
REM   218622271436 = wrist (gripper ego view)
REM   218622270812 = base (table / scene)
set "WRIST_SERIAL=218622271436"
set "BASE_SERIAL=218622270812"
set "WEBCAM_INDEX=2"
