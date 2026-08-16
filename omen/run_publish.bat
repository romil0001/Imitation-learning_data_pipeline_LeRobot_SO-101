@echo off
cd /d "%~dp0\.."
call conda activate omen-d405
python omen\publish_d405.py %*
