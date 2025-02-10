@echo off
call conda activate PurePlay-Anti-Cheat
if errorlevel 1 (
    echo No environment found. Please run install.bat first.
    pause
    exit /b
)
python source\main.py
pause