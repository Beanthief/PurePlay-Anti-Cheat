@echo off

IF NOT EXIST "venv" (
    echo Error: Virtual environment not found. Please run deploy.bat to install required packages.
    pause
    exit /b
)

call venv\Scripts\activate
python source\main.py
pause