@echo off
setlocal

set PYTHON_VERSION=3.11.9
set PYTHON_INSTALLER=python-3.11.9-amd64.exe
set REPO_URL=https://github.com/Beanthief/PurePlay-Anti-Cheat

echo Checking if Python %PYTHON_VERSION% is installed...
python --version 2>nul | findstr /r /c:"Python %PYTHON_VERSION%" >nul
if errorlevel 1 (
    echo Python %PYTHON_VERSION% is not installed.
    echo Downloading Python %PYTHON_VERSION%...
    curl -LO https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_INSTALLER%

    echo Installing Python...
    start /wait %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
) else (
    echo Python %PYTHON_VERSION% is already installed.
)

echo Creating virtual environment...
python -m venv venv

call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete!
endlocal
pause