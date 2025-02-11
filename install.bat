@echo off
setlocal

set "installer_url=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
set "installer_path=%USERPROFILE%\Downloads\Miniforge3-latest-Windows-x86_64.exe"

where conda >nul 2>&1
if errorlevel 1 (
    echo Downloading the latest Miniforge installer...
    echo Installer URL: %installer_url%
    echo Installer Path: %installer_path%
    powershell -Command "Invoke-WebRequest -Uri '%installer_url%' -OutFile '%installer_path%'"
    if errorlevel 1 (
        echo Failed to download the installer.
        pause
        exit /b
    )
    echo Running the installer. Please follow the prompts to install Miniforge.
    echo After installation, please restart this script.
    start "" "%installer_path%"
    pause
    exit /b
)

echo Creating conda environment with Python...
call conda create -y -q -n PurePlay-Anti-Cheat python
if errorlevel 1 (
    echo Failed to create the environment with Python.
    pause
    exit /b
)

echo Activating environment...
call conda activate PurePlay-Anti-Cheat

echo Installing PyTorch with Nvidia binaries...
call conda install -y -q -c pytorch pytorch
if errorlevel 1 (
    echo Failed to install Nvidia binaries.
    pause
    exit /b
)

echo Installing pip packages...
REM pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128
call pip install XInput-Python mouse keyboard scikit-learn pandas matplotlib pyautogui optuna 
if errorlevel 1 (
    echo Failed to install pip packages.
    pause
    exit /b
)

echo Run start.bat to begin!
pause
endlocal