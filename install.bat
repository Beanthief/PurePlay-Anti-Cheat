@echo off
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

echo Creating conda environment...
call conda create -n PurePlay-Anti-Cheat -y
if errorlevel 1 (
    echo Failed to create the environment.
    pause
    exit /b
)

echo Activating environment...
call conda activate PurePlay-Anti-Cheat

echo Installing python...
conda install python

echo Installing nvidia binaries...
conda install -c nvidia cudatoolkit=12.6 cudnn

echo Installing pip packages...
pip install keras-tuner XInput-Python mouse keyboard scikit-learn pandas matplotlib pyautogui optuna
pip install --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo Failed to install pip packages.
    pause
    exit /b
)

echo Run start.bat to begin!
pause