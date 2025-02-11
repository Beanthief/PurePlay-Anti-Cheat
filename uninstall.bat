@echo off
echo Removing the PurePlay-Anti-Cheat conda environment...
call conda env remove -n PurePlay-Anti-Cheat -y
if errorlevel 1 (
    echo Failed to remove the PurePlay-Anti-Cheat environment.
    pause
    exit /b
)
echo PurePlay-Anti-Cheat environment removed successfully.
pause