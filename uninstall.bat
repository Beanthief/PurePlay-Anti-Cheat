@echo off
echo Removing the PurePlay-Anti-Cheat conda environment...
call conda env remove -y -q -n PurePlay-Anti-Cheat
if errorlevel 1 (
    echo Failed to remove the PurePlay-Anti-Cheat environment.
    pause
    exit /b
)
echo PurePlay-Anti-Cheat environment removed successfully.
pause