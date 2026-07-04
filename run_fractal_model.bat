@echo off
rem Fractal Model launcher — starts the local Dash server and opens the browser.
rem Double-click this file, or use scripts\create_desktop_shortcut.ps1 to put
rem a "Fractal Model" shortcut on your Desktop.
title Fractal Model
cd /d "%~dp0"

set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY where py >nul 2>nul && set "PY=py"
if not defined PY (
    echo Could not find Python. Install it, or create a .venv in this folder.
    pause
    exit /b 1
)

echo Starting Fractal Model at http://127.0.0.1:8050 ...
echo (close this window to stop the server)
"%PY%" -m fractal_model.app
if errorlevel 1 pause
