@echo off
TITLE Installing SMA2150 Shortcut...
ECHO 🚀 Setting up your Desktop Shortcut...

:: Change directory to script location
CD /D "%~dp0"

:: Check for Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO ❌ Python is not installed.
    ECHO Please install Python from python.org and try again.
    PAUSE
    EXIT /B
)

:: Run the Python Installer
python install_shortcuts.py

ECHO.
ECHO ✅ Done! Check your Desktop.
PAUSE
