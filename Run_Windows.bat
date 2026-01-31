@echo off
TITLE SMA2150 Command Center
ECHO 🚀 Starting SMA2150 Command Center...
ECHO.
ECHO Please wait while the application loads...

:: Change directory to script location
CD /D "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO ❌ Python is not installed or not in PATH.
    ECHO Please install Python 3.10+ from python.org
    PAUSE
    EXIT /B
)

:: Run Streamlit App (Mobile Access Enabled)
streamlit run app.py --server.address 0.0.0.0

PAUSE
