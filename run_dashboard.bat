@echo off
title Bioprocess Dashboard
cd /d "%~dp0"

:: Verify Python version first
echo Verifying Python version...
call verify_python.bat
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Cannot continue without Python 3.10.9 installed.
    echo Please install Python 3.10.9 following the previous instructions.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Python 3.10.x verified successfully. Continuing...
echo ============================================================
echo.

:: Environment name
set ENV_DIR=.venv

:: Check if environment already exists
IF NOT EXIST "%ENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment in "%ENV_DIR%"...
    python -m venv %ENV_DIR%
)

:: Activate the environment
CALL "%ENV_DIR%\Scripts\activate.bat"

:: Install dependencies if necessary
IF EXIST "requirements.txt" (
    pip install --upgrade pip
    pip install -r requirements.txt
) ELSE (
    echo Installing Streamlit...
    pip install --upgrade pip
    pip install streamlit
)

:: Run the application
@REM streamlit run St_CABBIO03.py
streamlit run main.py

pause