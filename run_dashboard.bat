@echo off
title Bioprocess Dashboard (Dash)
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
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) ELSE (
    echo Installing Dash...
    python -m pip install --upgrade pip
    python -m pip install dash dash-bootstrap-components plotly
)

:: Run the Dash application
echo.
echo ============================================================
echo Starting Bioprocess Modeling Dashboard (Dash)...
echo Open your browser at: http://localhost:8050
echo ============================================================
echo.
python main.py

pause
