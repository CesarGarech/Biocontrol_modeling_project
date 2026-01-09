@echo off
REM Python Version Verification Script for Biocontrol Modeling Project
REM This script verifies Python 3.10.14 is installed and provides installation guidance

setlocal enabledelayedexpansion

set "REQUIRED_VERSION=3.10.14"
set "REQUIRED_MAJOR_MINOR=3.10"
set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.10.14/python-3.10.14-amd64.exe"
set "PYTHON_DOWNLOAD_PAGE=https://www.python.org/downloads/release/python-31014/"

echo ============================================================
echo Python Version Verification for Biocontrol Modeling Project
echo ============================================================
echo.
echo Required Python Version: %REQUIRED_VERSION% (or compatible 3.10.x)
echo.

REM Check if python is available in PATH
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH
    goto :InstallInstructions
)

REM Get Python version
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set CURRENT_VERSION=%%i

echo Current Python Version: %CURRENT_VERSION%
echo.

REM Extract major.minor version (e.g., 3.10 from 3.10.14)
for /f "tokens=1,2 delims=." %%a in ("%CURRENT_VERSION%") do set CURRENT_MAJOR_MINOR=%%a.%%b

REM Check if version is compatible (3.10.x)
if "%CURRENT_MAJOR_MINOR%"=="%REQUIRED_MAJOR_MINOR%" (
    echo [SUCCESS] Python 3.10.x is correctly installed!
    echo.
    exit /b 0
)

REM Version doesn't match - show warning
echo [WARNING] Python version mismatch detected!
echo.
echo The current Python version (%CURRENT_VERSION%) may cause compatibility issues.
echo This project requires Python 3.10.x specifically due to:
echo   - TensorFlow compatibility requirements
echo   - CasADi C++ binding compatibility
echo   - NumPy/SciPy version stability
echo.

:InstallInstructions
echo ============================================================
echo Python %REQUIRED_VERSION% Installation Instructions
echo ============================================================
echo.
echo Option 1: Download and Install Manually
echo -----------------------------------------
echo 1. Open your web browser and visit:
echo    %PYTHON_DOWNLOAD_PAGE%
echo.
echo 2. Download "Windows installer (64-bit)" for Python %REQUIRED_VERSION%
echo    Direct link: %PYTHON_INSTALLER_URL%
echo.
echo 3. Run the installer and IMPORTANT:
echo    - Check "Add Python 3.10 to PATH"
echo    - Choose "Install Now" or "Customize installation"
echo    - If customizing, ensure pip is included
echo.
echo 4. After installation, close this window and run run_dashboard.bat again
echo.
echo.
echo Option 2: Install via winget (Windows Package Manager)
echo -------------------------------------------------------
echo If you have winget installed, run this command in a NEW command prompt:
echo    winget install Python.Python.3.10 --version 3.10.14
echo.
echo.
echo Option 3: Use pyenv-win (Python Version Manager)
echo ------------------------------------------------
echo 1. Install pyenv-win if not already installed
echo 2. Run these commands in a NEW command prompt:
echo    pyenv install 3.10.14
echo    pyenv global 3.10.14
echo.
echo.
echo Note: After installing Python %REQUIRED_VERSION%, please:
echo   1. Close this command prompt window
echo   2. Open a NEW command prompt
echo   3. Run run_dashboard.bat again
echo.

pause
exit /b 1
