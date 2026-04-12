@echo off
:: =============================================================================
:: post_install.bat — Biocontrol Dashboard post-installation script
:: =============================================================================
setlocal enabledelayedexpansion

if "%~1"=="" (
    set "INSTALL_DIR=%~dp0"
) else (
    set "INSTALL_DIR=%~1"
)

:: Definición de versiones objetivo
set "PYTHON_VER=3.10.14"
set "DWSIM_VER=9.0.5"
set "DOTNET_VER=8.0"

echo.
echo ============================================================
echo  Biocontrol Dashboard — Installation and Verification
echo ============================================================
echo  Installation Directory: %INSTALL_DIR%
echo.

:: ---------------------------------------------------------------------------
:: Step 1 — Evaluate and Install Python 3.10.x
:: ---------------------------------------------------------------------------
echo [INFO] Evaluating Python 3.10.x...
set "PYTHON_EXE="

:: Search in known paths
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if exist "%ProgramFiles%\Python310\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python310\python.exe"
if exist "C:\Python310\python.exe" set "PYTHON_EXE=C:\Python310\python.exe"

if defined PYTHON_EXE (
    echo [SKIP] Python detected at %PYTHON_EXE%
) else (
    echo [INFO] Python 3.10 not detected. Starting silent download...
    curl -# -L -o python_installer.exe "https://www.python.org/ftp/python/%PYTHON_VER%/python-%PYTHON_VER%-amd64.exe"
    echo [INFO] Installing Python (this process may take a few minutes)...
    start /wait python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
)

:: ---------------------------------------------------------------------------
:: Step 2 — Evaluate and Install DWSIM
:: ---------------------------------------------------------------------------
echo [INFO] Evaluating DWSIM v%DWSIM_VER%...
set "DWSIM_FOUND=0"
set "DWSIM_PATH=%LOCALAPPDATA%\DWSIM"

if exist "%LOCALAPPDATA%\DWSIM\DWSIM.Automation.dll" set "DWSIM_FOUND=1"
if exist "%ProgramFiles%\DWSIM\DWSIM.Automation.dll" (
    set "DWSIM_FOUND=1"
    set "DWSIM_PATH=%ProgramFiles%\DWSIM"
)

if "%DWSIM_FOUND%"=="0" (
    echo [INFO] DWSIM not detected. Downloading silent installer...
    curl -# -L -o dwsim_installer.exe "https://github.com/DanWBR/dwsim/releases/download/v%DWSIM_VER%/DWSIM_bin_v905_setup_win7_win8_win10_win11_64bit.exe"
    echo [INFO] Installing DWSIM silently...
    start /wait dwsim_installer.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
    echo [SUCCESS] DWSIM installed successfully.
) else (
    echo [SKIP] DWSIM detected at: %DWSIM_PATH%
)

:: ---------------------------------------------------------------------------
:: Step 3 — Evaluate and Install .NET Runtime
:: ---------------------------------------------------------------------------
echo [INFO] Evaluating .NET Runtime %DOTNET_VER%...
dotnet --list-runtimes 2>nul | findstr /c:"Microsoft.NETCore.App %DOTNET_VER%" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Installing .NET %DOTNET_VER% via winget...
    winget install Microsoft.DotNet.Runtime.8 --silent --accept-package-agreements --accept-source-agreements
) else (
    echo [SKIP] .NET Runtime already configured on the system.
)

:: ---------------------------------------------------------------------------
:: Step 4 — Create Virtual Environment
:: ---------------------------------------------------------------------------
set "VENV_DIR=%INSTALL_DIR%\.venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment in %VENV_DIR% ...
    "%PYTHON_EXE%" -m venv "%VENV_DIR%"
) else (
    echo [INFO] Existing virtual environment detected. Skipping creation.
)

:: ---------------------------------------------------------------------------
:: Step 5 — Activate and Install Dependencies
:: ---------------------------------------------------------------------------
call "%VENV_DIR%\Scripts\activate.bat"
echo [INFO] Updating package manager pip...
python -m pip install --upgrade pip --quiet

set "REQ_FILE=%INSTALL_DIR%\requirements.txt"
if exist "%REQ_FILE%" (
    echo [INFO] Installing required dependencies (this may take several minutes)...
    pip install -r "%REQ_FILE%" --quiet
) else (
    echo [ERROR] requirements.txt file not found.
)

echo.
echo ============================================================
echo  Post-installation process successfully complete.
echo ============================================================
exit /b 0