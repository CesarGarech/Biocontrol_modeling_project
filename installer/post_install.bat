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

:: Target versions definition
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

:: Check in Program Files first (New default)
if exist "%ProgramFiles%\Python310\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python310\python.exe"
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if exist "C:\Python310\python.exe" set "PYTHON_EXE=C:\Python310\python.exe"

if defined PYTHON_EXE (
    echo [SKIP] Python detected at %PYTHON_EXE%
) else (
    echo [INFO] Python 3.10 not detected. Starting download...
    curl -# -L -o python_installer.exe "https://www.python.org/ftp/python/%PYTHON_VER%/python-%PYTHON_VER%-amd64.exe"
    echo [INFO] Installing Python (Showing progress bar)...
    :: /passive shows progress bar. InstallAllUsers=1 forces it to Program Files
    start /wait python_installer.exe /passive InstallAllUsers=1 TargetDir="%ProgramFiles%\Python310" PrependPath=1 Include_pip=1
    set "PYTHON_EXE=%ProgramFiles%\Python310\python.exe"
)

:: ---------------------------------------------------------------------------
:: Step 2 — Evaluate and Install DWSIM
:: ---------------------------------------------------------------------------
echo [INFO] Evaluating DWSIM v%DWSIM_VER%...
set "DWSIM_FOUND=0"
set "DWSIM_PATH=%ProgramFiles%\DWSIM"

:: Check for the DLL in Program Files to confirm it's installed
if exist "%DWSIM_PATH%\DWSIM.Automation.dll" (
    set "DWSIM_FOUND=1"
) else if exist "%LOCALAPPDATA%\DWSIM\DWSIM.Automation.dll" (
    set "DWSIM_FOUND=1"
    set "DWSIM_PATH=%LOCALAPPDATA%\DWSIM"
)

if "%DWSIM_FOUND%"=="0" (
    echo [INFO] DWSIM not detected in Program Files. Downloading installer...
    curl -# -L -o dwsim_installer.exe "https://github.com/DanWBR/dwsim/releases/download/v%DWSIM_VER%/DWSIM_bin_v905_setup_win7_win8_win10_win11_64bit.exe"
    echo [INFO] Installing DWSIM (Showing progress bar)...
    :: /SILENT shows a progress bar without asking questions (unlike /VERYSILENT which hides everything)
    start /wait dwsim_installer.exe /SILENT /SUPPRESSMSGBOXES /NORESTART /DIR="%ProgramFiles%\DWSIM"
    echo [SUCCESS] DWSIM installed successfully.
) else (
    echo [SKIP] DWSIM detected at: %DWSIM_PATH%
)

:: ---------------------------------------------------------------------------
:: Step 3 — Evaluate and Install .NET Runtime (>= Version)
:: ---------------------------------------------------------------------------
echo [INFO] Evaluating .NET Runtime (requires v%DOTNET_VER% or higher)...
set "DOTNET_OK=0"

:: Extract installed versions and verify if any is >= 8
for /f "tokens=2" %%A in ('dotnet --list-runtimes 2^>nul ^| findstr "Microsoft.NETCore.App"') do (
    for /f "tokens=1 delims=." %%V in ("%%A") do (
        if %%V GEQ 8 set "DOTNET_OK=1"
    )
)

if "!DOTNET_OK!"=="0" (
    echo [INFO] Installing .NET %DOTNET_VER% via winget...
    winget install Microsoft.DotNet.Runtime.8 --silent --accept-package-agreements --accept-source-agreements
) else (
    echo [SKIP] .NET Runtime 8.0 or higher is already configured on the system.
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