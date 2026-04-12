@echo off
:: =============================================================================
:: run_dashboard.bat — Biocontrol Dashboard launcher
:: =============================================================================
:: Strategy:
::   1. If the .venv already exists, activate it and launch directly.
::      (The venv was created by the installer with Python 3.10, so no need
::       to verify the system Python version at all.)
::   2. If the .venv does not exist yet, locate Python 3.10 via:
::         a) Well-known hard-coded paths  (reliable even when PATH is stale)
::         b) Windows registry             (handles both per-user and system installs)
::         c) PATH fallback                (last resort)
::      Then create the venv, install requirements, and launch.
:: =============================================================================
setlocal enabledelayedexpansion
title Bioprocess Dashboard
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"

:: ---------------------------------------------------------------------------
:: Fast path — venv already exists (normal case after installation)
:: ---------------------------------------------------------------------------
if exist "%VENV_ACTIVATE%" (
    echo [INFO] Activating existing virtual environment...
    call "%VENV_ACTIVATE%"
    goto :Launch
)

:: ---------------------------------------------------------------------------
:: Slow path — first run or venv was deleted; locate Python 3.10
:: ---------------------------------------------------------------------------
echo [INFO] Virtual environment not found. Locating Python 3.10...
set "PYTHON_EXE="

:: a) Hard-coded well-known paths
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :FoundPython
)
if exist "%ProgramFiles%\Python310\python.exe" (
    set "PYTHON_EXE=%ProgramFiles%\Python310\python.exe"
    goto :FoundPython
)
if exist "C:\Python310\python.exe" (
    set "PYTHON_EXE=C:\Python310\python.exe"
    goto :FoundPython
)

:: b) Registry — HKCU (per-user install)
for /f "tokens=2*" %%A in ('reg query "HKCU\SOFTWARE\Python\PythonCore\3.10\InstallPath" /ve 2^>nul') do (
    if not defined PYTHON_EXE (
        set "_P=%%B"
        if "!_P:~-1!"=="\" (set "_P=!_P:~0,-1!")
        if exist "!_P!\python.exe" set "PYTHON_EXE=!_P!\python.exe"
    )
)
if defined PYTHON_EXE goto :FoundPython

:: b) Registry — HKLM (system-wide install)
for /f "tokens=2*" %%A in ('reg query "HKLM\SOFTWARE\Python\PythonCore\3.10\InstallPath" /ve 2^>nul') do (
    if not defined PYTHON_EXE (
        set "_P=%%B"
        if "!_P:~-1!"=="\" (set "_P=!_P:~0,-1!")
        if exist "!_P!\python.exe" set "PYTHON_EXE=!_P!\python.exe"
    )
)
if defined PYTHON_EXE goto :FoundPython

for /f "tokens=2*" %%A in ('reg query "HKLM\SOFTWARE\WOW6432Node\Python\PythonCore\3.10\InstallPath" /ve 2^>nul') do (
    if not defined PYTHON_EXE (
        set "_P=%%B"
        if "!_P:~-1!"=="\" (set "_P=!_P:~0,-1!")
        if exist "!_P!\python.exe" set "PYTHON_EXE=!_P!\python.exe"
    )
)
if defined PYTHON_EXE goto :FoundPython

:: c) PATH fallback (last resort — may return a non-3.10 python)
for /f "tokens=* usebackq" %%P in (`where python 2^>nul`) do (
    if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
)

if not defined PYTHON_EXE (
    echo.
    echo [ERROR] Python 3.10 was not found on this computer.
    echo         Please install Python 3.10.9 from:
    echo           https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe
    echo         Then re-run this launcher.
    echo.
    pause
    exit /b 1
)

:FoundPython
:: Confirm it really is 3.10.x before creating the venv
for /f "tokens=2 delims= " %%V in ('"%PYTHON_EXE%" --version 2^>^&1') do set "PY_VER=%%V"
for /f "tokens=1,2 delims=." %%A in ("%PY_VER%") do set "PY_MM=%%A.%%B"
if not "!PY_MM!"=="3.10" (
    echo.
    echo [ERROR] Found Python !PY_VER! at %PYTHON_EXE%, but 3.10.x is required.
    echo         Please install Python 3.10.9 from:
    echo           https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe
    echo.
    pause
    exit /b 1
)

echo [INFO] Using Python !PY_VER! at %PYTHON_EXE%
echo [INFO] Creating virtual environment in %VENV_DIR% ...
"%PYTHON_EXE%" -m venv "%VENV_DIR%"
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

call "%VENV_ACTIVATE%"

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet

if exist "requirements.txt" (
    echo [INFO] Installing dependencies from requirements.txt ...
    pip install -r requirements.txt --quiet
    if !ERRORLEVEL! NEQ 0 (
        echo [WARNING] Some dependencies may not have installed correctly.
    )
)

:: ---------------------------------------------------------------------------
:: Launch the dashboard
:: ---------------------------------------------------------------------------
:Launch
echo.
echo ============================================================
echo  Starting Biocontrol Dashboard ...
echo ============================================================
echo.
streamlit run main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The dashboard exited with an error.
    echo         If the error mentions DWSIM or a missing DLL, DWSIM 8.x may not
    echo         be installed. Download it from https://dwsim.org/index.php/download/
    echo         and install it, then restart the dashboard.
    echo.
    pause
)
