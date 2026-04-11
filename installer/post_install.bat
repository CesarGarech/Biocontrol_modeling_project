@echo off
:: =============================================================================
:: post_install.bat — Biocontrol Dashboard post-installation script
:: Called automatically by the Inno Setup [Run] section after file copying.
:: Usage: post_install.bat "<install_dir>"
:: =============================================================================
setlocal enabledelayedexpansion

:: Installation directory is passed as the first argument
if "%~1"=="" (
    set "INSTALL_DIR=%~dp0"
) else (
    set "INSTALL_DIR=%~1"
)

echo.
echo ============================================================
echo  Biocontrol Dashboard — Post-Installation Setup
echo ============================================================
echo  Install directory: %INSTALL_DIR%
echo.

:: ---------------------------------------------------------------------------
:: Step 1 — Locate Python 3.10 executable
:: ---------------------------------------------------------------------------
set "PYTHON_EXE="

:: Check common installation paths first
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

:: Fallback: search PATH
for /f "tokens=* usebackq" %%P in (`where python 2^>nul`) do (
    if not defined PYTHON_EXE (
        set "PYTHON_EXE=%%P"
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python executable not found.
    echo         Please ensure Python 3.10.9 was installed correctly.
    exit /b 1
)

:FoundPython
echo [INFO] Python executable: %PYTHON_EXE%

:: ---------------------------------------------------------------------------
:: Step 2 — Verify it is Python 3.10.x
:: ---------------------------------------------------------------------------
for /f "tokens=2 delims= " %%V in ('"%PYTHON_EXE%" --version 2^>^&1') do set "PY_VER=%%V"
for /f "tokens=1,2 delims=." %%A in ("%PY_VER%") do set "PY_MAJOR_MINOR=%%A.%%B"

if not "%PY_MAJOR_MINOR%"=="3.10" (
    echo [ERROR] Found Python %PY_VER% but Python 3.10.x is required.
    exit /b 1
)

echo [INFO] Python version verified: %PY_VER%

:: ---------------------------------------------------------------------------
:: Step 3 — Create virtual environment (if it does not already exist)
:: ---------------------------------------------------------------------------
set "VENV_DIR=%INSTALL_DIR%\.venv"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment at %VENV_DIR% ...
    "%PYTHON_EXE%" -m venv "%VENV_DIR%"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo [INFO] Virtual environment already exists — skipping creation.
)

:: ---------------------------------------------------------------------------
:: Step 4 — Activate the virtual environment
:: ---------------------------------------------------------------------------
call "%VENV_DIR%\Scripts\activate.bat"
if !ERRORLEVEL! NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)
echo [INFO] Virtual environment activated.

:: ---------------------------------------------------------------------------
:: Step 5 — Upgrade pip
:: ---------------------------------------------------------------------------
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet
if !ERRORLEVEL! NEQ 0 (
    echo [WARNING] pip upgrade failed — continuing anyway.
)

:: ---------------------------------------------------------------------------
:: Step 6 — Install dependencies from requirements.txt
:: ---------------------------------------------------------------------------
set "REQ_FILE=%INSTALL_DIR%\requirements.txt"

if not exist "%REQ_FILE%" (
    echo [ERROR] requirements.txt not found at %REQ_FILE%
    exit /b 1
)

echo [INFO] Installing Python dependencies (this may take several minutes)...
pip install -r "%REQ_FILE%" --quiet
if !ERRORLEVEL! NEQ 0 (
    echo [WARNING] Quiet install failed — retrying with full output...
    pip install -r "%REQ_FILE%"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Dependency installation failed.
        exit /b 1
    )
)
echo [INFO] Python dependencies installed successfully.

:: ---------------------------------------------------------------------------
:: Step 7 — Verify DWSIM installation
:: ---------------------------------------------------------------------------
echo [INFO] Checking DWSIM installation...
if exist "%LOCALAPPDATA%\DWSIM\DWSIM.Automation.dll" (
    echo [INFO] DWSIM found at %LOCALAPPDATA%\DWSIM
) else (
    echo [WARNING] DWSIM not found at %LOCALAPPDATA%\DWSIM
    echo           The Digital Twin module will use synthetic data.
    echo           Install DWSIM 8.x manually from https://dwsim.org if needed.
)

:: ---------------------------------------------------------------------------
:: Step 8 — Quick verification of key libraries
:: ---------------------------------------------------------------------------
echo [INFO] Verifying key libraries...
set "VERIFY_FAILED=0"

python -c "import streamlit; print('[OK] streamlit', streamlit.__version__)" 2>nul || (
    echo [WARNING] streamlit not importable.
    set "VERIFY_FAILED=1"
)

python -c "import casadi; print('[OK] casadi', casadi.__version__)" 2>nul || (
    echo [WARNING] casadi not importable.
    set "VERIFY_FAILED=1"
)

python -c "import tensorflow; print('[OK] tensorflow', tensorflow.__version__)" 2>nul || (
    echo [WARNING] tensorflow not importable (may be expected on some systems).
)

python -c "import scipy; print('[OK] scipy', scipy.__version__)" 2>nul || (
    echo [WARNING] scipy not importable.
    set "VERIFY_FAILED=1"
)

if %VERIFY_FAILED%==1 (
    echo [WARNING] One or more core libraries could not be verified.
    echo           The application may not work correctly.
    exit /b 2
)

echo.
echo ============================================================
echo  Post-installation completed successfully!
echo  You can now launch the dashboard from the Start Menu.
echo ============================================================
echo.
exit /b 0
