@echo off
REM Verification script for Python 3.10.x, DWSIM, and .NET Runtime 8.0
setlocal enabledelayedexpansion

set "PYTHON_VER=3.10.14"
set "DWSIM_VER=9.0.5"
set "DOTNET_VER=8.0"

echo ============================================================
echo Environment Evaluation for Biocontrol Modeling Project
echo ============================================================
echo.

:: Evaluation of Python
set "PY_INSTALLED=0"
for /f "tokens=2 delims= " %%I in ('python --version 2^>^&1') do (
    echo %%I | findstr /b "3.10" >nul
    if !ERRORLEVEL! EQU 0 set "PY_INSTALLED=1"
)

if "!PY_INSTALLED!"=="0" (
    echo [INFO] Python 3.10 not detected. Starting silent installation...
    curl -# -L -o python_installer.exe "https://www.python.org/ftp/python/%PYTHON_VER%/python-%PYTHON_VER%-amd64.exe"
    start /wait python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1
) else (
    echo [SKIP] Python 3.10 verified correctly.
)

:: Evaluation of DWSIM
if not exist "%LOCALAPPDATA%\DWSIM\DWSIM.Automation.dll" (
    if not exist "%ProgramFiles%\DWSIM\DWSIM.Automation.dll" (
        echo [INFO] DWSIM not detected. Downloading installer...
        curl -# -L -o dwsim_installer.exe "https://github.com/DanWBR/dwsim/releases/download/v%DWSIM_VER%/DWSIM_bin_v905_setup_win7_win8_win10_win11_64bit.exe"
        start /wait dwsim_installer.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
    ) else (
        echo [SKIP] DWSIM verified in Program Files.
    )
) else (
    echo [SKIP] DWSIM verified in LocalAppData.
)

:: Evaluation of .NET
dotnet --list-runtimes 2>nul | findstr /c:"Microsoft.NETCore.App %DOTNET_VER%" >nul
if !ERRORLEVEL! NEQ 0 (
    echo [INFO] Installing .NET %DOTNET_VER% via winget...
    winget install Microsoft.DotNet.Runtime.8 --silent --accept-package-agreements --accept-source-agreements
) else (
    echo [SKIP] .NET Runtime 8.0 verified correctly.
)

exit /b 0