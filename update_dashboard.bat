@echo off
title Biocontrol Dashboard - Updater
cd /d "%~dp0"

echo ============================================================
echo   Biocontrol Dashboard - Updater
echo ============================================================
echo.

REM ---- Step 1: Try Git first ----
echo [1/3] Checking for Git...
where git >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :zip_download

echo       Git found. Pulling latest code from GitHub...
git pull
if %ERRORLEVEL% EQU 0 (
    echo       Code updated successfully via Git.
    goto :update_libs
)

echo       Git pull failed. Falling back to ZIP download...

REM ---- Step 2: ZIP download fallback ----
:zip_download
echo [1/3] Downloading latest code from GitHub (ZIP)...
set "ZIP_URL=https://github.com/CesarGarech/Biocontrol_modeling_project/archive/refs/heads/main.zip"
set "TEMP_ZIP=%TEMP%\biocontrol_update.zip"
set "TEMP_DIR=%TEMP%\biocontrol_update"

powershell -NoProfile -Command "Invoke-WebRequest -Uri '%ZIP_URL%' -OutFile '%TEMP_ZIP%'" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not download the update. Check your internet connection.
    pause
    exit /b 1
)

echo       Extracting files...
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
powershell -NoProfile -Command "Expand-Archive -Path '%TEMP_ZIP%' -DestinationPath '%TEMP_DIR%' -Force" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not extract the downloaded ZIP.
    pause
    exit /b 1
)

echo       Copying updated files (preserving .venv, dependencies, installer)...
robocopy "%TEMP_DIR%\Biocontrol_modeling_project-main" "." /E /XD .venv dependencies installer .git __pycache__ /XF "*.pyc" /NFL /NDL /NJH /NJS >nul 2>&1
REM robocopy exit codes 0-7 are success (files copied/skipped); 8+ are errors
if %ERRORLEVEL% GEQ 8 (
    echo ERROR: File copy failed (robocopy exit code %ERRORLEVEL%).
    pause
    exit /b 1
)

echo       Cleaning up temporary files...
del /f /q "%TEMP_ZIP%" >nul 2>&1
rmdir /s /q "%TEMP_DIR%" >nul 2>&1

echo       Code updated successfully via ZIP download.

REM ---- Step 3: Update Python libraries ----
:update_libs
echo [2/3] Activating virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo WARNING: Virtual environment not found at .venv\Scripts\activate.bat
    echo          Skipping library update.
    goto :done
)
call ".venv\Scripts\activate.bat"

echo [3/3] Updating Python libraries...
pip install -r requirements.txt --quiet --upgrade
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some libraries could not be updated. Check your internet connection.
) else (
    echo       Libraries updated successfully.
)

:done
echo.
echo ============================================================
echo   Update complete! Run run_dashboard.bat to start the app.
echo ============================================================
echo.
pause
