@echo off
title Dashboard de Bioprocesos
cd /d "%~dp0"

:: Verify Python version first
echo Verificando version de Python...
call verify_python.bat
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo No se puede continuar sin Python 3.10.14 instalado.
    echo Por favor, instale Python 3.10.14 siguiendo las instrucciones anteriores.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Python 3.10.14 verificado correctamente. Continuando...
echo ============================================================
echo.

:: Nombre del entorno
set ENV_DIR=.venv

:: Verificar si el entorno ya existe
IF NOT EXIST "%ENV_DIR%\Scripts\activate.bat" (
    echo Creando entorno virtual en "%ENV_DIR%"...
    python -m venv %ENV_DIR%
)

:: Activar el entorno
CALL "%ENV_DIR%\Scripts\activate.bat"

:: Instalar dependencias si es necesario
IF EXIST "requirements.txt" (
    pip install --upgrade pip
    pip install -r requirements.txt
) ELSE (
    echo Instalando Streamlit...
    pip install --upgrade pip
    pip install streamlit
)

:: Ejecutar la aplicación
@REM streamlit run St_CABBIO03.py
streamlit run main.py

pause
