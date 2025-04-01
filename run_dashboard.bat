@echo off
title Dashboard de Bioprocesos
cd /d "%~dp0"

:: Ruta a conda.bat (ajústala si es necesario)
CALL "%USERPROFILE%\anaconda3\Scripts\activate.bat"

:: Nombre del entorno
set ENV_NAME=cabbio_env

:: Verificar si el entorno existe
CALL conda info --envs | findstr /C:"%ENV_NAME%" >nul
IF ERRORLEVEL 1 (
    echo Creando entorno Conda "%ENV_NAME%"...
    call conda create -y -n %ENV_NAME% python=3.10
)

:: Activar el entorno
CALL conda activate %ENV_NAME%

:: Instalar dependencias si es necesario
IF EXIST "requirements.txt" (
    pip install -r requirements.txt
) ELSE (
    echo Instalando Streamlit...
    pip install streamlit
)

:: Ejecutar la aplicación
streamlit run St_CABBIO03.py

pause
