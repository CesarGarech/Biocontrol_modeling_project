@echo off
title Dashboard de Bioprocesos
cd /d "%~dp0"

:: Verificar si existe el entorno virtual
if not exist ".venv\" (
    echo Creando entorno virtual...
    python -m venv .venv
)

:: Activar entorno virtual y actualizar pip
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip

:: Instalar dependencias
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo Instalando Streamlit...
    pip install streamlit
)

:: Ejecutar aplicación
streamlit run St_CABBIO03.py

pause