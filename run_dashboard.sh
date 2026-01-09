#!/bin/bash
# Dashboard de Bioprocesos - Linux/Mac Startup Script
cd "$(dirname "$0")"

# Verify Python version first
echo "Verificando version de Python..."
./verify_python.sh
if [ $? -ne 0 ]; then
    echo ""
    echo "No se puede continuar sin Python 3.10.14 instalado."
    echo "Por favor, instale Python 3.10.14 siguiendo las instrucciones anteriores."
    exit 1
fi

echo ""
echo "============================================================"
echo "Python 3.10.14 verificado correctamente. Continuando..."
echo "============================================================"
echo ""

# Virtual environment directory
ENV_DIR=".venv"

# Check if virtual environment exists
if [ ! -d "$ENV_DIR" ]; then
    echo "Creando entorno virtual en '$ENV_DIR'..."
    python3 -m venv "$ENV_DIR"
fi

# Activate the virtual environment
source "$ENV_DIR/bin/activate"

# Install dependencies if needed
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Instalando Streamlit..."
    pip install --upgrade pip
    pip install streamlit
fi

# Run the application
streamlit run main.py
