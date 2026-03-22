#!/bin/bash
# Bioprocess Dashboard (Dash) - Linux/Mac Startup Script
cd "$(dirname "$0")"

# Verify Python version first
echo "Verifying Python version..."
./verify_python.sh
if [ $? -ne 0 ]; then
    echo ""
    echo "Cannot continue without Python 3.10.x installed."
    echo "Please install Python 3.10.x following the previous instructions."
    exit 1
fi

echo ""
echo "============================================================"
echo "Python 3.10.x verified successfully. Continuing..."
echo "============================================================"
echo ""

# Virtual environment directory
ENV_DIR=".venv"

# Check if virtual environment exists
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment in '$ENV_DIR'..."
    python3 -m venv "$ENV_DIR"
fi

# Activate the virtual environment
source "$ENV_DIR/bin/activate"

# Install dependencies if needed
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Installing Dash..."
    pip install --upgrade pip
    pip install dash dash-bootstrap-components plotly
fi

# Run the Dash application
echo ""
echo "============================================================"
echo "Starting Bioprocess Modeling Dashboard (Dash)..."
echo "Open your browser at: http://localhost:8050"
echo "============================================================"
echo ""
python3 main.py
