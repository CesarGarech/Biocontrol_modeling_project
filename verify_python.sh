#!/bin/bash
# Python Version Verification Script for Biocontrol Modeling Project
# This script verifies Python 3.10.14 is installed and provides installation guidance

REQUIRED_VERSION="3.10.14"
PYTHON_DOWNLOAD_PAGE="https://www.python.org/downloads/release/python-31014/"

echo "============================================================"
echo "Python Version Verification for Biocontrol Modeling Project"
echo "============================================================"
echo ""
echo "Required Python Version: $REQUIRED_VERSION"
echo ""

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH"
    INSTALL_NEEDED=1
else
    # Get Python version
    CURRENT_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    
    echo "Current Python Version: $CURRENT_VERSION"
    echo ""
    
    # Check if version matches exactly
    if [ "$CURRENT_VERSION" = "$REQUIRED_VERSION" ]; then
        echo "[SUCCESS] Python $REQUIRED_VERSION is correctly installed!"
        echo ""
        exit 0
    fi
    
    # Version doesn't match - show warning
    echo "[WARNING] Python version mismatch detected!"
    echo ""
    echo "The current Python version ($CURRENT_VERSION) may cause compatibility issues."
    echo "This project requires Python $REQUIRED_VERSION specifically due to:"
    echo "  - TensorFlow compatibility requirements"
    echo "  - CasADi C++ binding compatibility"
    echo "  - NumPy/SciPy version stability"
    echo ""
    INSTALL_NEEDED=1
fi

if [ "$INSTALL_NEEDED" = "1" ]; then
    echo "============================================================"
    echo "Python $REQUIRED_VERSION Installation Instructions"
    echo "============================================================"
    echo ""
    echo "Option 1: Download and Install from Python.org"
    echo "-----------------------------------------------"
    echo "Visit: $PYTHON_DOWNLOAD_PAGE"
    echo "Download and install Python $REQUIRED_VERSION for your operating system"
    echo ""
    echo ""
    echo "Option 2: Use pyenv (Python Version Manager)"
    echo "--------------------------------------------"
    echo "1. Install pyenv if not already installed:"
    echo "   curl https://pyenv.run | bash"
    echo ""
    echo "2. Add pyenv to your shell (if not already done):"
    echo "   echo 'export PYENV_ROOT=\"\$HOME/.pyenv\"' >> ~/.bashrc"
    echo "   echo 'export PATH=\"\$PYENV_ROOT/bin:\$PATH\"' >> ~/.bashrc"
    echo "   echo 'eval \"\$(pyenv init --path)\"' >> ~/.bashrc"
    echo "   echo 'eval \"\$(pyenv init -)\"' >> ~/.bashrc"
    echo "   source ~/.bashrc"
    echo ""
    echo "3. Install Python $REQUIRED_VERSION:"
    echo "   pyenv install $REQUIRED_VERSION"
    echo "   pyenv global $REQUIRED_VERSION"
    echo ""
    echo ""
    echo "Option 3: Use System Package Manager"
    echo "------------------------------------"
    echo "Ubuntu/Debian:"
    echo "   sudo apt-get update"
    echo "   sudo apt-get install python3.10"
    echo ""
    echo "macOS (using Homebrew):"
    echo "   brew install python@3.10"
    echo ""
    echo ""
    echo "Note: After installing Python $REQUIRED_VERSION, please:"
    echo "  1. Close this terminal"
    echo "  2. Open a NEW terminal"
    echo "  3. Run the dashboard script again"
    echo ""
    
    exit 1
fi
