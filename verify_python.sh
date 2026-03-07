#!/bin/bash
# Python Version Verification Script for Biocontrol Modeling Project
# This script verifies Python 3.10.9 is installed and provides installation guidance

REQUIRED_VERSION="3.10.9"
PYTHON_DOWNLOAD_PAGE="https://www.python.org/downloads/release/python-3109/"

echo "============================================================"
echo "Python Version Verification for Biocontrol Modeling Project"
echo "============================================================"
echo ""
echo "Required Python Version: $REQUIRED_VERSION (or compatible 3.10.x)"
echo ""

# Determine which Python command to use
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    # Check if python points to Python 3
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    if [[ $PYTHON_VERSION == 3.* ]]; then
        PYTHON_CMD="python"
    fi
fi

# Check if a suitable python command is available
if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3 is not installed or not in PATH"
    INSTALL_NEEDED=1
else
    # Get Python version
    CURRENT_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    
    echo "Current Python Version: $CURRENT_VERSION (using command: $PYTHON_CMD)"
    echo ""
    
    # Extract major and minor version
    CURRENT_MAJOR_MINOR=$(echo "$CURRENT_VERSION" | cut -d. -f1,2)
    REQUIRED_MAJOR_MINOR="3.10"
    
    # Check if version is compatible (3.10.x)
    if [ "$CURRENT_MAJOR_MINOR" = "$REQUIRED_MAJOR_MINOR" ]; then
        echo "[SUCCESS] Python 3.10.x is correctly installed!"
        echo ""
        exit 0
    fi
    
    # Version doesn't match - show warning
    echo "[WARNING] Python version mismatch detected!"
    echo ""
    echo "The current Python version ($CURRENT_VERSION) may cause compatibility issues."
    echo "This project requires Python 3.10.x specifically due to:"
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
    echo "2. Add pyenv to your shell configuration:"
    echo "   For bash (~/.bashrc) or zsh (~/.zshrc), add:"
    echo "   export PYENV_ROOT=\"\$HOME/.pyenv\""
    echo "   export PATH=\"\$PYENV_ROOT/bin:\$PATH\""
    echo "   eval \"\$(pyenv init --path)\""
    echo "   eval \"\$(pyenv init -)\""
    echo "   Then restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
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
