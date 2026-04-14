#!/bin/bash
# =============================================================================
# Python and DWSIM Version Verification Script for Biocontrol Modeling Project
# =============================================================================

TARGET_PYTHON_VERSION="3.10"
TARGET_DWSIM_VERSION="9.0.5"
TARGET_DOTNET_VERSION="8.0"

echo "============================================================"
echo "Evaluation of Environment for Biocontrol Modeling Project"
echo "============================================================"
echo ""

# Step 1: Evaluate and Install Python 3.10.x
echo "[1/3] Evaluating Python ${TARGET_PYTHON_VERSION}..."
if command -v python3.10 &> /dev/null; then
    echo "[SKIP] Python ${TARGET_PYTHON_VERSION} is already installed."
else
    echo "[INFO] Python ${TARGET_PYTHON_VERSION} not detected. Proceeding with installation..."
    sudo apt-get update -yqq
    sudo apt-get install -yqq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -yqq
    sudo apt-get install -yqq python3.10 python3.10-venv python3.10-dev
fi
echo ""

# Step 2: Evaluate and Install DWSIM
echo "[2/3] Evaluating DWSIM v${TARGET_DWSIM_VERSION}..."
if [ -d "/usr/lib/dwsim" ] && [ -f "/usr/lib/dwsim/DWSIM.Automation.dll" ]; then
    echo "[SKIP] DWSIM detected in /usr/lib/dwsim."
    export DWSIM_INSTALL_PATH=/usr/lib/dwsim
else
    echo "[INFO] DWSIM not found. Downloading official .deb package..."
    wget -q --show-progress "https://github.com/DanWBR/dwsim/releases/download/v${TARGET_DWSIM_VERSION}/dwsim_${TARGET_DWSIM_VERSION}-amd64.deb" -O /tmp/dwsim.deb
    sudo apt-get install -yqq /tmp/dwsim.deb
    rm /tmp/dwsim.deb
    export DWSIM_INSTALL_PATH=/usr/lib/dwsim
fi
echo ""

# Step 3: Evaluate and Install .NET Runtime (>= 8.0)
echo "[3/3] Evaluating .NET Runtime (requires >= ${TARGET_DOTNET_VERSION})..."
DOTNET_OK=0
if command -v dotnet &> /dev/null; then
    # Extraer las versiones principales de los runtimes instalados
    while read -r line; do
        VERSION=$(echo "$line" | awk '{print $2}')
        MAJOR_VERSION=$(echo "$VERSION" | cut -d'.' -f1)
        if [ "$MAJOR_VERSION" -ge 8 ]; then
            DOTNET_OK=1
            break
        fi
    done <<< "$(dotnet --list-runtimes 2>/dev/null | grep "Microsoft.NETCore.App")"
fi

if [ "$DOTNET_OK" -eq 1 ]; then
    echo "[SKIP] .NET Runtime >= ${TARGET_DOTNET_VERSION} is already configured."
else
    echo "[INFO] .NET ${TARGET_DOTNET_VERSION} (or higher) not detected. Installing..."
    UBUNTU_VERSION=$(lsb_release -rs)
    wget -q https://packages.microsoft.com/config/ubuntu/${UBUNTU_VERSION}/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb
    sudo dpkg -i /tmp/packages-microsoft-prod.deb
    rm /tmp/packages-microsoft-prod.deb
    
    sudo apt-get update -yqq
    sudo apt-get install -yqq dotnet-runtime-8.0
fi
echo ""

echo "============================================================"
echo "Verification completed successfully."
echo "============================================================"
exit 0