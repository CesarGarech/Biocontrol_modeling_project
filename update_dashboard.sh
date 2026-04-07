#!/usr/bin/env bash
# Biocontrol Dashboard - Updater (Linux/macOS)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Biocontrol Dashboard - Updater"
echo "============================================================"
echo ""

# ---- Step 1: Try Git first ----
echo "[1/3] Checking for Git..."
if command -v git &>/dev/null; then
    echo "      Git found. Pulling latest code from GitHub..."
    if git pull; then
        echo "      Code updated successfully via Git."
    else
        echo "      Git pull failed. Falling back to ZIP download..."
        UPDATE_VIA_ZIP=1
    fi
else
    echo "      Git not found. Using ZIP download..."
    UPDATE_VIA_ZIP=1
fi

# ---- Step 2: ZIP download fallback ----
if [ "${UPDATE_VIA_ZIP:-0}" = "1" ]; then
    ZIP_URL="https://github.com/CesarGarech/Biocontrol_modeling_project/archive/refs/heads/main.zip"
    TEMP_ZIP="/tmp/biocontrol_update.zip"
    TEMP_DIR="/tmp/biocontrol_update"

    echo "[1/3] Downloading latest code from GitHub (ZIP)..."
    if command -v curl &>/dev/null; then
        curl -fsSL "$ZIP_URL" -o "$TEMP_ZIP"
    elif command -v wget &>/dev/null; then
        wget -q "$ZIP_URL" -O "$TEMP_ZIP"
    else
        echo "ERROR: Neither curl nor wget is available. Cannot download the update."
        exit 1
    fi

    echo "      Extracting files..."
    rm -rf "$TEMP_DIR"
    unzip -q "$TEMP_ZIP" -d "$TEMP_DIR"

    echo "      Copying updated files (preserving .venv, dependencies, installer)..."
    rsync -a --exclude='.venv' --exclude='dependencies' --exclude='installer' \
              --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
              "$TEMP_DIR/Biocontrol_modeling_project-main/" "$SCRIPT_DIR/"

    echo "      Cleaning up temporary files..."
    rm -f "$TEMP_ZIP"
    rm -rf "$TEMP_DIR"

    echo "      Code updated successfully via ZIP download."
fi

# ---- Step 3: Update Python libraries ----
echo "[2/3] Activating virtual environment..."
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "WARNING: Virtual environment not found at .venv/bin/activate"
    echo "         Skipping library update."
else
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"

    echo "[3/3] Updating Python libraries..."
    if pip install -r requirements.txt --upgrade; then
        echo "      Libraries updated successfully."
    else
        echo "WARNING: Some libraries could not be updated. Check your internet connection."
    fi
fi

echo ""
echo "============================================================"
echo "  Update complete! Run ./run_dashboard.sh to start the app."
echo "============================================================"
echo ""
