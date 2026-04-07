# Build Instructions — Biocontrol Dashboard Windows Installer

This document explains how to compile the **Biocontrol Dashboard** Windows installer
using **Inno Setup 6.x**.

---

## Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Inno Setup | 6.x | <https://jrsoftware.org/isdl.php> |
| Python installer | 3.10.9 (amd64) | <https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe> |
| DWSIM | 8.x installed locally | <https://dwsim.org/index.php/download/> |

---

## Prepare the `dependencies/` Folder

The `dependencies/` directory is **excluded from git** (it contains ~700 MB of binaries).
You must populate it manually before compiling.

### 1. Download the Python 3.10.9 installer

```bat
mkdir dependencies
curl -L -o dependencies\python-3.10.9-amd64.exe ^
     https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe
```

### 2. Copy the DWSIM installation folder

DWSIM must already be installed on your build machine.

```bat
xcopy "%LOCALAPPDATA%\DWSIM" "dependencies\DWSIM\" /E /I /H /Y
```

After this step the directory tree should look like:

```
installer/
  biocontrol_setup.iss
  post_install.bat
  README_BUILD.md
  Output/               ← created by Inno Setup (git-ignored)
dependencies/           ← git-ignored (~700 MB)
  python-3.10.9-amd64.exe
  DWSIM/
    DWSIM.Automation.dll
    ...
```

---

## Compile the Installer

### Option A — Inno Setup GUI

1. Open **Inno Setup Compiler**.
2. Click **File → Open** and select `installer\biocontrol_setup.iss`.
3. Click **Build → Compile** (or press **F9**).
4. The output file will be:
   `installer\Output\BiocontrolDashboard-Setup-v1.0.0.exe`

### Option B — Command Line

```bat
:: Default ISCC path on most installations
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" ^
    /DAppVersion=1.0.0 ^
    installer\biocontrol_setup.iss
```

---

## What the Installer Does (User View)

| Step | Action |
|------|--------|
| 1 | Welcome screen with component list |
| 2 | Choose installation directory |
| 3 | Choose optional tasks (desktop icon, launch after install) |
| 4 | **Silently installs Python 3.10.9** (skipped if already present) |
| 5 | Copies DWSIM files to `%LOCALAPPDATA%\DWSIM` |
| 6 | Copies all project files to the chosen directory |
| 7 | Runs `post_install.bat`: creates `.venv`, installs Python libraries |
| 8 | Updates `Simulation\config.py` with the actual DWSIM path |
| 9 | Creates Start Menu shortcut (and optional Desktop shortcut) |
| 10 | Optionally launches the dashboard immediately |

---

## Version Update Checklist

When releasing a new version:

1. Update `version.py`:
   ```python
   __version__ = "1.1.0"
   ```
2. Update `setup.py` version field.
3. Add an entry to `CHANGELOG.md`.
4. Commit and push all code changes.
5. Create and push a version tag:
   ```bash
   git tag v1.1.0
   git push origin v1.1.0
   ```
6. GitHub Actions will automatically:
   - Download the Python installer and DWSIM release.
   - Compile the Inno Setup script.
   - Create a GitHub Release with the installer attached.
7. If building locally, update the `/DAppVersion` flag in the ISCC command above.

---

## Troubleshooting

### Inno Setup cannot find source files

Make sure you are running ISCC from the **repository root** or that the working
directory is the repository root. The `.iss` script uses relative paths like
`..\main.py` which are relative to the `installer\` directory.

### Python not detected after installation

The Python 3.10.9 silent installer uses `InstallAllUsers=1 PrependPath=1`.
On some systems the PATH change only takes effect after a reboot.
`post_install.bat` searches common hard-coded paths first to avoid this issue.

### DWSIM path not updated in config.py

Open `Simulation\config.py` and verify that `DWSIM_INSTALL_PATH` now points to
`%LOCALAPPDATA%\DWSIM\`. If not, run `post_install.bat` manually from the
installation directory.

### Dependencies installation fails

Run `post_install.bat` manually from an **elevated** Command Prompt:

```bat
cd "C:\Program Files\BiocontrolDashboard"
post_install.bat "C:\Program Files\BiocontrolDashboard"
```

Check the output for the specific pip error.

### The Digital Twin shows synthetic data only

This is expected when DWSIM is not installed. Install DWSIM 8.x from
<https://dwsim.org> and set `USE_DWSIM_LIVE = True` in
`Simulation\config.py`.
