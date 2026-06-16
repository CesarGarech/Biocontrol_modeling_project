# Changelog

## [1.0.3] - 2026-06-15 — Checkpoint: AI Guide grounded on all screens (Test_Chatbot)

### 🔌 AI Guide / Ollama connection fix (2026-06-16, within v1.0.3)

Fixes **"❌ Cannot connect to Ollama. Make sure it is running (ollama serve)"**,
where no model connected or downloaded. Root cause: the everyday launcher
`run_dashboard.bat` never started the Ollama **server** — only `post_install.bat`
started it once at install time, so after a reboot or closing the tray app the
server was down. See **`TROUBLESHOOTING.md`** for the full analysis.

#### Fixed
- **`run_dashboard.bat`** — added an `EnsureOllama` step that detects whether the
  Ollama API answers on `http://localhost:11434`, starts the server
  (`ollama app.exe` / `ollama serve` / `ollama` on PATH) if not, and waits for it
  to come up before launching Streamlit. The chatbot now connects on every run.
- **`installer/post_install.bat`** — the Ollama startup step now waits and confirms
  the server is reachable instead of fire-and-forget.

#### Added
- **`TROUBLESHOOTING.md`** — problem analysis, fix description, verification and
  manual recovery steps for the Ollama connection error (also shipped by the
  installer and linked from the README).

> Version intentionally kept at **1.0.3** (no checkpoint bump for this fix).

### 🤖 AI Guide / Llama Chatbot — Major Enhancement

This checkpoint makes the Ollama/Llama-powered **AI Guide** context-aware for **every**
screen of the application and able to suggest screen-specific parameters. The developer
should recompile after this checkpoint to produce `BiocontrolDashboard-Setup-v1.0.3.exe`.

#### Added
- **`Utils/llm_knowledge_base.py`** — Single source of truth that *grounds* the Llama model.
  Contains a curated, per-screen knowledge base (`PAGE_KNOWLEDGE`) for all 24 screens:
  - Modeling: `Batch`, `Fed-Batch`, `Continuous`, `Fermentation`
  - Analysis: `Sensitivity Analysis`
  - Parameter Optimization: `Batch / Fed-Batch / Fermentation Parameter Adjustment`
  - State Estimation: `EKF`, `ANN`
  - Regulatory Control: `Identification (pH)`, `Temperature`, `pH`, `Oxygen`,
    `Cascade-Oxygen`, `On-Off Feeding`
  - Advanced Control: `RTO`, `RTO Ferm`, `NMPC`, `LMPC`, `EKF-NMPC`, `Fuzzy Control`
  - Digital Twin: distillation (DWSIM/FUG), SCADA reconciliation, ML prediction
  - Each entry includes the method, description, governing equations, tunable parameters
    (typical value + range + unit + description) and curated references.
  - Page-name normalization (`resolve_page_key`) handles emoji/aggregate menu labels.
- **`installer/ollama/biocontrol_assistant.Modelfile`** — Ollama Modelfile that customizes
  ("re-trains"/specializes) `llama3.1:8b` into the grounded `biocontrol-llama` model by
  embedding the Biocontrol system prompt and recommended generation parameters.
- **`installer/ollama/build_model.py`** — CLI helper to build the `biocontrol-llama` model
  from the command line (`python installer/ollama/build_model.py`).
- **"🛠️ Build Custom Assistant"** button in the AI Guide sidebar to create the
  `biocontrol-llama` model directly from the app via the Ollama `/api/create` endpoint.
- **Offline page parameter table** — the sidebar now shows the typical parameter ranges of
  the current screen even when Ollama is not running.
- **Context indicator** — the sidebar shows which application area/method the assistant is
  currently grounded on.

#### Changed
- **`Utils/llm_helper.py`**:
  - `SYSTEM_PROMPT` rewritten to describe the full application scope (all 7 areas).
  - `build_context_prompt` now auto-injects the per-page method, equations and parameter
    ranges from the knowledge base, so answers/suggestions are specific to the active screen.
  - `get_relevant_references` now returns curated references for **every** screen via the
    knowledge base (with a legacy fallback for unknown pages).
  - `suggest_parameter_ranges` is now **page-aware** (uses the current screen's parameters
    first, then a global fallback table).
  - Added `BASE_MODEL`, `CUSTOM_MODEL_NAME`, `build_modelfile()` and `create_custom_model()`.
  - `query_ollama` no longer double-applies the system prompt when the customized
    `biocontrol-llama` model is used.
  - The customized `biocontrol-llama` is now the default model in `AVAILABLE_MODELS`.
- **`Utils/llm_ui_component.py`**: imports the knowledge base, injects page context into all
  prompts, page-aware "📊 Suggest params" action, "Build Custom Assistant" workflow.

#### Version
- Bumped application version to **1.0.3** (`version.py`, `setup.py`,
  `installer/biocontrol_setup.iss` → output `BiocontrolDashboard-Setup-v1.0.3.exe`).

---

## [Unreleased] - 2025-10-31

### Repository Cleanup and Restructuring

#### Removed
- **St_CABBIO03.py** - Legacy file that duplicated functionality in main.py (99KB)
  - This was an old version of the dashboard application that is no longer needed
  - All functionality is now consolidated in main.py

#### Renamed
- **teste/** → **test_data/** - Improved directory naming for clarity
  - Contains test data files for simulations
  
- **Body/estimacion_parametros/** → **Body/parameter_estimation/** - Consistent English naming
  - Maintains consistency with other directory names
  - Updated all imports in main.py to reflect the change

#### Added
- **Package Structure** - Added `__init__.py` files to all packages:
  - `Body/__init__.py` - Main package documentation
  - `Body/modeling/__init__.py` - Modeling modules documentation
  - `Body/parameter_estimation/__init__.py` - Parameter estimation documentation
  - `Body/estimation/__init__.py` - State estimation documentation
  - `Body/control/__init__.py` - Control modules documentation
  - `Body/control/regulatorio/__init__.py` - Regulatory control documentation
  - `Body/control/avanzado/__init__.py` - Advanced control documentation

- **setup.py** - Proper Python package configuration
  - Enables installation via `pip install -e .`
  - Defines package metadata and dependencies
  - Supports development and production installations

- **.gitignore updates** - Added build/dist exclusions
  - Excludes `build/`, `dist/`, `*.egg-info/` directories
  - Prevents package build artifacts from being committed

#### Modified
- **README.md** - Updated with:
  - Detailed repository structure tree
  - Improved installation instructions (Quick Start + Manual)
  - Package-based installation option
  - More comprehensive directory descriptions

- **main.py** - Updated imports:
  - Changed `from Body.estimacion_parametros` to `from Body.parameter_estimation`
  - All three parameter estimation modules updated

### Testing
- All package imports verified successfully
- All module functions tested and working
- Application runs without errors
- No breaking changes to functionality

### Impact
- **No functional changes** - All features work exactly as before
- **Improved code organization** - Clearer package structure
- **Better maintainability** - Standard Python package conventions
- **Easier installation** - Can now be installed as a package
- **Reduced repository size** - Removed 99KB of duplicate code
