# Changelog

## [1.0.4] - 2026-06-16 — Checkpoint: Fix biocontrol-llama Ollama API errors (Test_Chatbot)

### 🤖 AI Guide / Llama Chatbot — Bug Fix

This checkpoint repairs the customized **`biocontrol-llama`** workflow, which broke
against newer Ollama releases (v0.5+). The base models continued to work, but
building/using the grounded custom model failed. The developer should recompile
after this checkpoint to produce `BiocontrolDashboard-Setup-v1.0.4.exe`.

#### Fixed
- **`create_custom_model` → `400 - {"error":"neither 'from' or 'files' was specified"}`.**
  Ollama v0.5+ replaced the deprecated `modelfile` body field of `/api/create`
  with a structured schema. `Utils/llm_helper.py` now sends the modern
  `from`/`system`/`parameters` payload and transparently falls back to the legacy
  `modelfile` payload for older Ollama servers. A pre-flight check verifies the
  base model is installed and returns an actionable message if it is not.
- **`pull_ollama_model` → `Server error while pulling: 500`.** Attempting to
  `pull` the local-only `biocontrol-llama` model from the public registry returned
  500. The function now detects the custom model and instructs the user to pull the
  base model and build the assistant instead. Pull also sends the new `model` key
  (keeping `name` for backward compatibility) and surfaces the server's error body.
- **`query_ollama` → `Server error: 404`.** Querying `biocontrol-llama` before it
  was built returned an opaque 404. The 404 case now returns a clear, actionable
  message (pull the base model and build the custom assistant, or pull the selected
  base model).

#### Added
- **`model_exists()`** helper in `Utils/llm_helper.py` to check whether a model
  (exact tag or bare name) is installed locally.
- **`CUSTOM_MODEL_PARAMETERS`** constant shared by the Modelfile builder and the
  modern `/api/create` payload to keep generation parameters in a single source.
- **`_extract_error_message()`** helper to parse Ollama's JSON `error` field for
  clearer UI messages.
- README **"🛠️ Troubleshooting the `biocontrol-llama` model"** section explaining
  the errors, root causes, fixes and the correct setup order.

#### Changed
- Version bumped to **1.0.4** in `version.py`, `setup.py` and
  `installer/biocontrol_setup.iss`.

---

## [1.0.3] - 2026-06-15 — Checkpoint: AI Guide grounded on all screens (Test_Chatbot)

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
