# 🛠️ Troubleshooting — AI Guide / Llama Chatbot (Ollama)

This guide explains the **"❌ Cannot connect to Ollama. Make sure it is running
(ollama serve)"** error reported in the AI Guide, why it happens, and the fix
shipped in **v1.0.3**.

---

## 1. Symptom

In the sidebar **🤖 AI Guide**:

```
❌ Cannot connect to Ollama. Make sure it is running (ollama serve)
```

- No model connects.
- No model can be downloaded/pulled.
- The rest of the application (modeling, control, digital twin) works normally.

---

## 2. Root-cause analysis

The AI Guide talks to a **local Ollama server** over its REST API at
`http://localhost:11434`. The check that produces the error lives in
`Utils/llm_helper.py`:

```python
def check_ollama_availability(base_url="http://localhost:11434"):
    response = requests.get(f"{base_url}/api/tags", timeout=2)
    ...
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Make sure it is running (ollama serve)"
```

A `ConnectionError` means **nothing is listening on port 11434** — i.e. the
Ollama **server process is not running**. Every other AI Guide feature (pulling
a model, building `biocontrol-llama`, asking questions) depends on that server,
so when it is down, *nothing* connects or downloads. This is a **runtime/process
problem, not a Python code bug.**

### Why the server ends up down

The Ollama **server was only ever started once**, during installation, by
`installer/post_install.bat`:

```bat
:: Step 4 — Evaluate, Install and Start Ollama   (post_install.bat)
start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
```

But the **everyday launcher**, `run_dashboard.bat` — the shortcut the user clicks
every time — **did not start Ollama at all**. It only activated the Python
virtual environment and ran Streamlit:

```bat
:Launch
streamlit run main.py        <-- Ollama never started here
```

So the server is up *right after installing*, but on any later launch it is down
whenever:

- the machine was **rebooted** (the tray app is not registered to auto-start for
  every scenario), or
- the user **closed the Ollama tray icon**, or
- a previous **uninstall/reinstall cycle** left the tray app not running.

The result is exactly the reported symptom: the dashboard opens, but the chatbot
cannot reach Ollama.

### About the uninstall hypothesis

A previous suspicion was that uninstalling v1.0.3 also uninstalled Ollama. That
is **not** the case — `installer/biocontrol_setup.iss` only removes the app and
its virtual environment on uninstall:

```ini
[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\BiocontrolDashboard\.venv"
Type: filesandordirs; Name: "{localappdata}\BiocontrolDashboard"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\Output"
```

Ollama is **not** in that list, so it is left installed. The visible symptom is
identical, though: after a reinstall the dashboard launches without the Ollama
**server** running.

### Is Ollama installed during installation? — Yes

Ollama **is** installed during the installation phase, in two complementary
places, and both *skip* if it is already present:

| Location | What it does |
| --- | --- |
| `installer/biocontrol_setup.iss` → `InstallOllama` | Runs the bundled `OllamaSetup.exe` if `…\Programs\Ollama\ollama.exe` is missing (`Check: not IsOllamaInstalled`). |
| `installer/post_install.bat` → Step 4 | Detects Ollama, downloads `OllamaSetup.exe` as a fallback if missing, installs it, and **starts the server**. |

So installation/detection was fine. The gap was purely that **the launcher never
(re)started the server on subsequent runs.**

---

## 3. The fix (shipped in v1.0.3)

The launcher now **ensures the Ollama server is running before starting the
dashboard**, so the chatbot connects on every launch — no manual `ollama serve`
needed.

### `run_dashboard.bat` — new `EnsureOllama` step

Before `streamlit run main.py`, the launcher:

1. Checks whether the API answers on `http://localhost:11434/api/tags`.
2. If not, starts `ollama app.exe` (tray app) or `ollama.exe serve`, falling back
   to `ollama` on `PATH`.
3. Waits up to ~20 s for the API to become reachable, then launches the app.
4. If Ollama is not installed at all, it prints a clear message and continues
   (the rest of the dashboard still works).

### `installer/post_install.bat` — verified startup

The post-install Ollama step now **waits and confirms** the server actually came
up (instead of firing-and-forgetting), so the first launch is already connected.

---

## 4. How to verify

1. Launch the dashboard via **`run_dashboard.bat`** (or its Start-menu/desktop
   shortcut). The console shows:
   - `[INFO] Ollama server already running.`  *or*
   - `[INFO] Ollama server not responding. Attempting to start it...` followed by
     `[INFO] Ollama server is now running.`
2. In the sidebar, enable **🤖 AI Guide** → **⚙️ Settings** → **🔍 Check
   Connection**. You should see **✅ Connected**.
3. Select a base model (e.g. `llama3.1:8b`) and click **⬇️ Download Model**; the
   pull now succeeds because the server is up.

---

## 5. Manual recovery (if you ever need it)

If the chatbot still cannot connect, start the server by hand:

```bat
:: Option A — start the tray app
"%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"

:: Option B — start the server in a terminal
ollama serve
```

Then verify it is listening:

```bat
curl http://localhost:11434/api/tags
```

A JSON response (even with an empty `models` list) means the server is up. If
`ollama` is not recognized, reinstall it from <https://ollama.com/download>.

---

## 6. Quick reference

| Check | Command / Action | Healthy result |
| --- | --- | --- |
| Server reachable | `curl http://localhost:11434/api/tags` | JSON with `models` |
| Ollama installed | `where ollama` | A path is printed |
| Start server | `ollama serve` or run `ollama app.exe` | API becomes reachable |
| In-app test | AI Guide → **🔍 Check Connection** | ✅ Connected |
