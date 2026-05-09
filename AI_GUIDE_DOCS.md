# AI Guide - User Documentation

## Overview
The AI Guide is an integrated LLM-powered assistant that helps users understand bioprocess modeling concepts, equations, methods, and parameters. It uses Ollama to run free, open-source language models locally.

## Features

### 1. Equation Explanations
The AI Guide can explain any equation shown in the application:
- **Variables:** What each symbol means
- **Units:** Dimensional analysis
- **Assumptions:** Model limitations
- **Interpretation:** Physical meaning

**Example question:** "¿Qué significa μmax en la ecuación de Monod?"

### 2. Method Descriptions
Get clear explanations of simulation and control methods:
- **Batch/Fed-Batch/Continuous** modeling
- **PID control** tuning
- **EKF/ANN** state estimation
- **RTO/NMPC** optimization

**Example question:** "Explica cómo funciona el control predictivo NMPC"

### 3. Parameter Suggestions
Receive typical parameter ranges from academic literature:
- Kinetic parameters (μmax, Ks, Yxs)
- Controller gains (Kc, Ti, Td)
- Process conditions (kLa, yields)

**Example question:** "¿Cuáles son valores típicos para Ks en cultivos bacterianos?"

### 4. Bibliographic References
Get curated academic references relevant to your current page:
- Classic textbooks (Bailey & Ollis, Shuler & Kargi)
- Control theory (Smith & Corripio, Camacho & Bordons)
- Specific papers (Monod, Luedeking-Piret)

**Example question:** "Dame referencias sobre Extended Kalman Filter en bioprocesos"

## Installation Guide

### Step 1: Install Ollama

**Linux/Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download the installer from [ollama.com/download](https://ollama.com/download)

### Step 2: Download a Model

Choose one based on your system resources:

**Recommended for most users (7-8B parameters):**
```bash
ollama pull llama3.1:8b    # Meta's Llama 3.1, well-balanced
ollama pull qwen2.5:7b     # Good multilingual support
```

**For faster responses (3B parameters):**
```bash
ollama pull llama3.2:3b    # Smaller, quicker, still capable
ollama pull phi3:mini      # Microsoft Phi-3
```

**For better quality (13B+ parameters):**
```bash
ollama pull llama3.1:13b   # Requires more RAM (>16GB)
```

### Step 3: Start Ollama Server

```bash
ollama serve
```

Keep this running in a terminal. On some systems, Ollama starts automatically as a service.

### Step 4: Enable in the Application

1. Start the Bioprocess Modeling application:
   ```bash
   streamlit run main.py
   ```

2. In the sidebar, scroll to **"🤖 AI Guide (Beta)"**

3. Check **"Activar Asistente IA"**

4. Click **"🔍 Verificar Conexión"** to test connectivity

5. If successful, you'll see "✅ Conectado. Modelos disponibles: ..."

## Using the AI Guide

### Quick Actions

Two buttons provide instant help:

1. **📖 Explicar método**: Explains the current page's main method/model
2. **📊 Sugerir parámetros**: Suggests typical parameter ranges

### Custom Questions

Type your question in the text area and click **🚀 Preguntar**:

**Good questions:**
- "¿Qué es la constante Ks y cómo afecta el crecimiento?"
- "Explica la diferencia entre control RTO y NMPC"
- "¿Cómo se sintoniza un PID para temperatura?"
- "¿Qué rangos típicos usa Yxs en fermentaciones aerobias?"

**Tips for better answers:**
- Be specific about what you want to know
- Mention the equation/parameter name if known
- Ask one question at a time for clearer answers

### Chat History

The last 3 interactions are saved in the sidebar. Click on any to review the Q&A.

Click **🗑️ Limpiar** to clear the history.

## Configuration

### ⚙️ Settings (Expander)

- **URL de Ollama**: Change if running on a different host/port (default: http://localhost:11434)
- **Modelo**: Select which Ollama model to use from your installed models
- **Verificar Conexión**: Test connectivity to Ollama server

### Model Selection

Different models offer different trade-offs:

| Model | Size | Speed | Quality | RAM Needed |
|-------|------|-------|---------|------------|
| llama3.2:3b | 2GB | ⚡⚡⚡ | ⭐⭐ | 4GB+ |
| llama3.1:8b | 4.7GB | ⚡⚡ | ⭐⭐⭐ | 8GB+ |
| qwen2.5:7b | 4.4GB | ⚡⚡ | ⭐⭐⭐ | 8GB+ |
| llama3.1:13b | 7.4GB | ⚡ | ⭐⭐⭐⭐ | 16GB+ |

**Recommendation:** Start with `llama3.1:8b` for the best balance.

## Troubleshooting

### "No se puede conectar con Ollama"

**Cause:** Ollama server is not running or not accessible.

**Solutions:**
1. Start the server: `ollama serve`
2. Check if it's running: `curl http://localhost:11434/api/tags`
3. Verify firewall settings if using remote Ollama
4. Check the URL in configuration matches your setup

### "La solicitud tardó demasiado tiempo"

**Cause:** Model is too large for your hardware or busy with another request.

**Solutions:**
1. Switch to a smaller model (e.g., `llama3.2:3b`)
2. Close other applications to free RAM
3. Wait for current query to finish
4. Increase timeout (requires code modification)

### "Error del servidor: 404"

**Cause:** Selected model is not installed on your system.

**Solutions:**
1. Check installed models: `ollama list`
2. Pull the missing model: `ollama pull <model_name>`
3. Select an installed model in the configuration

### Responses in Wrong Language

**Cause:** Some models default to English despite Spanish prompts.

**Solutions:**
1. Try `qwen2.5:7b` which has better multilingual support
2. Phrase questions in Spanish explicitly
3. Add "Responde en español:" at the start of your question

### App Works but AI Guide Doesn't Appear

**Cause:** Import error or missing dependencies.

**Solutions:**
1. Check that `requests` is installed: `pip install requests`
2. Verify files exist: `Utils/llm_helper.py` and `Utils/llm_ui_component.py`
3. Check console for error messages
4. The app is designed to work normally even if AI Guide fails

## Best Practices

### Do's ✅
- **Use for learning:** Understanding concepts and equations
- **Cross-check information:** Verify suggestions with experiments
- **Start with quick actions:** Good way to explore capabilities
- **Keep Ollama updated:** `ollama pull <model>` to update models
- **Report issues:** Help improve the feature

### Don'ts ❌
- **Don't use for critical decisions:** This is an educational tool
- **Don't trust blindly:** Always validate parameter suggestions experimentally
- **Don't expect 100% accuracy:** LLMs can make mistakes
- **Don't share sensitive data:** Keep process-specific info confidential
- **Don't use without citations:** Always include provided references in reports

## Privacy & Data

### What Stays Private
- **All processing is local:** When using local Ollama, no data leaves your machine
- **No API keys needed:** Free and open-source
- **No telemetry:** The AI Guide doesn't send usage data anywhere

### What's Shared
- **Nothing by default:** With local Ollama, everything is private
- **If using remote Ollama:** Your questions/responses would go to that server

## Academic Integrity

The AI Guide provides **curated bibliographic references** from the project's documentation:
- Bailey & Ollis (1986)
- Shuler & Kargi (2002)
- Smith & Corripio (2005)
- Camacho & Bordons (2007)
- And many more...

**Always cite these references** when using information in academic work, not the AI Guide itself.

Example citation:
> "The Monod equation describes microbial growth kinetics with substrate limitation (Monod, 1949; Shuler & Kargi, 2002)."

## Feedback & Contributing

Found a bug? Have a suggestion? Want to improve the curated references?

1. Open an issue on GitHub
2. Contact: cesar.garech@gmail.com
3. Contribute via pull request

## Version History

**v1.0.0 (Current)**
- Initial implementation
- Support for 5 Ollama models
- 9 reference categories
- Contextual help per page
- Parameter range suggestions
- Chat history (last 3)
- Graceful fallback if Ollama unavailable

## License

The AI Guide is part of the Biocontrol Modeling Project and shares its license.
The LLM models (Llama, Qwen, etc.) have their own licenses - check Ollama documentation.

---

**Remember:** The AI Guide is a tool to enhance learning, not replace it. Always validate, experiment, and think critically! 🧪🔬
