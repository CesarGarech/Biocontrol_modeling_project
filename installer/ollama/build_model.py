# -*- coding: utf-8 -*-
"""
Build the customized Biocontrol Ollama model (``biocontrol-llama``).

This is a thin command-line wrapper around
``Utils.llm_helper.create_custom_model`` so the grounded assistant model can be
created without opening the Streamlit UI. The base model (default
``llama3.1:8b``) must already be pulled in Ollama.

Usage:
    python installer/ollama/build_model.py
    python installer/ollama/build_model.py --base-model llama3.2:3b
    python installer/ollama/build_model.py --url http://localhost:11434
"""

import argparse
import os
import sys

# Allow running the script directly from anywhere by adding the project root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Utils.llm_helper import (  # noqa: E402
    BASE_MODEL,
    CUSTOM_MODEL_NAME,
    DEFAULT_OLLAMA_URL,
    check_ollama_availability,
    create_custom_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Biocontrol Ollama assistant model.")
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="Ollama server URL.")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base Llama model to build from.")
    parser.add_argument("--name", default=CUSTOM_MODEL_NAME, help="Name of the customized model.")
    args = parser.parse_args()

    available, message = check_ollama_availability(args.url)
    print(f"Ollama: {message}")
    if not available:
        print("ERROR: Ollama is not reachable. Start it with 'ollama serve' and pull the base model.")
        return 1

    print(f"Creating '{args.name}' from '{args.base_model}' ...")
    success, msg = create_custom_model(args.url, args.base_model, args.name)
    print(msg)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
