from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Utils.llm_helper import (
    BASE_MODEL,
    CUSTOM_MODEL_NAME,
    SYSTEM_PROMPT,
    build_modelfile,
    create_custom_model,
    query_ollama,
)


class TestLLMHelper(unittest.TestCase):
    def test_build_modelfile_embeds_current_prompt_and_recommended_settings(self):
        modelfile = build_modelfile("qwen2.5:7b")

        self.assertTrue(modelfile.startswith("FROM qwen2.5:7b\n"))
        self.assertIn("PARAMETER temperature 0.3", modelfile)
        self.assertIn(SYSTEM_PROMPT, modelfile)
        self.assertIn("same language as the user's question", modelfile)

    @patch("Utils.llm_helper.subprocess.run")
    def test_create_custom_model_uses_ollama_cli_and_generated_modelfile(self, mock_run):
        modelfile_contents = []

        def run_cli(command, **kwargs):
            modelfile_path = command[command.index("-f") + 1]
            with open(modelfile_path, encoding="utf-8") as modelfile:
                modelfile_contents.append(modelfile.read())
            return SimpleNamespace(returncode=0, stdout="success", stderr="")

        mock_run.side_effect = run_cli
        success, message = create_custom_model(
            "http://ollama.example:11434",
            "qwen2.5:7b",
            "biocontrol-test",
        )

        self.assertTrue(success)
        self.assertIn("biocontrol-test", message)
        self.assertEqual(
            mock_run.call_args.args[0][:4],
            ["ollama", "create", "biocontrol-test", "-f"],
        )
        self.assertEqual(mock_run.call_args.kwargs["env"]["OLLAMA_HOST"], "http://ollama.example:11434")
        self.assertTrue(modelfile_contents[0].startswith("FROM qwen2.5:7b\n"))
        self.assertEqual(mock_run.call_args.kwargs["timeout"], 600)

    @patch("Utils.llm_helper.subprocess.run", side_effect=FileNotFoundError)
    def test_create_custom_model_reports_missing_ollama_cli(self, _mock_run):
        success, message = create_custom_model()

        self.assertFalse(success)
        self.assertIn("Ollama CLI not found", message)

    @patch("Utils.llm_helper.requests.post")
    def test_query_ollama_sends_system_prompt_separately_for_base_model(self, mock_post):
        mock_post.return_value = SimpleNamespace(
            status_code=200,
            json=lambda: {"response": "Respuesta"}
        )

        success, response = query_ollama("Contexto de la página", model=BASE_MODEL)

        self.assertTrue(success)
        self.assertEqual(response, "Respuesta")
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["prompt"], "Contexto de la página")
        self.assertEqual(payload["system"], SYSTEM_PROMPT)
        self.assertEqual(payload["options"]["temperature"], 0.3)

    @patch("Utils.llm_helper.requests.post")
    def test_query_ollama_does_not_override_custom_model_system_prompt(self, mock_post):
        mock_post.return_value = SimpleNamespace(
            status_code=200,
            json=lambda: {"response": "Respuesta"}
        )

        success, _ = query_ollama("Contexto de la página", model=CUSTOM_MODEL_NAME)

        self.assertTrue(success)
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["prompt"], "Contexto de la página")
        self.assertNotIn("system", payload)
