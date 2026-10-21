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

    @patch("Utils.llm_helper.requests.post")
    def test_create_custom_model_uses_current_ollama_api_field(self, mock_post):
        mock_post.return_value = SimpleNamespace(status_code=200)

        success, message = create_custom_model()

        self.assertTrue(success)
        self.assertIn(CUSTOM_MODEL_NAME, message)
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:11434/api/create")
        self.assertEqual(kwargs["json"]["model"], CUSTOM_MODEL_NAME)
        self.assertNotIn("name", kwargs["json"])
        self.assertFalse(kwargs["json"]["stream"])
        self.assertTrue(kwargs["json"]["modelfile"].startswith(f"FROM {BASE_MODEL}\n"))

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
