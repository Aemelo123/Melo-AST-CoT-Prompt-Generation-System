from unittest.mock import patch

from melo_ast_cot.llm_client import get_gpt_response, get_anthropic_response


def test_get_gpt_response():
    with patch("melo_ast_cot.llm_client.OpenAI") as mock:
        mock.return_value.responses.create.return_value.output_text = "hello"
        result = get_gpt_response("test")
        assert result == "hello"


def test_get_anthropic_response():
    with patch("melo_ast_cot.llm_client.Anthropic") as mock:
        mock.return_value.messages.create.return_value.content = [type("Obj", (), {"text": "hello"})()]
        result = get_anthropic_response("test")
        assert result == "hello"
