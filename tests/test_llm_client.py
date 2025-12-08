from unittest.mock import patch

from melo_ast_cot import llm_client


def test_get_gpt_response():
    with patch("melo_ast_cot.llm_client.OpenAI") as mock:
        mock.return_value.chat.completions.create.return_value.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": "hello"})()})()
        ]
        result = llm_client.get_gpt_response("test")
        assert result == "hello"


def test_get_anthropic_response():
    with patch("melo_ast_cot.llm_client.Anthropic") as mock:
        mock.return_value.messages.create.return_value.content = [type("Obj", (), {"text": "hello"})()]
        result = llm_client.get_anthropic_response("test")
        assert result == "hello"
