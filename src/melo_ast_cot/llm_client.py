from openai import OpenAI
from anthropic import Anthropic

gpt_client = OpenAI()
anthropic_client = Anthropic()


def get_gpt_response(prompt: str, model: str = "gpt-5-nano") -> str:
    response = gpt_client.responses.create(
        model = model,
        input = prompt
    )
    return response.output_text


def get_anthropic_response(prompt: str, model: str = "claude-sonnet-4-5", max_tokens = 1000) -> str:
    response = anthropic_client.messages.create(
        model = model,
        max_tokens = max_tokens,
        messages = [
            {
            "role": "user",
            "content": prompt
            }
        ]
    )
    return response.content[0].text