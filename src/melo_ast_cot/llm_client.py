from openai import OpenAI
from anthropic import Anthropic


# (OpenAI, n.d.) https://platform.openai.com/docs/quickstart
def get_gpt_response(prompt: str, model: str = "gpt-5-nano") -> str:
    client = OpenAI()
    response = client.responses.create(
        model = model,
        input = prompt
    )
    return response.output_text


# (Anthropic, n.d.) https://platform.claude.com/docs/en/get-started#python
def get_anthropic_response(prompt: str, model: str = "claude-sonnet-4-5", max_tokens = 1000) -> str:
    client = Anthropic()
    response = client.messages.create(
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