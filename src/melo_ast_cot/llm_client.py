from openai import OpenAI
from anthropic import Anthropic


# Standardized generation parameters per methodology
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.95


# (OpenAI, n.d.) https://platform.openai.com/docs/quickstart
def get_gpt_response(prompt: str, model: str = "gpt-4-0613") -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        top_p=DEFAULT_TOP_P,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


def get_gpt_response_json(prompt: str, model: str = "gpt-4-0613") -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        top_p=DEFAULT_TOP_P,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content


# (Anthropic, n.d.) https://platform.claude.com/docs/en/get-started#python
def get_anthropic_response(prompt: str, model: str = "claude-sonnet-4-5-20250929") -> str:
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=DEFAULT_TEMPERATURE
    )
    return response.content[0].text