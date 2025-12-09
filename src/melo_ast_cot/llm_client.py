from openai import OpenAI
from anthropic import Anthropic


# 0.7 is a temperature that strikes an optimal balance between output diversity and
# code correctness in most cases, although it can be anywhere in the range of 0.4 to 0.8.
# Reference: DAIR.AI, "Prompt Engineering Guide,"
#   https://www.promptingguide.ai/introduction/settings
DEFAULT_TEMPERATURE = 0.7

# 2048 tokens is sufficient for function implementations
DEFAULT_MAX_TOKENS = 2048

# Nucleus sampling (top_p=0.95) filters out low-probability tokens while maintaining diversity
# Reference: Holtzman et al., "The Curious Case of Neural Text Degeneration," ICLR 2020
#   https://arxiv.org/abs/1904.09751
DEFAULT_TOP_P = 0.95


# Based on OpenAI's quickstart guide (OpenAI, n.d.) 
# https://platform.openai.com/docs/quickstart
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


# Based on Anthropic's quickstart guide (Anthropic, n.d.) 
# https://docs.anthropic.com/en/docs/initial-setup#python
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