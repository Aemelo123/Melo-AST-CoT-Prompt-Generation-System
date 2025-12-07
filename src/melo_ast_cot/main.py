"""Main entry point for the experiment."""

import json
from melo_ast_cot import llm_client, securityeval, ast_parser, nl_cot_baseline

def main() -> None:
    """Run the experiment."""

    '''--- commenting out LLM calls for faster testing and not to waste money ---'''
    # print("Melo AST CoT Guided Prompting Experiment")

    # print("---GPT Response---")
    # gpt_response = llm_client.get_gpt_response(
    #     prompt="Write a one-sentence bedtime story about a unicorn."
    # )
    # print(gpt_response)

    # print("---Anthropic Response---")
    # anthropic_response = llm_client.get_anthropic_response(
    #     prompt="Write a one-sentence bedtime story about a donkey."
    # )
    # print(anthropic_response)

    print("---First SecurityEval Coding Example---")
    print(securityeval.first_coding_example())

    print("---Parsed Prompt---")
    parsed_prompt = ast_parser.parse_prompt(securityeval.first_coding_example())
    print(parsed_prompt)

    print("---JSON to Code Test---")
    json_schema = {
        'type': 'With',
        'context_expr': {
            'type': 'Call',
            'func': 'open',
            'args': ['filename', 'r']
        },
        'var': 'f',
        'body': [
            {
                'type': 'Assign',
                'target': 'data',
                'value': {
                    'type': 'Call',
                    'func': {'type': 'Attribute', 'value': 'yaml', 'attr': 'safe_load'},
                    'args': ['f']
                }
            },
            {
                'type': 'Return',
                'value': 'data'
            }
        ]
    }
    generated_code = ast_parser.json_to_code(json_schema)
    print(generated_code)

    print("---Prompt LLM for AST JSON and Generate Code---")
    parsed_prompt = ast_parser.parse_prompt(securityeval.first_coding_example())
    llm_json = ast_parser.get_llm_ast_json(parsed_prompt, llm_client.get_anthropic_response)

    print("---LLM JSON---")
    print(json.dumps(llm_json, indent=2))

    print("---Generated Code from LLM JSON---")
    generated_code = ast_parser.json_to_code(llm_json)
    print(generated_code)

    print("Natural Language Chain-of-Thought Baseline Code Generation---")
    nl_cot_code = nl_cot_baseline.get_nl_cot_code(
        parsed_prompt, llm_client.get_anthropic_response
    )
    print(nl_cot_code)

    


if __name__ == "__main__":
    main()
