"""Main entry point for the experiment."""

from melo_ast_cot import llm_client, securityeval

def main() -> None:
    """Run the experiment."""
    print("Melo AST CoT Guided Prompting Experiment")

    print("---GPT Response---")
    gpt_response = llm_client.get_gpt_response(
        prompt="Write a one-sentence bedtime story about a unicorn."
    )
    print(gpt_response)

    print("---Anthropic Response---")
    anthropic_response = llm_client.get_anthropic_response(
        prompt="Write a one-sentence bedtime story about a donkey."
    )
    print(anthropic_response)

    print("---First SecurityEval Coding Example---")
    print(securityeval.first_coding_example())


if __name__ == "__main__":
    main()
