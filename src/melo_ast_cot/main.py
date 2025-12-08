"""Main entry point for the experiment."""

import sys
from melo_ast_cot import llm_client, experiment_runner


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m melo_ast_cot.main <iteration>")
        print("Run one iteration every 6 hours (iterations 1-5)")
        sys.exit(1)

    iteration = int(sys.argv[1])
    print(f"Running iteration {iteration} of 5...")

    print("Starting experiment with GPT-4...")
    experiment_runner.run_experiment(llm_client.get_gpt_response, "GPT4", iteration)

    print("Starting experiment with Claude Sonnet 4.5...")
    experiment_runner.run_experiment(llm_client.get_anthropic_response, "CLAUDE", iteration)

    experiment_runner.export_results_to_csv(iteration)
    print(f"Iteration {iteration} complete!")




if __name__ == "__main__":
    main()
