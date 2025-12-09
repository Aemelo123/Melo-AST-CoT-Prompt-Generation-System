"""
Based on SecurityEval's quickstart guide for loading the dataset of prompts from HuggingFace
(s2e-lab, "SecurityEval," GitHub, https://github.com/s2e-lab/SecurityEval#loading-the-dataset-of-prompts-from-huggingface)

Additions:
- RANDOM_SEED for reproducibility of task selection
- load_tasks() selects a random subset of tasks for the experiment
- first_coding_example() retrieves a single prompt for testing
    - this is how I initially explored the dataset and developed the 
      AST-Guided CoT prompting strategy
"""

import random
from datasets import load_dataset


RANDOM_SEED = 999


def load_tasks(num_tasks: int = 50) -> list:
    # (Siddiq & Santos, 2022)
    dataset = load_dataset("s2e-lab/SecurityEval")
    all_tasks = list(dataset["train"])

    random.seed(RANDOM_SEED)
    selected_tasks = random.sample(all_tasks, num_tasks)

    return selected_tasks


def first_coding_example():
    dataset = load_dataset("s2e-lab/SecurityEval") # (Siddiq & Santos, 2022)
    security_eval1 = dataset["train"][0]
    first_prompt = security_eval1['Prompt']
    return first_prompt