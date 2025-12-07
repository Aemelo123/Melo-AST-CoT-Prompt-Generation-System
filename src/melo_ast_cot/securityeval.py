"""
https://github.com/s2e-lab/SecurityEval

"""

import random
from datasets import load_dataset


RANDOM_SEED = 999


def load_tasks(num_tasks: int = 50) -> list:
    dataset = load_dataset("s2e-lab/SecurityEval")
    all_tasks = list(dataset["train"])

    random.seed(RANDOM_SEED)
    selected_tasks = random.sample(all_tasks, num_tasks)

    return selected_tasks


def first_coding_example():
    dataset = load_dataset("s2e-lab/SecurityEval")
    security_eval1 = dataset["train"][0]
    first_prompt = security_eval1['Prompt']
    return first_prompt