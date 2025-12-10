"""
Based on SecurityEval's quickstart guide for loading the dataset of prompts from HuggingFace
(s2e-lab, "SecurityEval," GitHub, https://github.com/s2e-lab/SecurityEval#loading-the-dataset-of-prompts-from-huggingface)

"""

import random
from datasets import load_dataset


RANDOM_SEED = 999 # for reproducibility of task selection


def load_tasks(num_tasks: int = 50) -> list:
    # (Siddiq & Santos, 2022)
    dataset = load_dataset("s2e-lab/SecurityEval")
    all_tasks = list(dataset["train"])

    random.seed(RANDOM_SEED)
    selected_tasks = random.sample(all_tasks, num_tasks)

    return selected_tasks