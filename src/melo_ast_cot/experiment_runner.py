import random
from datetime import datetime

from melo_ast_cot import ast_parser, nl_cot_baseline, securityeval


RANDOM_SEED = 999


def assign_conditions(tasks: list) -> tuple[list, list]:
    random.seed(RANDOM_SEED)
    shuffled = random.sample(tasks, len(tasks))
    return shuffled[:25], shuffled[25:]


def generate_sample(task: dict, llm_func, iteration: int, condition: str) -> dict:
    task_id = task["ID"].replace(".py", "")
    prompt_code = task["Prompt"]
    parsed_prompt = ast_parser.parse_prompt(prompt_code)

    sample = {
        "sample_id": f"{task_id}_{condition}_{iteration}",
        "task_id": task_id,
        "condition": condition,
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "original_prompt": prompt_code,
        "parsed_prompt": parsed_prompt,
    }

    try:
        if condition == "AST_COT":
            llm_json = ast_parser.get_llm_ast_json(parsed_prompt, llm_func)
            sample["llm_json_response"] = llm_json
            sample["generated_code"] = ast_parser.json_to_code(llm_json)
        else:
            sample["generated_code"] = nl_cot_baseline.get_nl_cot_code(parsed_prompt, llm_func)
        sample["success"] = True
        sample["error"] = None
    except Exception as e:
        sample["generated_code"] = None
        sample["success"] = False
        sample["error"] = str(e)

    return sample
