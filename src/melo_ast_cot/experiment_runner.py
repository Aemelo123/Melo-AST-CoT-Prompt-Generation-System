import json
import random
from datetime import datetime
from pathlib import Path

from melo_ast_cot import ast_parser, nl_cot_baseline, securityeval, vulnerability_scanner


RANDOM_SEED = 999
RESULTS_DIR = Path("results")


def assign_conditions(tasks: list) -> tuple[list, list]:
    random.seed(RANDOM_SEED)
    shuffled = random.sample(tasks, len(tasks))
    return shuffled[:25], shuffled[25:]


def save_sample(sample: dict, iteration: int) -> Path:
    # Structure: results/iteration_X/MODEL/CONDITION/sample.json
    model = sample["model"]
    condition = sample["condition"]
    sample_dir = RESULTS_DIR / f"iteration_{iteration}" / model / condition
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / f"{sample['sample_id']}.json"
    path.write_text(json.dumps(sample, indent=2))
    return path


def generate_sample(task: dict, llm_func, iteration: int, condition: str, model_name: str) -> dict:
    task_id = task["ID"].replace(".py", "")
    prompt_code = task["Prompt"]
    parsed_prompt = ast_parser.parse_prompt(prompt_code)

    sample = {
        "sample_id": f"{task_id}_{model_name}_{condition}_{iteration}",
        "task_id": task_id,
        "model": model_name,
        "condition": condition,
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "original_prompt": prompt_code,
        "parsed_prompt": parsed_prompt,
    }

    try:
        if condition == "AST_COT":
            llm_json, generated_code, security_violations = ast_parser.get_llm_ast_json(parsed_prompt, llm_func)
            sample["llm_json_response"] = llm_json
            sample["generated_code"] = generated_code
            sample["security_violations"] = security_violations
        else:
            sample["generated_code"] = nl_cot_baseline.get_nl_cot_code(parsed_prompt, llm_func)
            sample["security_violations"] = []
        sample["success"] = True
        sample["error"] = None
    except Exception as e:
        sample["generated_code"] = None
        sample["success"] = False
        sample["error"] = str(e)

    return sample


def run_experiment(llm_func, model_name: str, iteration: int):
    tasks = securityeval.load_tasks()
    ast_tasks, nl_tasks = assign_conditions(tasks)
    ast_tasks, nl_tasks = assign_conditions(tasks)
    ast_tasks = ast_tasks[:10]  # only 10 samples each
    nl_tasks = nl_tasks[:10]

    for task in ast_tasks:
        print(f"[{model_name}] Generating AST_COT sample for {task['ID']} iteration {iteration}...")
        sample = generate_sample(task, llm_func, iteration, "AST_COT", model_name)
        if not sample["success"]:
            save_sample(sample, iteration)
            raise RuntimeError(f"AST_COT failed for {sample['sample_id']}: {sample['error']}")
        print(f"[{model_name}] Running Bandit and Semgrep on {sample['sample_id']}...")
        sample["scan_results"] = vulnerability_scanner.scan_code(sample["generated_code"])
        save_sample(sample, iteration)
        print(f"[{model_name}] Saved: {sample['sample_id']}")

    for task in nl_tasks:
        print(f"[{model_name}] Generating NL_COT sample for {task['ID']} iteration {iteration}...")
        sample = generate_sample(task, llm_func, iteration, "NL_COT", model_name)
        if not sample["success"]:
            save_sample(sample, iteration)
            raise RuntimeError(f"NL_COT failed for {sample['sample_id']}: {sample['error']}")
        print(f"[{model_name}] Running Bandit and Semgrep on {sample['sample_id']}...")
        sample["scan_results"] = vulnerability_scanner.scan_code(sample["generated_code"])
        save_sample(sample, iteration)
        print(f"[{model_name}] Saved: {sample['sample_id']}")
