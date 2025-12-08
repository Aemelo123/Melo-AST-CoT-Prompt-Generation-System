import csv
import json
import random
from datetime import datetime
from pathlib import Path

from melo_ast_cot import ast_parser, nl_cot_baseline, securityeval, vulnerability_scanner


RANDOM_SEED = 999
RESULTS_DIR = Path("results")




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
            generated_code, security_violations = nl_cot_baseline.get_nl_cot_code(parsed_prompt, llm_func)
            sample["generated_code"] = generated_code
            sample["security_violations"] = security_violations
        sample["success"] = True
        sample["error"] = None
    except Exception as e:
        sample["generated_code"] = None
        sample["success"] = False
        sample["error"] = str(e)

    return sample


def select_tasks(tasks: list, n: int = 50) -> list:
    """Select n random tasks using fixed seed for reproducibility."""
    random.seed(RANDOM_SEED)
    return random.sample(tasks, n)


def run_experiment(llm_func, model_name: str, iteration: int):
    tasks = securityeval.load_tasks()
    selected_tasks = select_tasks(tasks, 50)

    # Run both conditions on the same 50 tasks (within-subjects design)
    for task in selected_tasks:
        # AST_COT condition
        print(f"[{model_name}] Generating AST_COT sample for {task['ID']} iteration {iteration}...")
        sample = generate_sample(task, llm_func, iteration, "AST_COT", model_name)
        if not sample["success"]:
            save_sample(sample, iteration)
            raise RuntimeError(f"AST_COT failed for {sample['sample_id']}: {sample['error']}")
        print(f"[{model_name}] Running Bandit and Semgrep on {sample['sample_id']}...")
        sample["scan_results"] = vulnerability_scanner.scan_code(sample["generated_code"])
        save_sample(sample, iteration)
        print(f"[{model_name}] Saved: {sample['sample_id']}")

        # NL_COT condition (same task)
        print(f"[{model_name}] Generating NL_COT sample for {task['ID']} iteration {iteration}...")
        sample = generate_sample(task, llm_func, iteration, "NL_COT", model_name)
        if not sample["success"]:
            save_sample(sample, iteration)
            raise RuntimeError(f"NL_COT failed for {sample['sample_id']}: {sample['error']}")
        print(f"[{model_name}] Running Bandit and Semgrep on {sample['sample_id']}...")
        sample["scan_results"] = vulnerability_scanner.scan_code(sample["generated_code"])
        save_sample(sample, iteration)
        print(f"[{model_name}] Saved: {sample['sample_id']}")


def export_results_to_csv(iteration: int) -> Path:
    iteration_dir = RESULTS_DIR / f"iteration_{iteration}"
    csv_dir = RESULTS_DIR / "csv_exports"
    csv_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for f in sorted(iteration_dir.rglob("*.json")):
        data = json.loads(f.read_text())
        scan = data.get("scan_results", {})

        rows.append({
            "sample_id": data.get("sample_id", ""),
            "task_id": data.get("task_id", ""),
            "model": data.get("model", ""),
            "condition": data.get("condition", ""),
            "iteration": data.get("iteration", ""),
            "success": data.get("success", False),
            "num_vulnerabilities": scan.get("num_vulnerabilities", 0),
            "loc": scan.get("loc", 0),
            "vulnerability_density": scan.get("vulnerability_density", 0),
            "security_violations_count": len(data.get("security_violations", [])),
            "findings": json.dumps(scan.get("findings", [])),
            "timestamp": data.get("timestamp", ""),
        })

    csv_path = csv_dir / f"experiment_results_iteration_{iteration}.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Exported {len(rows)} samples to {csv_path}")
    return csv_path
