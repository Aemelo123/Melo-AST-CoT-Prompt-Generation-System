# Zero-shot Chain-of-Thought baseline
# Based on: Kojima et al. (2022) "Large Language Models are Zero-Shot Reasoners"
# arXiv: https://arxiv.org/abs/2205.11916

from melo_ast_cot.ast_parser import _extract_imports, _extract_args


def get_nl_cot_code(parsed_prompt: dict, llm_func) -> str:
    func_info = None
    imports = []

    for node in parsed_prompt["nodes"]:
        if node["type"] == "FunctionDef":
            func_info = node
        elif node["type"] in ("Import", "ImportFrom"):
            imports.extend(_extract_imports(node))

    if not func_info:
        raise ValueError("No function found in parsed prompt")

    args = func_info.get("args", [])
    if isinstance(args, dict):
        args = _extract_args(args)

    docstring = func_info.get("docstring")
    if docstring is None and "body" in func_info:
        body = func_info.get("body", [])
        if body and isinstance(body[0], dict) and body[0].get("type") == "Expr":
            expr_value = body[0].get("value", {})
            if isinstance(expr_value, dict) and expr_value.get("type") == "Constant":
                docstring = expr_value.get("value")

    prompt = f"""Implement this function:
- Name: {func_info["name"]}
- Arguments: {args}
- Description: {docstring}
- Available imports: {imports}

Let's think step by step."""

    response = llm_func(prompt)
    
    print("---NL CoT Raw Response---")
    print(response)

    # extract code from response
    code = response
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    return code.strip()
