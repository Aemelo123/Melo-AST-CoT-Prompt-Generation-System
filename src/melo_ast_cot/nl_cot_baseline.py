# Zero-shot Chain-of-Thought baseline
# Based on: Kojima et al. (2022) "Large Language Models are Zero-Shot Reasoners"
# arXiv: https://arxiv.org/abs/2205.11916


def get_nl_cot_code(parsed_prompt: dict, llm_func) -> str:
    func_info = None
    imports = []

    for node in parsed_prompt["nodes"]:
        if node["type"] == "FunctionDef":
            func_info = node
        elif node["type"] == "Import":
            imports.extend(node["modules"])
        elif node["type"] == "ImportFrom":
            imports.append(node["module"])

    if not func_info:
        raise ValueError("No function found in parsed prompt")

    prompt = f"""Implement this function:
- Name: {func_info["name"]}
- Arguments: {func_info["args"]}
- Description: {func_info["docstring"]}
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
