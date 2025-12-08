# Zero-shot Chain-of-Thought baseline
# Based on: Kojima et al. (2022) "Large Language Models are Zero-Shot Reasoners"
# arXiv: https://arxiv.org/abs/2205.11916

from melo_ast_cot.ast_parser import _extract_imports, _extract_args, parse_prompt as parse_code, SecurityVisitor


def _extract_code(response: str) -> str:
    code = response
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    return code.strip()


def get_nl_cot_code(parsed_prompt: dict, llm_func) -> tuple[str, list]:
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

    code = _extract_code(response)

    security_visitor = SecurityVisitor()
    violations = []

    max_retries = 2
    for attempt in range(max_retries):
        try:
            parsed_code = parse_code(code)

            for node in parsed_code.get("nodes", []):
                node_violations = security_visitor.validate(node)
                violations.extend(node_violations)

            if violations and attempt < max_retries - 1:
                violation_msgs = [f"- {v['message']}" for v in violations]
                fix_prompt = f"""The following code has security vulnerabilities:

```python
{code}
```

Security issues found:
{chr(10).join(violation_msgs)}

Please rewrite the function to avoid these security issues. Use safe alternatives:
- Instead of eval/exec, use ast.literal_eval for data parsing
- Instead of os.system/subprocess, use safer APIs or validate inputs
- Instead of string concatenation in SQL, use parameterized queries

Return ONLY the corrected code."""

                print(f"---Security Fix Attempt {attempt + 1}---")
                fix_response = llm_func(fix_prompt)
                code = _extract_code(fix_response)
                print(code)
                print("---End Fix Attempt---")
                violations = []
                continue

            break

        except SyntaxError as e:
            print(f"---Syntax Error in generated code: {e}---")
            break

    if violations:
        print(f"---Security Violations Found: {len(violations)}---")
        for v in violations:
            print(f"  [{v['severity']}] {v['message']}")

    return code, violations
