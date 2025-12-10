"""
AST Parser with Security Validation

References:
    - Li, J., Li, G., Li, Y., & Jin, Z. (2025). Structured chain-of-thought prompting for
        code generation. ACM Transactions on Software Engineering and Methodology, 34(2),
        Article 37. https://doi.org/10.1145/3690635

    - Python Software Foundation. "ast — Abstract Syntax Trees." Python 3.x Documentation.
        https://docs.python.org/3/library/ast.html
"""

import ast
import json
from typing import Any, Callable, Dict, List

from ast2json import ast2json


def _get_node_type(node: Dict[str, Any]) -> str | None:
    return node.get("_type") or node.get("type")


class SecurityValidator:
    DANGEROUS_CALLS = {
        "eval", "exec", "compile", "__import__",
        "getattr", "setattr", "delattr",
        "globals", "locals", "vars",
    }

    DANGEROUS_MODULES = {
        "os": {"system", "popen", "spawn", "exec"},
        "subprocess": {"call", "run", "Popen"},
        "pickle": {"loads", "load"},
        "marshal": {"loads", "load"},
    }

    SQL_FUNCTIONS = {"execute", "executemany", "raw"}

    def __init__(self):
        self._security_rules: Dict[str, List[Callable]] = {}
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        self.add_rule("Call", self._check_dangerous_call)
        self.add_rule("Call", self._check_sql_injection)
        self.add_rule("Import", self._check_dangerous_import)
        self.add_rule("ImportFrom", self._check_dangerous_import)

    def add_rule(self, node_type: str, rule_func: Callable) -> None:
        if node_type not in self._security_rules:
            self._security_rules[node_type] = []
        self._security_rules[node_type].append(rule_func)


    def validate(self, node_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        violations = []
        stack = [node_json]

        while stack:
            node = stack.pop()

            if isinstance(node, dict):
                node_type = _get_node_type(node)
                if node_type in self._security_rules:
                    for rule in self._security_rules[node_type]:
                        if result := rule(node):
                            violations.append(result)

                for value in node.values():
                    if isinstance(value, (dict, list)):
                        stack.append(value)

            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, (dict, list)):
                        stack.append(item)

        return violations

    def _check_dangerous_call(self, node: Dict[str, Any]) -> Dict[str, Any] | None:
        func = node.get("func")
        func_name = None

        if isinstance(func, str):
            func_name = func
        elif isinstance(func, dict):
            func_type = _get_node_type(func)

            if func_type == "Name":
                func_name = func.get("id")
            elif func_type == "Attribute":
                module = func.get("value")
                attr = func.get("attr")
                module_name = module.get("id") if isinstance(module, dict) and _get_node_type(module) == "Name" else None

                if module_name in self.DANGEROUS_MODULES and attr in self.DANGEROUS_MODULES[module_name]:
                    return {
                        "rule": "dangerous_module_call",
                        "severity": "high",
                        "message": f"Dangerous call detected: {module_name}.{attr}",
                        "node": node,
                    }

        if func_name in self.DANGEROUS_CALLS:
            return {
                "rule": "dangerous_builtin",
                "severity": "high",
                "message": f"Dangerous builtin detected: {func_name}",
                "node": node,
            }

        return None

    def _check_sql_injection(self, node: Dict[str, Any]) -> Dict[str, Any] | None:
        func = node.get("func")
        if not isinstance(func, dict) or _get_node_type(func) != "Attribute":
            return None

        if func.get("attr") not in self.SQL_FUNCTIONS:
            return None

        for arg in node.get("args", []):
            if not isinstance(arg, dict):
                continue

            arg_type = _get_node_type(arg)
            if arg_type == "BinOp" and arg.get("op") in ("+", "%"):
                return {
                    "rule": "sql_injection",
                    "severity": "critical",
                    "message": "String concatenation detected in SQL query",
                    "node": node,
                }

            if arg_type == "JoinedStr":
                return {
                    "rule": "sql_injection",
                    "severity": "critical",
                    "message": "F-string detected in SQL query",
                    "node": node,
                }

        return None

    def _check_dangerous_import(self, node: Dict[str, Any]) -> Dict[str, Any] | None:
        node_type = _get_node_type(node)

        if node_type == "Import":
            for alias in node.get("names", []):
                name = alias.get("name") if isinstance(alias, dict) else alias
                if name in self.DANGEROUS_MODULES:
                    return {
                        "rule": "dangerous_import",
                        "severity": "medium",
                        "message": f"Dangerous module import: {name}",
                        "node": node,
                    }

        elif node_type == "ImportFrom":
            module = node.get("module")
            if module in self.DANGEROUS_MODULES:
                return {
                    "rule": "dangerous_import",
                    "severity": "medium",
                    "message": f"Dangerous import from module: {module}",
                    "node": node,
                }

        return None


_security_validator = SecurityValidator()


# formatting functions

def code_to_ast_string(code: str) -> str:
    tree = ast.parse(code)
    return ast.dump(tree, indent=2)

def code_to_ast_json(code: str) -> str:
    tree = ast.parse(code)
    return json.dumps({"ast": ast.dump(tree)})

def parse_prompt(code: str) -> dict:
    tree = ast.parse(code)
    return {"nodes": [ast2json(node) for node in tree.body]}


_STRING_FIELDS = {"name", "attr", "arg", "id", "module", "asname"}


def _convert_value(value: Any, field_name: str = "") -> Any:
    if isinstance(value, dict) and "type" in value:
        return json_to_ast(value)
    elif isinstance(value, str):
        if field_name in _STRING_FIELDS:
            return value
        return ast.Name(id=value, ctx=ast.Load())
    elif isinstance(value, list):
        return [_convert_value(v, field_name) for v in value]
    elif isinstance(value, (int, float, bool)) or value is None:
        return ast.Constant(value=value)
    else:
        return value


_LEGACY_TYPES = {"Str": "Constant", "Num": "Constant", "Bytes": "Constant"}


def json_to_ast(node_json: dict) -> ast.AST:
    node_type_name = node_json.get("type")
    if node_type_name is None:
        raise ValueError(f"Node missing 'type' field: {node_json}")

    if node_type_name in _LEGACY_TYPES:
        value = node_json.get("s") or node_json.get("n") or node_json.get("value")
        return ast.Constant(value=value)

    node_class = getattr(ast, node_type_name, None)

    if node_class is None:
        raise ValueError(f"Unknown AST type: {node_type_name}")

    fields = getattr(node_class, "_fields", ())
    if not fields:
        return node_class()

    kwargs = {}
    for field in fields:
        if field in node_json:
            value = node_json[field]
            kwargs[field] = _convert_value(value, field)

    # Handle special cases for LLM output format

    # Assign: LLM uses "target" (string), AST needs "targets" (list of nodes)
    if node_type_name == "Assign" and "target" in node_json and "targets" not in kwargs:
        target = node_json["target"]
        if isinstance(target, str):
            kwargs["targets"] = [ast.Name(id=target, ctx=ast.Store())]
        else:
            kwargs["targets"] = [_convert_value(target)]
            kwargs["targets"][0].ctx = ast.Store()

    # Attribute: needs ctx
    if node_type_name == "Attribute" and "ctx" not in kwargs:
        kwargs["ctx"] = ast.Load()

    # Name: needs ctx
    if node_type_name == "Name" and "ctx" not in kwargs:
        kwargs["ctx"] = ast.Load()

    # Call: needs empty keywords if not provided
    if node_type_name == "Call" and "keywords" not in kwargs:
        kwargs["keywords"] = []

    return node_class(**kwargs)


def json_to_code(node_json) -> str:
    # handle array of nodes (multiple statements)
    if isinstance(node_json, list):
        nodes = [json_to_ast(n) for n in node_json]
        for n in nodes:
            ast.fix_missing_locations(n)
        return "\n".join(ast.unparse(n) for n in nodes)
    # handle {"body": [...]} wrapper without type
    if isinstance(node_json, dict) and "body" in node_json and "type" not in node_json:
        nodes = [json_to_ast(n) for n in node_json["body"]]
        for n in nodes:
            ast.fix_missing_locations(n)
        return "\n".join(ast.unparse(n) for n in nodes)
    node = json_to_ast(node_json)
    ast.fix_missing_locations(node)
    return ast.unparse(node)


def _clean_json_response(response: str) -> str:
    """Strip markdown and fix Python literals in JSON response."""
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    # Find the first { or [ if response starts with text
    if response and response[0] not in "{[":
        brace = response.find("{")
        bracket = response.find("[")
        if brace == -1:
            start = bracket
        elif bracket == -1:
            start = brace
        else:
            start = min(brace, bracket)
        if start != -1:
            response = response[start:]
    response = response.replace(": True", ": true")
    response = response.replace(": False", ": false")
    response = response.replace(": None", ": null")
    return response


# Helper to handle both dynamic and legacy AST argument formats
def _extract_args(args_node: dict) -> List[str]:
    """Extract argument names from an arguments node (dynamic format)."""
    if isinstance(args_node, dict) and "args" in args_node:
        # New dynamic format: args is a list of arg nodes
        return [arg.get("arg", "") for arg in args_node.get("args", []) if isinstance(arg, dict)]
    elif isinstance(args_node, list):
        # Legacy format: args is already a list of strings
        return [a if isinstance(a, str) else a.get("arg", "") for a in args_node]
    return []


def _extract_imports(node: dict) -> List[str]:
    node_type = _get_node_type(node)
    if node_type == "Import":
        names = node.get("names", [])
        if names and isinstance(names[0], dict):
            return [alias.get("name", "") for alias in names]
        return node.get("modules", [])
    elif node_type == "ImportFrom":
        module = node.get("module", "")
        return [module] if module else []
    return []


def get_llm_ast_json(parsed_prompt: dict, llm_func) -> tuple[dict, str, list]:
    func_info = None
    imports = []

    for node in parsed_prompt["nodes"]:
        node_type = _get_node_type(node)
        if node_type == "FunctionDef":
            func_info = node
        elif node_type in ("Import", "ImportFrom"):
            imports.extend(_extract_imports(node))

    if not func_info:
        raise ValueError("No function found in parsed prompt")

    # Extract args - handle both old format (list of strings) and new format (arguments node)
    args = func_info.get("args", [])
    if isinstance(args, dict):
        args = _extract_args(args)

    docstring = func_info.get("docstring")
    if docstring is None and "body" in func_info:
        body = func_info.get("body", [])
        if body and isinstance(body[0], dict) and _get_node_type(body[0]) == "Expr":
            expr_value = body[0].get("value", {})
            if isinstance(expr_value, dict) and _get_node_type(expr_value) == "Constant":
                docstring = expr_value.get("value")

    # Prompt going to stay static, because if we provide the dynamic list, it's going to get 
    # too complicated and i could see it confusing the LLM more than helping it.
    prompt = f"""You are generating Python code as AST nodes in JSON format.
    
Function to implement:
- Name: {func_info["name"]}
- Arguments: {args}
- Description: {docstring}
- Available imports: {imports}

Generate the function body as a JSON object representing AST nodes.
Use these node types:
- With: {{"type": "With", "context_expr": <Call node>, "var": "variable_name", "body": [<nodes>]}}
- Call: {{"type": "Call", "func": "function_name" or {{"type": "Attribute", "value": "module", "attr": "function"}}, "args": [<args>]}}
- Assign: {{"type": "Assign", "target": "variable_name", "value": <node>}}
- Return: {{"type": "Return", "value": "variable_name" or <node>}}
- Constant: {{"type": "Constant", "value": "literal_value"}}
- If: {{"type": "If", "test": <Compare or value>, "body": [<nodes>], "orelse": [<nodes>]}}
- For: {{"type": "For", "target": "variable_name", "iter": <node>, "body": [<nodes>], "orelse": [<nodes>]}}
- While: {{"type": "While", "test": <Compare or value>, "body": [<nodes>], "orelse": [<nodes>]}}
- Try: {{"type": "Try", "body": [<nodes>], "handlers": [{{"type": "exception_type", "name": "e", "body": [<nodes>]}}], "orelse": [<nodes>], "finalbody": [<nodes>]}}
- BinOp: {{"type": "BinOp", "left": <node>, "op": "+|-|*|/|//|%|**", "right": <node>}}
- Compare: {{"type": "Compare", "left": <node>, "comparators": [{{"op": "==|!=|<|<=|>|>=|in|not in", "value": <node>}}]}}

Return ONLY the JSON object, no explanation."""

    response = llm_func(prompt)

    print("---AST CoT Raw Response---")
    print(response)
    print("---End Raw Response---")

    response = _clean_json_response(response)

    # try to parse JSON and convert to AST with retries
    max_retries = 3
    last_error = None
    response_json = None

    for attempt in range(max_retries):
        # Step 1: Try to parse JSON
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries - 1:
                fix_prompt = f"""The following JSON is invalid. Fix it to be valid JSON and return ONLY the corrected JSON, nothing else.

Error: {e}

Invalid JSON:
{response}"""
                response = _clean_json_response(llm_func(fix_prompt))
                print(f"---JSON Fix Attempt {attempt + 1}---")
                print(response[:500])
                print("---End Fix Attempt---")
                continue
            raise last_error

        # Step 2: Validate against security rules
        violations = _security_validator.validate(response_json)
        if violations:
            print(f"---Security Violations Found: {len(violations)}---")
            for v in violations:
                print(f"  [{v['severity']}] {v['message']}")

        # Step 3: Ask LLM to convert its AST JSON to Python code
        code_prompt = f"""Convert this AST JSON to Python code. Return ONLY the code, no explanation.

{json.dumps(response_json, indent=2)}"""
        code = llm_func(code_prompt)
        code = code.strip()
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        code = code.strip()

        print("---Generated Code---")
        print(code)
        print("---End Generated Code---")

        return response_json, code, violations

    raise last_error