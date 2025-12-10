"""
Dynamic AST Parser with Registry Pattern and Security Validation

DynamicASTVisitor:
    With Fowler's (2002) Registry Pattern, the AST Parser can call on the correct handler 
    in O(1) time. Instead of having to go through approximately 70 conditionals to see 
    which class of AST Node was matched, we register each AST Node type's handler by class 
    in a Hash Table so when we traverse the AST Tree, we can find it in constant time. Using 
    this pattern makes the design both scalable and extensible. If a developer wants to create 
    a new Handler for an AST Node Type, they only have to add it to the Registry and do not 
    have to alter their existing code.

References:
    - Li, J., Li, G., Li, Y., & Jin, Z. (2025). Structured chain-of-thought prompting for
        code generation. ACM Transactions on Software Engineering and Methodology, 34(2),
        Article 37. https://doi.org/10.1145/3690635
        (Where the idea stems from for the abstract syntax tree)

    - Gamma et al., "Design Patterns" (1994) - Visitor Pattern
    - Martin Fowler, Patterns of Enterprise Application Architecture (Addison-Wesley, 2002),
        pp. 480-485.
    - Python Software Foundation. "ast — Abstract Syntax Trees." Python 3.x Documentation.
        https://docs.python.org/3/library/ast.html
    - GeeksforGeeks. "DFS Traversal of a Tree Using Recursion."
        https://www.geeksforgeeks.org/dsa/dfs-traversal-of-a-tree-using-recursion/
"""

import ast
import json
from typing import Any, Callable, Dict, List, Type


# DYNAMIC AST VISITOR: Auto-discovery + O(1) Registry Dispatch
class DynamicASTVisitor:
    def __init__(self):
        self._handlers: Dict[Type[ast.AST], Callable[[ast.AST], Dict[str, Any]]] = {}
        self._visit_count = 0
        self._max_depth = 0
        self._type_counts: Dict[str, int] = {}
        self._register_builtin_handlers()


    def _register_builtin_handlers(self) -> None:
        """Register simple auto-generated handlers for all known AST types."""
        for name in dir(ast):
            obj = getattr(ast, name)
            if isinstance(obj, type) and issubclass(obj, ast.AST) and obj is not ast.AST:
                self._handlers[obj] = self._generate_handler(obj)

    def _generate_handler(self, node_type: Type[ast.AST]) -> Callable[[ast.AST], Dict[str, Any]]:
        fields = getattr(node_type, "_fields", ())

        def handler(node: ast.AST) -> Dict[str, Any]:
            out: Dict[str, Any] = {"type": node_type.__name__}
            for field in fields:
                out[field] = self._process_value(getattr(node, field, None))

            # include source location if available
            if hasattr(node, "lineno"):
                out["lineno"] = node.lineno
            if hasattr(node, "col_offset"):
                out["col_offset"] = node.col_offset

            return out

        return handler


    def _process_value(self, value: Any) -> Any:
        if isinstance(value, ast.AST):
            return self.visit(value)
        if isinstance(value, list):
            return [self._process_value(item) for item in value]
        return value


    def visit(self, node: ast.AST) -> Dict[str, Any]:
        self._visit_count += 1
        node_type = type(node)
        self._type_counts[node_type.__name__] = self._type_counts.get(node_type.__name__, 0) + 1

        handler = self._handlers.get(node_type)
        if handler is None:
            handler = self._generate_handler(node_type)
            self._handlers[node_type] = handler

        return handler(node)


    def register_handler(self, node_type: Type[ast.AST], handler: Callable[[ast.AST], Dict[str, Any]]) -> None:
        """Register a custom handler that overrides the built-in handler for a node type."""
        self._handlers[node_type] = handler


    def parse_tree(self, code: str) -> Dict[str, Any]:
        self._visit_count = 0
        self._max_depth = 0
        self._type_counts = {}

        tree = ast.parse(code)
        results = [self._walk(node, 0) for node in tree.body]

        return {
            "nodes": results,
            "metadata": {
                "total_nodes_visited": self._visit_count,
                "max_depth": self._max_depth,
                "top_level_count": len(results),
                "type_frequency": self._type_counts,
                "registered_handlers": len(self._handlers),
            },
        }

    def _walk(self, node: ast.AST, depth: int) -> Dict[str, Any]:
        self._max_depth = max(self._max_depth, depth)
        data = self.visit(node)
        data["depth"] = depth
        return data


    def get_metrics(self) -> Dict[str, Any]:
        return {
            "visit_count": self._visit_count,
            "max_depth": self._max_depth,
            "type_counts": dict(self._type_counts),
            "registered_handlers": len(self._handlers),
        }

    def get_registered_types(self) -> List[str]:
        return sorted(t.__name__ for t in self._handlers)




# Part of the **novel** approach: Intermediate Security Validation
#
# NL_COT (Natural Language Chain-of-Thought) pipeline:
#   Prompt -> LLM reasoning -> Raw code -> Bandit/Semgrep (post-hoc validation)
#
# AST_COT (AST-Guided Chain-of-Thought) pipeline:
#   Prompt -> LLM produces AST JSON -> SecurityVisitor (pre-synthesis) -> Code synthesis -> Bandit/Semgrep
#                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                      This step is only possible because we have
#                                      structured output before code generation
#
# Foundational References:
#   - Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language
#     Models," NeurIPS 2022. https://arxiv.org/abs/2201.11903
#     (Intermediate reasoning steps improve LLM performance)
#
#   - Yamaguchi et al., "Generalized Vulnerability Extrapolation using Abstract
#     Syntax Trees," ACSAC 2012. https://dl.acm.org/doi/10.1145/2420950.2421003
#     (AST-based security analysis for vulnerability detection)
#
#   - Gamma et al., "Design Patterns" (1994) - Visitor Pattern
#     (Structural pattern for AST traversal)
#
#   - Fowler, "Patterns of Enterprise Application Architecture" (2002), pp. 480-485
#     (Registry Pattern for O(1) handler dispatch)

class SecurityVisitor(DynamicASTVisitor):
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

    def __init__(self, debug: bool = False):
        super().__init__()
        self._security_rules: Dict[str, List[Callable]] = {}
        self._violations: List[Dict[str, Any]] = []
        self.debug = debug

        self._register_default_rules()

    def _log(self, *msg):
        if self.debug:
            print("[SecurityVisitor]", *msg)

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
        self._violations = []
        self._log("Starting validation")

        stack = [node_json]

        while stack:
            node = stack.pop()

            if isinstance(node, dict):
                node_type = node.get("type")
                self._log("Visiting:", node_type)

                # apply rules for this node type
                if node_type in self._security_rules:
                    for rule in self._security_rules[node_type]:
                        result = rule(node)
                        if result:
                            self._log("  Violation:", result["rule"])
                            self._violations.append(result)

                # push child structures for later
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        stack.append(value)

            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, (dict, list)):
                        stack.append(item)

        self._log("Validation complete. Found", len(self._violations), "issues.")
        return self._violations

    def _check_dangerous_call(self, node: Dict[str, Any]) -> Dict[str, Any] | None:
        self._log("Checking dangerous call")
        func = node.get("func")
        func_name = None

        # attempt straightforward extraction
        if isinstance(func, str):
            func_name = func
        elif isinstance(func, dict):
            node_type = func.get("type")

            if node_type == "Name":
                func_name = func.get("id")

            elif node_type == "Attribute":
                module = func.get("value")
                attr = func.get("attr")

                module_name = None
                if isinstance(module, dict) and module.get("type") == "Name":
                    module_name = module.get("id")

                # check module-based dangerous calls
                if module_name in self.DANGEROUS_MODULES:
                    if attr in self.DANGEROUS_MODULES[module_name]:
                        return {
                            "rule": "dangerous_module_call",
                            "severity": "high",
                            "message": f"Dangerous call detected: {module_name}.{attr}",
                            "node": node,
                        }

        # check builtin dangerous calls
        if func_name in self.DANGEROUS_CALLS:
            return {
                "rule": "dangerous_builtin",
                "severity": "high",
                "message": f"Dangerous builtin detected: {func_name}",
                "node": node,
            }

        return None

    def _check_sql_injection(self, node: Dict[str, Any]) -> Dict[str, Any] | None:
        self._log("Checking SQL injection")

        func = node.get("func")
        if not isinstance(func, dict):
            return None

        if func.get("type") != "Attribute":
            return None

        attr = func.get("attr")
        if attr not in self.SQL_FUNCTIONS:
            return None

        args = node.get("args", [])

        for arg in args:
            if not isinstance(arg, dict):
                continue

            # concatenation-based queries
            if arg.get("type") == "BinOp" and arg.get("op") in ("+", "%"):
                return {
                    "rule": "sql_injection",
                    "severity": "critical",
                    "message": "String concatenation detected in SQL query",
                    "node": node,
                }

            # f-string SQL
            if arg.get("type") == "JoinedStr":
                return {
                    "rule": "sql_injection",
                    "severity": "critical",
                    "message": "F-string detected in SQL query",
                    "node": node,
                }

        return None

    def _check_dangerous_import(self, node: Dict[str, Any]) -> Dict[str, Any] | None:
        self._log("Checking dangerous import")

        if node.get("type") == "Import":
            for alias in node.get("names", []):
                name = alias.get("name") if isinstance(alias, dict) else alias
                if name in self.DANGEROUS_MODULES:
                    return {
                        "rule": "dangerous_import",
                        "severity": "medium",
                        "message": f"Dangerous module import: {name}",
                        "node": node,
                    }

        elif node.get("type") == "ImportFrom":
            module = node.get("module")
            if module in self.DANGEROUS_MODULES:
                return {
                    "rule": "dangerous_import",
                    "severity": "medium",
                    "message": f"Dangerous import from module: {module}",
                    "node": node,
                }

        return None


# GLOBAL VISITOR INSTANCE

_security_visitor = SecurityVisitor()


# PUBLIC API FUNCTIONS

def code_to_ast_string(code: str) -> str:
    """Convert code to AST dump string (legacy function).

    References:
        - Python Software Foundation. "ast — Abstract Syntax Trees." Python 3.x Documentation.
          https://docs.python.org/3/library/ast.html
    """
    tree = ast.parse(code)
    return ast.dump(tree, indent=2)



# By representing the Abstract Syntax Tree (AST) as JSON, the large language model will
# be thinking about the AST nodes (FunctionDef, Assign, Call, etc), rather than just raw
# text; this will allow us to implement AST-Guided CoT.

# The below functions are all in regards to formatting with JSON

def code_to_ast_json(code: str) -> str:
    tree = ast.parse(code)
    return json.dumps({"ast": ast.dump(tree)})

def parse_prompt(code: str) -> dict:
    visitor = DynamicASTVisitor()
    tree = ast.parse(code)
    nodes = [visitor.visit(node) for node in tree.body]
    return {"nodes": nodes}


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


# Helper to handle both dynamic and legacy AST import formats
def _extract_imports(node: dict) -> List[str]:
    """Extract module names from Import/ImportFrom nodes (dynamic format)."""
    if node["type"] == "Import":
        # New format: names is a list of alias nodes
        names = node.get("names", [])
        if names and isinstance(names[0], dict):
            return [alias.get("name", "") for alias in names]
        # Legacy format: modules is already a list of strings
        return node.get("modules", [])
    elif node["type"] == "ImportFrom":
        module = node.get("module", "")
        return [module] if module else []
    return []


# Core function of the AST-Guided CoT pipeline - orchestrates the novel approach
def get_llm_ast_json(parsed_prompt: dict, llm_func) -> tuple[dict, str, list]:
    func_info = None
    imports = []

    for node in parsed_prompt["nodes"]:
        if node["type"] == "FunctionDef":
            func_info = node
        elif node["type"] in ("Import", "ImportFrom"):
            imports.extend(_extract_imports(node))

    if not func_info:
        raise ValueError("No function found in parsed prompt")

    # Extract args - handle both old format (list of strings) and new format (arguments node)
    args = func_info.get("args", [])
    if isinstance(args, dict):
        args = _extract_args(args)

    # Extract docstring - handle both formats
    docstring = func_info.get("docstring")
    if docstring is None and "body" in func_info:
        # Try to get docstring from body in new format
        body = func_info.get("body", [])
        if body and isinstance(body[0], dict) and body[0].get("type") == "Expr":
            expr_value = body[0].get("value", {})
            if isinstance(expr_value, dict) and expr_value.get("type") == "Constant":
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
        violations = _security_visitor.validate(response_json)
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