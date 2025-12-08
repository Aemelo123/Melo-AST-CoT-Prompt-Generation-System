"""
Dynamic AST Parser with Auto-Discovery Registry Pattern

This module implements a fully dynamic AST parser that automatically discovers
and registers handlers for ALL Python AST node types using introspection.
No manual handler definitions required - scales automatically with Python versions.

- Auto-discovery: Introspects ast module to find all node types
- Auto-generation: Builds handlers from node._fields automatically
- O(1) dispatch: Registry pattern for constant-time handler lookup
- Recursive traversal: Full tree depth, not just top-level nodes

References:
- Gamma et al., "Design Patterns" (1994) - Visitor Pattern
- Tarjan, "Amortized Computational Complexity" (1985)

https://docs.python.org/3/library/ast.html
"""

import ast
import json
from typing import Any, Callable, Dict, List, Type


# DYNAMIC AST VISITOR: Auto-discovery + O(1) Registry Dispatch


class DynamicASTVisitor:
    """
    dynamic AST visitor pattern that auto-discovers all node types.

    auto-generates handlers based on each node's _fields attribute.

    Complexity:
    - Initialization: O(k) where k = number of AST node types (~70)
    - Handler lookup: O(1) average case (hash table)
    - Tree traversal: O(n) where n = total nodes in tree
    - Space: O(k) for registry + O(d) stack depth for recursion

    Usage:
        visitor = DynamicASTVisitor()
        result = visitor.parse_tree("def foo(): pass")
    """

    def __init__(self):
        """Initialize visitor and auto-register all AST node types."""
        self._handlers: Dict[Type[ast.AST], Callable[[ast.AST], Dict[str, Any]]] = {}
        self._custom_handlers: Dict[Type[ast.AST], Callable[[ast.AST], Dict[str, Any]]] = {}

        # Metrics for analysis
        self._visit_count = 0
        self._max_depth = 0
        self._type_counts: Dict[str, int] = {}

        # Auto-discover and register all AST node types
        self._auto_register_all()

    def _auto_register_all(self) -> None:
        """
        Discover and register handlers for ALL AST node types.
        """
        for name in dir(ast):
            obj = getattr(ast, name)
            # Check if it's a class that inherits from ast.AST
            if isinstance(obj, type) and issubclass(obj, ast.AST) and obj is not ast.AST:
                self._handlers[obj] = self._auto_generate_handler(obj)

    def _auto_generate_handler(
        self,
        node_type: Type[ast.AST]
    ) -> Callable[[ast.AST], Dict[str, Any]]:
        """
        Build a handler dynamically from the node's _fields.

        Every AST node class has a _fields attribute that defines its
        structure. For example:
            ast.Assign._fields = ('targets', 'value')
            ast.If._fields = ('test', 'body', 'orelse')

        use this & auto-generate extraction logic without
        writing handlers manually.
        """
        fields = getattr(node_type, '_fields', ())

        def handler(node: ast.AST) -> Dict[str, Any]:
            result = {"type": node_type.__name__}

            for field in fields:
                value = getattr(node, field, None)
                result[field] = self._process_value(value)

            # Add source info if available (line numbers, etc.)
            if hasattr(node, 'lineno'):
                result["lineno"] = node.lineno
            if hasattr(node, 'col_offset'):
                result["col_offset"] = node.col_offset

            return result

        return handler

    def _process_value(self, value: Any) -> Any:
        """
        Recursively process any value from an AST node field.

        Handles three cases:
        1. AST nodes -> recurse with visit()
        2. Lists -> process each element
        3. Primitives (str, int, None) -> return as-is

        This enables full recursive traversal of the entire tree.
        """
        if isinstance(value, ast.AST):
            return self.visit(value)
        elif isinstance(value, list):
            return [self._process_value(v) for v in value]
        else:
            # Primitives: str, int, float, None, bool
            return value

    def visit(self, node: ast.AST) -> Dict[str, Any]:
        """
        O(1) dispatch to the appropriate handler.

        If the node type isn't registered (e.g., new Python version),
        auto-generates and registers a handler on the fly.
        """
        self._visit_count += 1
        node_type = type(node)

        # Track type frequency for analysis
        type_name = node_type.__name__
        self._type_counts[type_name] = self._type_counts.get(type_name, 0) + 1

        # Check custom handlers first (allows user overrides)
        if node_type in self._custom_handlers:
            return self._custom_handlers[node_type](node)

        # Check auto-generated handlers
        handler = self._handlers.get(node_type)

        if handler is None:
            # Unknown type - auto-register it now (future-proofing)
            handler = self._auto_generate_handler(node_type)
            self._handlers[node_type] = handler

        return handler(node)

    def register(self, node_type: Type[ast.AST]) -> Callable:
        """
        Decorator to register custom handlers that override auto-generated ones.

        Use this when specialized extraction is needed for specific nodes.
            **trying this without any specialized extraction for now**

        Usage:
            @visitor.register(ast.FunctionDef)
            def custom_function_handler(node):
                return {
                    "type": "FunctionDef",
                    "name": node.name,
                    "is_async": False,
                    "complexity": calculate_complexity(node),
                }
        """
        def decorator(func: Callable[[ast.AST], Dict[str, Any]]) -> Callable:
            self._custom_handlers[node_type] = func
            return func
        return decorator

    def register_handler(
        self,
        node_type: Type[ast.AST],
        handler: Callable[[ast.AST], Dict[str, Any]]
    ) -> None:
        """Programmatically register a custom handler without decorator syntax."""
        self._custom_handlers[node_type] = handler

    def parse_tree(self, code: str) -> Dict[str, Any]:
        """
        Parse code string into full recursive AST representation.
        """
        # Reset metrics
        self._visit_count = 0
        self._max_depth = 0
        self._type_counts = {}

        tree = ast.parse(code)

        nodes = []
        for node in tree.body:
            parsed = self._parse_recursive(node, depth=0)
            nodes.append(parsed)

        return {
            "nodes": nodes,
            "metadata": {
                "total_nodes_visited": self._visit_count,
                "max_depth": self._max_depth,
                "top_level_count": len(nodes),
                "type_frequency": self._type_counts,
                "registered_handlers": len(self._handlers),
            }
        }

    def _parse_recursive(
        self,
        node: ast.AST,
        depth: int = 0
    ) -> Dict[str, Any]:
        """
        Parse a node with depth tracking for complexity analysis.
        """
        self._max_depth = max(self._max_depth, depth)

        result = self.visit(node)
        result["depth"] = depth

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Return parsing metrics for complexity analysis."""
        return {
            "visit_count": self._visit_count,
            "max_depth": self._max_depth,
            "type_counts": self._type_counts.copy(),
            "registered_handlers": len(self._handlers),
            "custom_handlers": len(self._custom_handlers),
        }

    def get_registered_types(self) -> List[str]:
        return sorted([t.__name__ for t in self._handlers.keys()])


# SECURITY VISITOR: Dynamic security rule enforcement on AST nodes

class SecurityVisitor(DynamicASTVisitor):
    """
    Extends DynamicASTVisitor with security rule validation.

    Rules are stored in a registry keyed by node type, enabling O(1) lookup.
    Rules can block dangerous patterns or transform nodes to safe alternatives.
    """

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
        super().__init__()
        self._security_rules: Dict[str, List[Callable]] = {}
        self._violations: List[Dict[str, Any]] = []
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
        """Validate a node and its children against security rules."""
        self._violations = []
        self._validate_recursive(node_json)
        return self._violations

    def _validate_recursive(self, node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type and node_type in self._security_rules:
                for rule in self._security_rules[node_type]:
                    violation = rule(node)
                    if violation:
                        self._violations.append(violation)
            for value in node.values():
                self._validate_recursive(value)
        elif isinstance(node, list):
            for item in node:
                self._validate_recursive(item)

    def _check_dangerous_call(self, node: Dict) -> Dict[str, Any] | None:
        func = node.get("func")
        func_name = None

        if isinstance(func, str):
            func_name = func
        elif isinstance(func, dict):
            if func.get("type") == "Name":
                func_name = func.get("id")
            elif func.get("type") == "Attribute":
                module = func.get("value")
                attr = func.get("attr")
                if isinstance(module, str) and module in self.DANGEROUS_MODULES:
                    if attr in self.DANGEROUS_MODULES[module]:
                        return {
                            "rule": "dangerous_module_call",
                            "severity": "high",
                            "message": f"Dangerous call: {module}.{attr}",
                            "node": node,
                        }

        if func_name and func_name in self.DANGEROUS_CALLS:
            return {
                "rule": "dangerous_builtin",
                "severity": "high",
                "message": f"Dangerous builtin: {func_name}",
                "node": node,
            }
        return None

    def _check_sql_injection(self, node: Dict) -> Dict[str, Any] | None:
        func = node.get("func")
        if isinstance(func, dict) and func.get("type") == "Attribute":
            if func.get("attr") in self.SQL_FUNCTIONS:
                args = node.get("args", [])
                for arg in args:
                    if isinstance(arg, dict):
                        if arg.get("type") == "BinOp" and arg.get("op") in ("+", "%"):
                            return {
                                "rule": "sql_injection",
                                "severity": "critical",
                                "message": "Potential SQL injection: string concatenation in query",
                                "node": node,
                            }
                        if arg.get("type") == "JoinedStr":
                            return {
                                "rule": "sql_injection",
                                "severity": "critical",
                                "message": "Potential SQL injection: f-string in query",
                                "node": node,
                            }
        return None

    def _check_dangerous_import(self, node: Dict) -> Dict[str, Any] | None:
        if node.get("type") == "Import":
            names = node.get("names", [])
            for alias in names:
                name = alias.get("name") if isinstance(alias, dict) else alias
                if name in self.DANGEROUS_MODULES:
                    return {
                        "rule": "dangerous_import",
                        "severity": "medium",
                        "message": f"Importing potentially dangerous module: {name}",
                        "node": node,
                    }
        elif node.get("type") == "ImportFrom":
            module = node.get("module")
            if module in self.DANGEROUS_MODULES:
                return {
                    "rule": "dangerous_import",
                    "severity": "medium",
                    "message": f"Importing from dangerous module: {module}",
                    "node": node,
                }
        return None


# GLOBAL VISITOR INSTANCES

_default_visitor = DynamicASTVisitor()
_security_visitor = SecurityVisitor()


# PUBLIC API FUNCTIONS

def code_to_ast_string(code: str) -> str:
    """Convert code to AST dump string (legacy function)."""
    tree = ast.parse(code)
    return ast.dump(tree, indent=2)


def code_to_ast_json(code: str) -> str:
    """Convert code to AST JSON string (legacy function)."""
    tree = ast.parse(code)
    return json.dumps({"ast": ast.dump(tree)})

def parse_prompt(code: str) -> dict:
    """
    Parse code into structured AST representation

    Uses auto-discovery to handle any Python AST node type
    """
    visitor = DynamicASTVisitor()
    tree = ast.parse(code)
    nodes = [visitor.visit(node) for node in tree.body]
    return {"nodes": nodes}


# def _parse_value(value):
#     """Helper to parse a value that can be a string, dict, or literal."""
#     if isinstance(value, str):
#         return ast.Name(id=value, ctx=ast.Load())
#     elif isinstance(value, dict):
#         return json_to_ast(value)
#     else:
#         return ast.Constant(value=value)


_STRING_FIELDS = {"name", "attr", "arg", "id", "module", "asname"}


def _convert_value(value: Any, field_name: str = "") -> Any:
    """Convert a value from LLM JSON to AST node."""
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


def json_to_ast(node_json: dict) -> ast.AST:
    node_type_name = node_json.get("type")
    node_class = getattr(ast, node_type_name, None)

    if node_class is None:
        raise ValueError(f"Unknown AST type: {node_type_name}")

    kwargs = {}
    for field in node_class._fields:
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
    response = response.replace(": True", ": true")
    response = response.replace(": False", ": false")
    response = response.replace(": None", ": null")
    return response


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

    # Prompt going to stay static, because if we provide the dynamic list, it's going to get too complicated and i could see it 
    # confusing the LLM more than helping it.
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

        # Step 3: Try to convert JSON to code
        try:
            code = json_to_code(response_json)
            return response_json, code, violations
        except (ValueError, KeyError, TypeError) as e:
            last_error = e
            if attempt < max_retries - 1:
                fix_prompt = f"""The following AST JSON has an error when converting to Python code. Fix the JSON structure and return ONLY the corrected JSON, nothing else.

Error: {e}

Invalid AST JSON:
{json.dumps(response_json, indent=2)}"""
                response = _clean_json_response(llm_func(fix_prompt))
                print(f"---AST Fix Attempt {attempt + 1}---")
                print(response[:500])
                print("---End Fix Attempt---")
                continue
            raise last_error

    raise last_error