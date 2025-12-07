"""
https://docs.python.org/3/library/ast.html#

"""

import ast
import json


def code_to_ast_string(code: str) -> str:
    tree = ast.parse(code)
    return ast.dump(tree, indent=2)


def code_to_ast_json(code: str) -> str:
    tree = ast.parse(code)
    return json.dumps({"ast": ast.dump(tree)})

def parse_prompt(code: str) -> dict:
    tree = ast.parse(code)
    
    nodes = []
    
    for node in tree.body:
        node_info = {"type": type(node).__name__}
        
        # import statements
        if isinstance(node, ast.Import):
            node_info["modules"] = [alias.name for alias in node.names]
        
        # import from statements
        elif isinstance(node, ast.ImportFrom):
            node_info["module"] = node.module
            node_info["names"] = [alias.name for alias in node.names]
        
        # function definitions
        elif isinstance(node, ast.FunctionDef):
            node_info["name"] = node.name
            node_info["args"] = [arg.arg for arg in node.args.args]
            node_info["docstring"] = ast.get_docstring(node)
        
        # class definitions
        elif isinstance(node, ast.ClassDef):
            node_info["name"] = node.name
            node_info["bases"] = [ast.dump(base) for base in node.bases]
            node_info["docstring"] = ast.get_docstring(node)

        # variable assignments
        elif isinstance(node, ast.Assign):
            node_info["targets"] = [ast.dump(t) for t in node.targets]
            node_info["value"] = ast.dump(node.value)

        # expression statements
        elif isinstance(node, ast.Expr):
            node_info["value"] = ast.dump(node.value)

        # Fallback for anything else
        else:
            node_info["raw"] = ast.dump(node)
        
        nodes.append(node_info)
    
    return {"nodes": nodes}


def json_to_ast(node_json: dict) -> ast.AST:
    node_type = node_json.get("type")

    # name - variable reference
    if node_type == "Name":
        return ast.Name(id=node_json["id"], ctx=ast.Load())

    # constant - literal value
    elif node_type == "Constant":
        return ast.Constant(value=node_json["value"])

    # attribute - e.g., yaml.safe_load
    elif node_type == "Attribute":
        return ast.Attribute(
            value=ast.Name(id=node_json["value"], ctx=ast.Load()),
            attr=node_json["attr"],
            ctx=ast.Load()
        )

    # call - function call
    elif node_type == "Call":
        # handle func (can be string or nested Attribute)
        func = node_json["func"]
        if isinstance(func, str):
            func_node = ast.Name(id=func, ctx=ast.Load())
        else:
            func_node = json_to_ast(func)

        # handle args (can be strings or nested nodes)
        args = []
        for arg in node_json.get("args", []):
            if isinstance(arg, str):
                args.append(ast.Name(id=arg, ctx=ast.Load()))
            elif isinstance(arg, dict):
                args.append(json_to_ast(arg))
            else:
                args.append(ast.Constant(value=arg))

        return ast.Call(func=func_node, args=args, keywords=[])

    # assign - variable assignment
    elif node_type == "Assign":
        target = node_json["target"]
        if isinstance(target, str):
            target_node = ast.Name(id=target, ctx=ast.Store())
        else:
            target_node = json_to_ast(target)
            target_node.ctx = ast.Store()

        value = node_json["value"]
        if isinstance(value, dict):
            value_node = json_to_ast(value)
        else:
            value_node = ast.Name(id=value, ctx=ast.Load())

        return ast.Assign(targets=[target_node], value=value_node)

    # return
    elif node_type == "Return":
        value = node_json.get("value")
        if value is None:
            value_node = None
        elif isinstance(value, str):
            value_node = ast.Name(id=value, ctx=ast.Load())
        else:
            value_node = json_to_ast(value)

        return ast.Return(value=value_node)

    # With - context manager
    elif node_type == "With":
        context_expr = json_to_ast(node_json["context_expr"])
        var = node_json.get("var")
        optional_vars = ast.Name(id=var, ctx=ast.Store()) if var else None

        body = [json_to_ast(item) for item in node_json.get("body", [])]

        return ast.With(
            items=[ast.withitem(context_expr=context_expr, optional_vars=optional_vars)],
            body=body
        )

    else:
        raise ValueError(f"Unknown node type: {node_type}")


def json_to_code(node_json: dict) -> str:
    node = json_to_ast(node_json)
    ast.fix_missing_locations(node)
    return ast.unparse(node)


def get_llm_ast_json(parsed_prompt: dict, llm_func) -> dict:
    # extract function info from parsed prompt
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

    prompt = f"""You are generating Python code as AST nodes in JSON format.

Function to implement:
- Name: {func_info["name"]}
- Arguments: {func_info["args"]}
- Description: {func_info["docstring"]}
- Available imports: {imports}

Generate the function body as a JSON object representing AST nodes.
Use these node types:
- With: {{"type": "With", "context_expr": <Call node>, "var": "variable_name", "body": [<nodes>]}}
- Call: {{"type": "Call", "func": "function_name" or {{"type": "Attribute", "value": "module", "attr": "function"}}, "args": [<args>]}}
- Assign: {{"type": "Assign", "target": "variable_name", "value": <node>}}
- Return: {{"type": "Return", "value": "variable_name" or <node>}}
- Constant: {{"type": "Constant", "value": "literal_value"}}

Return ONLY the JSON object, no explanation."""

    response = llm_func(prompt)

    # strip markdown code blocks if present
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    print("---LLM Raw Response---")
    print(response)

    # parse the JSON from the response
    response_json = json.loads(response)

    return response_json