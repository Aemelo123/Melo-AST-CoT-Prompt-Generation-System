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


def _parse_value(value):
    """Helper to parse a value that can be a string, dict, or literal."""
    if isinstance(value, str):
        return ast.Name(id=value, ctx=ast.Load())
    elif isinstance(value, dict):
        return json_to_ast(value)
    else:
        return ast.Constant(value=value)


def json_to_ast(node_json: dict) -> ast.AST:
    node_type = node_json.get("type")

    # name - variable reference
    if node_type == "Name":
        return ast.Name(id=node_json.get("id") or node_json.get("value"), ctx=ast.Load())

    # constant - literal value
    elif node_type == "Constant":
        return ast.Constant(value=node_json["value"])

    # Variable/Identifier - aliases for Name (GPT sometimes uses these)
    elif node_type in ("Variable", "Identifier"):
        return ast.Name(id=node_json.get("id") or node_json.get("name") or node_json.get("value"), ctx=ast.Load())

    # Dict - dictionary literal
    elif node_type == "Dict":
        keys = []
        values = []
        for key in node_json.get("keys", []):
            keys.append(_parse_value(key) if key is not None else None)
        for val in node_json.get("values", []):
            values.append(_parse_value(val))
        return ast.Dict(keys=keys, values=values)

    # List - list literal
    elif node_type == "List":
        elts = [_parse_value(e) for e in node_json.get("elements", node_json.get("elts", []))]
        return ast.List(elts=elts, ctx=ast.Load())

    # Tuple - tuple literal
    elif node_type == "Tuple":
        elts = node_json.get("elts") or node_json.get("elements") or node_json.get("values", [])
        return ast.Tuple(elts=[_parse_value(e) for e in elts], ctx=ast.Load())

    # Subscript - indexing (e.g., data[0])
    elif node_type == "Subscript":
        value = _parse_value(node_json["value"])
        slice_node = node_json.get("slice")
        if isinstance(slice_node, dict) and slice_node.get("type") == "Index":
            slice_ast = _parse_value(slice_node["value"])
        else:
            slice_ast = _parse_value(slice_node)
        return ast.Subscript(value=value, slice=slice_ast, ctx=ast.Load())

    # Lambda - anonymous function
    elif node_type == "Lambda":
        args_list = node_json.get("args", [])
        arg_nodes = []
        for a in args_list:
            name = a.get("arg") or a.get("name") if isinstance(a, dict) else a
            arg_nodes.append(ast.arg(arg=name))
        args = ast.arguments(posonlyargs=[], args=arg_nodes, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
        return ast.Lambda(args=args, body=_parse_value(node_json["body"]))

    # Expr - expression statement (wraps an expression)
    elif node_type == "Expr":
        value = _parse_value(node_json["value"])
        return ast.Expr(value=value)

    # Raise - raise exception
    elif node_type == "Raise":
        exc = None
        if "exc" in node_json:
            exc = _parse_value(node_json["exc"])
        elif "exception" in node_json:
            exc = _parse_value(node_json["exception"])
        return ast.Raise(exc=exc, cause=None)

    # Break/Continue - loop control
    elif node_type == "Break":
        return ast.Break()
    elif node_type == "Continue":
        return ast.Continue()

    # Pass - no-op statement
    elif node_type == "Pass":
        return ast.Pass()

    # attribute - e.g., yaml.safe_load
    elif node_type == "Attribute":
        value = node_json["value"]
        if isinstance(value, str):
            value_node = ast.Name(id=value, ctx=ast.Load())
        else:
            value_node = json_to_ast(value)
        return ast.Attribute(
            value=value_node,
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
        target = node_json.get("target") or node_json.get("targets")
        if isinstance(target, list):
            # tuple unpacking or single target in list
            if len(target) == 1:
                t = target[0]
                target_node = ast.Name(id=t, ctx=ast.Store()) if isinstance(t, str) else json_to_ast(t)
                target_node.ctx = ast.Store()
            else:
                elts = [ast.Name(id=t, ctx=ast.Store()) if isinstance(t, str) else json_to_ast(t) for t in target]
                for e in elts:
                    e.ctx = ast.Store()
                target_node = ast.Tuple(elts=elts, ctx=ast.Store())
        elif isinstance(target, str):
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

    # If - conditional
    elif node_type == "If":
        test = _parse_value(node_json["test"])
        body = [json_to_ast(item) for item in node_json.get("body", [])]
        orelse = [json_to_ast(item) for item in node_json.get("orelse", [])]

        return ast.If(test=test, body=body, orelse=orelse)

    # For - for loop
    elif node_type == "For":
        target = node_json["target"]
        if isinstance(target, str):
            target_node = ast.Name(id=target, ctx=ast.Store())
        else:
            target_node = json_to_ast(target)
            target_node.ctx = ast.Store()

        iter_node = _parse_value(node_json["iter"])
        body = [json_to_ast(item) for item in node_json.get("body", [])]
        orelse = [json_to_ast(item) for item in node_json.get("orelse", [])]

        return ast.For(target=target_node, iter=iter_node, body=body, orelse=orelse)

    # While - while loop
    elif node_type == "While":
        test = _parse_value(node_json["test"])
        body = [json_to_ast(item) for item in node_json.get("body", [])]
        orelse = [json_to_ast(item) for item in node_json.get("orelse", [])]

        return ast.While(test=test, body=body, orelse=orelse)

    # Try - try/except/finally
    elif node_type == "Try":
        body = [json_to_ast(item) for item in node_json.get("body", [])]
        orelse = [json_to_ast(item) for item in node_json.get("orelse", [])]
        finalbody = [json_to_ast(item) for item in node_json.get("finalbody", [])]

        handlers = []
        for handler in node_json.get("handlers", []):
            exc_type = handler.get("type")
            if exc_type:
                if isinstance(exc_type, str):
                    exc_type_node = ast.Name(id=exc_type, ctx=ast.Load())
                else:
                    exc_type_node = json_to_ast(exc_type)
            else:
                exc_type_node = None

            handler_name = handler.get("name")
            handler_body = [json_to_ast(item) for item in handler.get("body", [])]

            handlers.append(ast.ExceptHandler(
                type=exc_type_node,
                name=handler_name,
                body=handler_body
            ))

        return ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody)

    # BinOp - binary operation (e.g., a + b, x * y)
    elif node_type == "BinOp":
        left = _parse_value(node_json["left"])
        right = _parse_value(node_json["right"])
        op_str = node_json["op"]

        # Handle malformed "." operator as Attribute access (LLM mistake)
        if op_str == ".":
            # Convert to Attribute node - this is what the LLM meant
            return ast.Attribute(value=left, attr=node_json["right"].get("name") or node_json["right"].get("id") or str(node_json["right"]), ctx=ast.Load())

        op_map = {
            "Add": ast.Add(), "+": ast.Add(),
            "Sub": ast.Sub(), "-": ast.Sub(),
            "Mult": ast.Mult(), "*": ast.Mult(),
            "Div": ast.Div(), "/": ast.Div(),
            "FloorDiv": ast.FloorDiv(), "//": ast.FloorDiv(),
            "Mod": ast.Mod(), "%": ast.Mod(),
            "Pow": ast.Pow(), "**": ast.Pow(),
            "BitOr": ast.BitOr(), "|": ast.BitOr(),
            "BitXor": ast.BitXor(), "^": ast.BitXor(),
            "BitAnd": ast.BitAnd(), "&": ast.BitAnd(),
        }
        op = op_map.get(op_str)
        if not op:
            raise ValueError(f"Unknown binary operator: {op_str}")

        return ast.BinOp(left=left, op=op, right=right)

    # Compare - comparison (e.g., a < b, x == y)
    elif node_type == "Compare":
        left = _parse_value(node_json["left"])

        cmp_map = {
            "Eq": ast.Eq(), "==": ast.Eq(),
            "NotEq": ast.NotEq(), "!=": ast.NotEq(),
            "Lt": ast.Lt(), "<": ast.Lt(),
            "LtE": ast.LtE(), "<=": ast.LtE(),
            "Gt": ast.Gt(), ">": ast.Gt(),
            "GtE": ast.GtE(), ">=": ast.GtE(),
            "Is": ast.Is(), "is": ast.Is(),
            "IsNot": ast.IsNot(), "is not": ast.IsNot(),
            "In": ast.In(), "in": ast.In(),
            "NotIn": ast.NotIn(), "not in": ast.NotIn(),
        }

        ops = []
        comparators = []
        for comp in node_json["comparators"]:
            op_str = comp["op"]
            op = cmp_map.get(op_str)
            if not op:
                raise ValueError(f"Unknown comparison operator: {op_str}")
            ops.append(op)
            comparators.append(_parse_value(comp["value"]))

        return ast.Compare(left=left, ops=ops, comparators=comparators)

    # FunctionDef - function definition
    elif node_type == "FunctionDef":
        name = node_json["name"]
        args_list = node_json.get("args", [])
        body = [json_to_ast(item) for item in node_json.get("body", [])]

        # Build arguments - handle both string and dict formats
        arg_nodes = []
        for arg in args_list:
            if isinstance(arg, str):
                arg_nodes.append(ast.arg(arg=arg))
            elif isinstance(arg, dict):
                # LLM returned {"type": "Arg", "name": "x"} format
                arg_name = arg.get("name") or arg.get("arg") or arg.get("id")
                arg_nodes.append(ast.arg(arg=arg_name))

        arguments = ast.arguments(
            posonlyargs=[],
            args=arg_nodes,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        )

        return ast.FunctionDef(
            name=name,
            args=arguments,
            body=body,
            decorator_list=[],
            returns=None
        )

    else:
        raise ValueError(f"Unknown node type: {node_type}")


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


def get_llm_ast_json(parsed_prompt: dict, llm_func) -> tuple[dict, str]:
    """Generate AST JSON and convert to code, with retries for errors.

    Returns tuple of (json_response, generated_code).
    """
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

        # Step 2: Try to convert JSON to code
        try:
            code = json_to_code(response_json)
            return response_json, code
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