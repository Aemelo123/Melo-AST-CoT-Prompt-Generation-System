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