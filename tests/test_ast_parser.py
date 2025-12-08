"""Basic unit tests for the Dynamic AST Parser."""

from melo_ast_cot.ast_parser import (
    DynamicASTVisitor,
    SecurityVisitor,
    parse_prompt,
    _extract_args,
    _extract_imports,
)


def test_auto_registers_handlers():
    visitor = DynamicASTVisitor()
    assert len(visitor.get_registered_types()) > 60


def test_parse_prompt_returns_nodes():
    code = "x = 1"
    result = parse_prompt(code)
    assert "nodes" in result
    assert len(result["nodes"]) == 1


def test_parse_function():
    code = "def foo(x, y): pass"
    result = parse_prompt(code)
    func = result["nodes"][0]
    assert func["type"] == "FunctionDef"
    assert func["name"] == "foo"


def test_extract_args_new_format():
    args = {"args": [{"arg": "x"}, {"arg": "y"}]}
    assert _extract_args(args) == ["x", "y"]


def test_extract_args_legacy_format():
    args = ["a", "b"]
    assert _extract_args(args) == ["a", "b"]


def test_extract_imports():
    node = {"type": "Import", "names": [{"name": "os"}]}
    assert _extract_imports(node) == ["os"]


def test_extract_import_from():
    node = {"type": "ImportFrom", "module": "pathlib"}
    assert _extract_imports(node) == ["pathlib"]


def test_security_detects_eval():
    visitor = SecurityVisitor()
    node = {"type": "Call", "func": "eval", "args": []}
    violations = visitor.validate(node)
    assert len(violations) == 1
    assert violations[0]["rule"] == "dangerous_builtin"


def test_security_detects_os_system():
    visitor = SecurityVisitor()
    node = {"type": "Call", "func": {"type": "Attribute", "value": "os", "attr": "system"}, "args": []}
    violations = visitor.validate(node)
    assert len(violations) == 1
    assert violations[0]["rule"] == "dangerous_module_call"


def test_security_detects_sql_injection():
    visitor = SecurityVisitor()
    node = {
        "type": "Call",
        "func": {"type": "Attribute", "value": "cursor", "attr": "execute"},
        "args": [{"type": "BinOp", "left": "query", "op": "+", "right": "user_input"}]
    }
    violations = visitor.validate(node)
    assert len(violations) == 1
    assert violations[0]["rule"] == "sql_injection"


def test_security_allows_safe_code():
    visitor = SecurityVisitor()
    node = {"type": "Call", "func": "print", "args": [{"type": "Constant", "value": "hello"}]}
    violations = visitor.validate(node)
    assert len(violations) == 0
