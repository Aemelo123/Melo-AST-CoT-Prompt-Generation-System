"""Tests for main module."""

from melo_ast_cot.main import main


def test_main(capsys) -> None:
    """Test main function runs without error."""
    main()
    captured = capsys.readouterr()
    assert "Melo AST CoT" in captured.out
