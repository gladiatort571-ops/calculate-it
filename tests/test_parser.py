"""Tests for the expression parser."""

from calculate_it.parser import parse_expression, ExpressionParser


def test_simple_number():
    result = parse_expression("42")
    assert "42" in result.ast_string


def test_simple_variable():
    result = parse_expression("x")
    assert "x" in result.ast_string


def test_implicit_multiplication():
    result = parse_expression("2x")
    assert "2" in result.ast_string
    assert "x" in result.ast_string


def test_function_parsing():
    result = parse_expression("sinx")
    assert "sin" in result.ast_string


def test_empty_expression():
    result = parse_expression("")
    assert result.ast_string == ""


def test_constants():
    result = parse_expression("pi")
    assert "pi" in result.ast_string


def test_equation():
    result = parse_expression("2x+3=7")
    assert "=" in result.ast_string


def test_scientific_notation():
    result = parse_expression("1.5e3")
    assert "1.5" in result.ast_string
