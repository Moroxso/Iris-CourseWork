"""Interpret a small integer-function grammar. Never execute submitted Python."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

LIMIT = 10**12
BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
BUILTINS = {"min", "max", "abs"}


class Unsupported(ValueError):
    """Source is outside the bounded benchmark grammar."""


def bounded(value: int) -> int:
    if abs(value) > LIMIT:
        raise Unsupported("integer bound exceeded")
    return value


def expression(node: ast.expr, names: dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
        return bounded(node.value)
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        return bounded(
            BINARY[type(node.op)](expression(node.left, names), expression(node.right, names))
        )
    if isinstance(node, ast.UnaryOp):
        value = expression(node.operand, names)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.Not):
            return not value
    if isinstance(node, ast.IfExp):
        return expression(node.body if expression(node.test, names) else node.orelse, names)
    if isinstance(node, ast.BoolOp):
        for item in node.values[:-1]:
            value = expression(item, names)
            if bool(value) == isinstance(node.op, ast.Or):
                return value
        return expression(node.values[-1], names)
    if isinstance(node, ast.Compare):
        left = expression(node.left, names)
        for op, right_node in zip(node.ops, node.comparators, strict=True):
            right = expression(right_node, names)
            if not COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        values = [expression(item, names) for item in node.args]
        if node.func.id == "min":
            return min(values)
        if node.func.id == "max":
            return max(values)
        if node.func.id == "abs" and len(values) == 1:
            return abs(values[0])
    raise Unsupported("expression outside grammar")


def statements(body: list[ast.stmt], names: dict[str, int]) -> int | None:
    for node in body:
        if isinstance(node, ast.Return) and node.value is not None:
            return expression(node.value, names)
        if isinstance(node, ast.Assign) and isinstance(target := node.targets[0], ast.Name):
            names[target.id] = expression(node.value, names)
        if isinstance(node, ast.If):
            result = statements(node.body if expression(node.test, names) else node.orelse, names)
            if result is not None:
                return result
    return None


def parse(answer: str, name: str, parameters: list[str]) -> list[ast.stmt]:
    if len(answer.encode("utf-8")) > 4096:
        raise Unsupported("source too large")
    source = answer.strip()
    fenced = re.fullmatch(r"\x60{3}(?:python|py)?[ \t]*\r?\n([\s\S]*?)\r?\n\x60{3}", source, re.I)
    tree = ast.parse(fenced[1] if fenced else source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise Unsupported("exactly one function required")
    function = tree.body[0]
    args = function.args

    def annotation_ok(node: ast.expr | None) -> bool:
        return node is None or (isinstance(node, ast.Name) and node.id == "int")

    if (
        function.name != name
        or [a.arg for a in args.args] != parameters
        or args.defaults
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or function.decorator_list
        or function.type_params
        or not annotation_ok(function.returns)
        or any(not annotation_ok(a.annotation) for a in args.args)
    ):
        raise Unsupported("function signature outside grammar")
    body = function.body
    if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    allowed = {
        ast.Return,
        ast.Assign,
        ast.If,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.IfExp,
        ast.BoolOp,
        ast.Compare,
        ast.Call,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.And,
        ast.Or,
        *BINARY,
        *COMPARE,
    }
    nodes = [node for stmt in body for node in ast.walk(stmt)]
    if not nodes or len(nodes) > 160 or any(type(node) not in allowed for node in nodes):
        raise Unsupported("statement outside grammar")
    for node in nodes:
        if isinstance(node, ast.Constant) and (
            type(node.value) not in (int, bool) or abs(node.value) > LIMIT
        ):
            raise Unsupported("constant outside grammar")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in BUILTINS
            or node.keywords
            or (len(node.args) != 1 if node.func.id == "abs" else not 2 <= len(node.args) <= 8)
        ):
            raise Unsupported("call outside grammar")
        if isinstance(node, ast.Assign) and (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or node.targets[0].id in BUILTINS
        ):
            raise Unsupported("assignment outside grammar")
        if isinstance(node, ast.Return) and node.value is None:
            raise Unsupported("return must contain a value")
    return body


def grade_function(answer: str, spec: dict[str, Any]) -> dict[str, Any]:
    checks = spec["checks"]
    try:
        body = parse(answer, spec["name"], spec["parameters"])
    except SyntaxError:
        return dict(exact=False, matched=0, total=len(checks), format_valid=False, status="syntax")
    except (Unsupported, RecursionError, UnicodeError):
        return dict(
            exact=False, matched=0, total=len(checks), format_valid=False, status="unsupported"
        )
    outcomes = []
    for check in checks:
        names = dict(zip(spec["parameters"], check["args"], strict=True))
        try:
            value = statements(body, names)
            outcomes.append("passed" if type(value) is int and value == check["value"] else "wrong")
        except (ArithmeticError, KeyError, Unsupported):
            outcomes.append("evaluation-error")
    matched = outcomes.count("passed")
    return dict(
        exact=matched == len(checks),
        matched=matched,
        total=len(checks),
        format_valid=True,
        status="evaluated",
        outcomes=outcomes,
    )
