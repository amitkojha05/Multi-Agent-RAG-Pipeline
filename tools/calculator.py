"""tools/calculator.py

safe_eval() evaluates mathematical expressions without using eval().
Uses Python ast module to parse and evaluate only safe node types.
"""
import ast
import math
import operator
from typing import Union


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
}


def safe_eval(expression: str) -> str:
    """Safely evaluate a math expression. Returns result as string or error."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return str(round(result, 10) if isinstance(result, float) else result)
    except Exception as e:  # pragma: no cover - error path
        return f"Calculator error: {e}"


def _eval_node(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value}")

    if isinstance(node, ast.Name):
        if node.id in SAFE_NAMES:
            return SAFE_NAMES[node.id]
        raise ValueError(f"Unknown name: {node.id}")

    if isinstance(node, ast.BinOp):
        op_fn = SAFE_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op_fn(_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op_fn = SAFE_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError("Unsupported unary operator")
        return op_fn(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_NAMES:
            fn = SAFE_NAMES[node.func.id]
            args = [_eval_node(a) for a in node.args]
            return fn(*args)
        raise ValueError(f"Function call not permitted: {ast.dump(node.func)}")

    raise ValueError(f"Unsupported AST node: {type(node)}")

