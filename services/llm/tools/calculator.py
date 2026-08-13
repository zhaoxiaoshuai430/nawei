"""Safe calculator tool for tutoring math questions."""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable


CALC_MARKERS = ("计算", "算一下", "等于多少", "是多少", "平方根", "开方")

ALLOWED_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
ALLOWED_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
ALLOWED_FUNCS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
}
ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}


@dataclass(frozen=True)
class CalculatorResult:
    expression: str
    result: float
    result_text: str
    source: str = "safe-local-calculator"

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "result": self.result,
            "result_text": self.result_text,
            "source": self.source,
        }


class CalculatorTool:
    """Evaluate simple numeric expressions without Python eval."""

    def extract_expression(self, text: str) -> str | None:
        if not any(marker in text for marker in CALC_MARKERS) and not re.search(r"\d\s*[\+\-\*/×÷^]\s*\d", text):
            return None
        normalized = (
            text.replace("×", "*")
            .replace("÷", "/")
            .replace("^", "**")
            .replace("（", "(")
            .replace("）", ")")
            .replace("，", ",")
        )
        match = re.search(r"([0-9piPIeE\s\+\-\*/%\.\(\),\*]+)", normalized)
        if not match:
            return None
        expression = match.group(1).strip()
        return expression if any(char.isdigit() for char in expression) else None

    def calculate_from_text(self, text: str) -> CalculatorResult | None:
        expression = self.extract_expression(text)
        if not expression:
            return None
        return self.calculate(expression)

    def calculate(self, expression: str) -> CalculatorResult:
        expression = expression.strip()
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree.body)
        return CalculatorResult(expression=expression, result=value, result_text=_format_number(value))


def format_calculator_system_prompt(result: CalculatorResult) -> str:
    return (
        "实时工具结果：计算器\n"
        f"- 表达式：{result.expression}\n"
        f"- 结果：{result.result_text}\n"
        "回答计算题时必须基于以上工具结果，并简要说明关键步骤。"
    )


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in ALLOWED_CONSTANTS:
        return float(ALLOWED_CONSTANTS[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return float(ALLOWED_BINARY[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY:
        return float(ALLOWED_UNARY[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS:
        args = [_eval_node(arg) for arg in node.args]
        return float(ALLOWED_FUNCS[node.func.id](*args))
    raise ValueError("表达式包含不支持的内容")


def _format_number(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:.10g}"
