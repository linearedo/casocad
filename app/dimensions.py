from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import numpy as np

UNIT_FACTORS = {
    "": 1.0,
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "millimetre": 0.001,
    "millimetres": 0.001,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "kilometre": 1000.0,
    "kilometres": 1000.0,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    '"': 0.0254,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "'": 0.3048,
}
DIMENSION_UNIT_PATTERN = (
    "kilometers|kilometer|kilometres|kilometre|"
    "millimeters|millimeter|millimetres|millimetre|"
    "centimeters|centimeter|centimetres|centimetre|"
    "meters|meter|metres|metre|"
    "inches|inch|feet|foot|"
    "km|mm|cm|m|in|ft|\"|'"
)
DIMENSION_INPUT_PATTERN = re.compile(
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    rf"\s*({DIMENSION_UNIT_PATTERN})?",
    re.IGNORECASE,
)
DIMENSION_SEPARATOR_PATTERN = re.compile(r"[\s,;xX]+")
DIMENSION_EXPRESSION_SEPARATOR_PATTERN = re.compile(r"\s*[,;xX]\s*")
DIMENSION_ENTRY_BASE_ALLOWED = set("0123456789.,xX; \"'")
DIMENSION_ENTRY_UNIT_WORD_ALLOWED = set(
    "kilometresmillimeterscentimetresmetersinchesfoot"
)
DIMENSION_ENTRY_FORMULA_ALLOWED = set("*/()+-")
DISPLACEMENT_SIGN_CHARS = set("+-")
DIMENSION_ENTRY_START_CHARS = set("0123456789.(")


@dataclass(frozen=True)
class LengthUnit:
    """A user-selectable working unit for dimension entry and display.

    ``key`` is the canonical abbreviation (also used as the display suffix)
    and must be parseable, so adding a unit means one ``LengthUnit`` row here
    plus its spellings in ``UNIT_FACTORS`` / ``DIMENSION_UNIT_PATTERN``.
    """

    key: str
    label: str
    factor: float  # meters per unit


LENGTH_UNITS: tuple[LengthUnit, ...] = (
    LengthUnit("km", "Kilometers", UNIT_FACTORS["km"]),
    LengthUnit("m", "Meters", UNIT_FACTORS["m"]),
    LengthUnit("cm", "Centimeters", UNIT_FACTORS["cm"]),
    LengthUnit("mm", "Millimeters", UNIT_FACTORS["mm"]),
)
DEFAULT_LENGTH_UNIT = LENGTH_UNITS[1]


def length_unit(key: str) -> LengthUnit:
    for unit in LENGTH_UNITS:
        if unit.key == key:
            return unit
    raise KeyError(f"unknown working unit: {key}")


def _literal_factor(unit: str, unit_factor: float) -> float:
    """Meters per typed literal: explicit units win, bare numbers are read in
    the working unit."""
    return UNIT_FACTORS[unit] if unit else unit_factor


def parse_measurement_entry(
    text: str, unit_factor: float = 1.0
) -> tuple[float, ...]:
    """Parse dimension text into values expressed in the working unit
    (``unit_factor`` meters per unit; the default keeps meters in, meters
    out)."""
    try:
        return tuple(
            _parse_measurement_expression(part, unit_factor)
            for part in _split_measurement_expressions(text)
        )
    except ValueError:
        pass
    values: list[float] = []
    consumed = []
    for match in DIMENSION_INPUT_PATTERN.finditer(text):
        unit = (match.group(2) or "").lower()
        if unit not in UNIT_FACTORS:
            raise ValueError(f"unknown unit: {match.group(2)}")
        values.append(
            float(match.group(1)) * _literal_factor(unit, unit_factor) / unit_factor
        )
        consumed.append((match.start(), match.end()))
    if not values:
        raise ValueError("dimension entries must contain at least one number")
    cursor = 0
    separators = []
    for start, end in consumed:
        separators.append(text[cursor:start])
        cursor = end
    separators.append(text[cursor:])
    if any(
        segment and not DIMENSION_SEPARATOR_PATTERN.fullmatch(segment)
        for segment in separators
    ):
        raise ValueError(
            "dimension entries must be numbers with optional units"
        )
    return tuple(values)


def _split_measurement_expressions(text: str) -> tuple[str, ...]:
    parts = tuple(
        part.strip()
        for part in DIMENSION_EXPRESSION_SEPARATOR_PATTERN.split(text)
        if part.strip()
    )
    if parts:
        return parts
    raise ValueError("dimension entries must contain at least one number")


def _parse_measurement_expression(text: str, unit_factor: float = 1.0) -> float:
    expression = _replace_measurement_literals(text, unit_factor)
    try:
        parsed = ast.parse(expression, mode="eval")
        value = _evaluate_measurement_ast(parsed)
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid dimension expression") from error
    if not np.isfinite(value):
        raise ValueError("dimension expression must be finite")
    return float(value)


def _replace_measurement_literals(text: str, unit_factor: float = 1.0) -> str:
    pieces: list[str] = []
    cursor = 0
    found = False
    for match in DIMENSION_INPUT_PATTERN.finditer(text):
        number_text = match.group(1)
        prefix = text[cursor:match.start()]
        if number_text.startswith(("+", "-")):
            previous = text[: match.start()].rstrip()
            previous_character = previous[-1] if previous else ""
            if previous_character and previous_character not in "(+-*/,:;xX":
                prefix += number_text[0]
                number_text = number_text[1:]
        unit = (match.group(2) or "").lower()
        if unit not in UNIT_FACTORS:
            raise ValueError(f"unknown unit: {match.group(2)}")
        pieces.append(prefix)
        pieces.append(
            str(float(number_text) * _literal_factor(unit, unit_factor) / unit_factor)
        )
        cursor = match.end()
        found = True
    if not found:
        raise ValueError("dimension entries must contain at least one number")
    pieces.append(text[cursor:])
    return "".join(pieces)


def _evaluate_measurement_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_measurement_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_measurement_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate_measurement_ast(node.left)
        right = _evaluate_measurement_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("invalid dimension expression")


def parse_scalar_entry(text: str) -> float:
    try:
        parsed = ast.parse(text.strip(), mode="eval")
        value = _evaluate_measurement_ast(parsed)
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid scalar expression") from error
    if not np.isfinite(value):
        raise ValueError("scalar expression must be finite")
    return float(value)


def parse_dimension_entry(
    text: str, unit_factor: float = 1.0
) -> tuple[float, ...]:
    values = parse_measurement_entry(text, unit_factor)
    if not values or any(value <= 0.0 or not np.isfinite(value) for value in values):
        raise ValueError("dimension entries must be positive finite numbers")
    return values


def parse_displacement_entry(
    text: str, unit_factor: float = 1.0
) -> tuple[float, ...]:
    values = parse_measurement_entry(text, unit_factor)
    if not values or any(not np.isfinite(value) for value in values):
        raise ValueError("displacement entries must be finite numbers")
    return values


def dimension_entry_text(
    text: str,
    action: str,
    has_create_start: bool,
    current_text: str = "",
) -> str:
    normalized = text.lower()
    allowed = set(DIMENSION_ENTRY_BASE_ALLOWED)
    allowed.update(DIMENSION_ENTRY_FORMULA_ALLOWED)
    if current_text:
        allowed.update(DIMENSION_ENTRY_UNIT_WORD_ALLOWED)
    exponent_sign = current_text.rstrip().lower().endswith("e")
    sign_allowed = (
        action == "move"
        or (action == "create" and not has_create_start)
        or exponent_sign
    )
    if sign_allowed:
        allowed.update(DISPLACEMENT_SIGN_CHARS)
    if not current_text:
        start_chars = set(DIMENSION_ENTRY_START_CHARS)
        if sign_allowed:
            start_chars.update(DISPLACEMENT_SIGN_CHARS)
        if not normalized or any(character not in start_chars for character in normalized):
            return ""
    if normalized and all(character in allowed for character in normalized):
        return normalized
    return ""
