"""Turn what the vision agent said about attributes into values CVAT accepts.

CVAT labels carry attributes with an input type: checkbox, radio, select, number, or text.
A value the agent returns can be free text ("Yes"), a Python bool, a number as a float, or a
choice that is not in the list. Writing those as they are makes CVAT show a wrong value or
refuse the whole annotation upload. This module maps each value onto the attribute's own
rules, and fills the attributes the agent skipped with their default.
"""

from __future__ import annotations

import math

TRUE_WORDS = {"true", "yes", "y", "1", "on", "checked", "present"}
FALSE_WORDS = {"false", "no", "n", "0", "off", "unchecked", "absent", "none"}


def _text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _choices(spec: dict) -> list[str]:
    return [str(item) for item in spec.get("values") or []]


def _default(spec: dict) -> str | None:
    kind = spec.get("input_type") or "text"
    value = spec.get("default_value")
    if value is None:
        value = spec.get("default")
    value = "" if value is None else str(value)
    if kind == "checkbox":
        return "true" if value.strip().lower() in TRUE_WORDS else "false"
    if kind in {"select", "radio"}:
        choices = _choices(spec)
        match = _match_choice(value, choices) if value else None
        return match if match is not None else (choices[0] if choices else None)
    if kind == "number":
        number = _number(value, spec) if value != "" else None
        return number
    return value if value != "" else None


def _match_choice(value: str, choices: list[str]) -> str | None:
    if value in choices:
        return value
    lowered = value.strip().lower()
    for choice in choices:
        if choice.strip().lower() == lowered:
            return choice
    return None


def _number(value, spec: dict) -> str | None:
    try:
        number = float(_text(value))
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    bounds = _choices(spec)  # CVAT stores a number attribute's range as [min, max, step]
    try:
        low, high = float(bounds[0]), float(bounds[1])
        number = min(max(number, low), high)
        step = float(bounds[2]) if len(bounds) > 2 else 0.0
        if step > 0:
            number = low + round((number - low) / step) * step
            number = min(max(number, low), high)
    except (IndexError, ValueError):
        pass
    return str(int(number)) if number.is_integer() else str(round(number, 6))


def clean_value(value, spec: dict) -> str | None:
    """A value that fits the attribute, or None when it cannot be made to fit."""
    kind = spec.get("input_type") or "text"
    if value is None:
        return None
    if kind == "checkbox":
        word = _text(value).lower()
        if word in TRUE_WORDS:
            return "true"
        if word in FALSE_WORDS:
            return "false"
        return None
    if kind in {"select", "radio"}:
        return _match_choice(_text(value), _choices(spec))
    if kind == "number":
        return _number(value, spec)
    text = _text(value)
    return text if text else None


def cvat_attributes(given: dict | None, specs: list[dict]) -> list[dict]:
    """[{spec_id, value}] for every attribute of the label, valid for CVAT.

    The agent's value wins when it fits. Otherwise the attribute's default is used, so
    CVAT never shows an empty or invalid attribute on a shape the agent drew.
    """
    wanted = {str(name).strip().lower(): value for name, value in (given or {}).items()}
    out: list[dict] = []
    for spec in specs or []:
        if spec.get("id") is None or not spec.get("name"):
            continue
        value = clean_value(wanted.get(str(spec["name"]).strip().lower()), spec)
        if value is None:
            value = _default(spec)
        if value is None:
            continue
        out.append({"spec_id": spec["id"], "value": value})
    return out


def describe(spec: dict) -> dict:
    """What the vision agent is told about one attribute."""
    kind = spec.get("input_type") or "text"
    item: dict = {"name": spec.get("name"), "input_type": kind}
    values = _choices(spec)
    if kind == "checkbox":
        item["values"] = ["true", "false"]
    elif kind == "number":
        if len(values) >= 2:
            item["min"], item["max"] = values[0], values[1]
            if len(values) > 2:
                item["step"] = values[2]
    elif values:
        item["values"] = values
    default = _default(spec)
    if default is not None:
        item["default"] = default
    return item
