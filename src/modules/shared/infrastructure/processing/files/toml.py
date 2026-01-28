import datetime
from pathlib import Path
from typing import Any

import tomllib

from .base import BaseFileHandler


class TomlFileHandler(BaseFileHandler):
    """Handles serialization/deserialization of parameter dicts using TOML."""

    def save(self, obj: Any, file_path: Path):
        if not isinstance(obj, dict):
            raise TypeError("TOML handler expects a dict payload.")
        try:
            content = _dump_toml(obj)
            with open(file_path, "w") as f:
                f.write(content)
        except Exception as e:
            raise IOError(f"Failed to save object to {file_path}: {e}") from e

    def load(self, file_path: Path) -> Any:
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found at {file_path}")
        try:
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            raise IOError(f"Failed to load object from {file_path}: {e}") from e


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    _write_table(lines, data, prefix=[])
    return "\n".join(lines) + "\n"


def _write_table(lines: list[str], data: dict[str, Any], prefix: list[str]) -> None:
    scalar_items: dict[str, Any] = {}
    table_items: dict[str, dict[str, Any]] = {}

    for key in sorted(data.keys()):
        value = data[key]
        if value is None:
            continue
        if isinstance(value, dict):
            table_items[key] = value
        else:
            scalar_items[key] = value

    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")

    for key, value in scalar_items.items():
        lines.append(f"{key} = {_format_value(value)}")

    for key, value in table_items.items():
        lines.append("")
        _write_table(lines, value, prefix + [key])


def _format_value(value: Any) -> str:
    if isinstance(value, (datetime.date, datetime.datetime)):
        return _quote_string(value.isoformat())
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _quote_string(value)
    if isinstance(value, (list, tuple)):
        if any(v is None for v in value):
            raise TypeError("TOML arrays cannot contain null values.")
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def _quote_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
    return f"\"{escaped}\""
