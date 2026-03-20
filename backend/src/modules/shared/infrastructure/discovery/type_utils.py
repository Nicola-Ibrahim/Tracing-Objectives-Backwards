import inspect
from enum import Enum
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel


def is_pydantic_model(cls: Any) -> bool:
    """Checks if a class is a Pydantic model."""
    return inspect.isclass(cls) and issubclass(cls, BaseModel)


def is_enum(cls: Any) -> bool:
    """Checks if a class is an Enum."""
    return inspect.isclass(cls) and issubclass(cls, Enum)


def _is_discriminator(name: str, annotation: Any) -> bool:
    """Checks if a field is a Pydantic discriminator (Literal 'type')."""
    return name == "type" and get_origin(annotation) is Literal


def get_type_name(typ: Any) -> str:
    """Returns a human-readable type name, handling Pydantic models, Lists, and Dicts."""
    if typ == inspect.Parameter.empty:
        return "any"

    if is_pydantic_model(typ):
        return typ.__name__

    if is_enum(typ):
        return "enum"

    origin = get_origin(typ)
    if origin is list:
        args = get_args(typ)
        if args:
            return f"list[{get_type_name(args[0])}]"
        return "list"

    if origin is dict:
        return "dict"

    return getattr(typ, "__name__", str(typ))
