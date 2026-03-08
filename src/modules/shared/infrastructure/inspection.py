import inspect
from enum import Enum
from typing import Any, Dict, List, Optional, Type, get_args, get_origin

from pydantic import BaseModel, ValidationError


def is_pydantic_model(cls: Any) -> bool:
    return inspect.isclass(cls) and issubclass(cls, BaseModel)


def is_enum(cls: Any) -> bool:
    return inspect.isclass(cls) and issubclass(cls, Enum)


def get_type_name(typ: Any) -> str:
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


def inspect_parameter(
    name: str,
    annotation: Any,
    default: Any = inspect.Parameter.empty,
    description: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Inspects a single parameter and returns one or more parameter definitions.
    Flattens pydantic models.
    """
    if is_pydantic_model(annotation):
        parameters = []
        for field_name, field in annotation.model_fields.items():
            # Recursive call could be done here if nested flattening is desired
            # For now, let's just do one level as requested ("the arg is just a namespace")
            parameters.extend(
                inspect_parameter(
                    name=field_name,
                    annotation=field.annotation,
                    default=field.default
                    if field.default != ...
                    else inspect.Parameter.empty,
                    description=field.description,
                )
            )
        return parameters

    if is_enum(annotation):
        return [
            {
                "name": name,
                "type": "enum",
                "required": default == inspect.Parameter.empty,
                "default": default if default != inspect.Parameter.empty else None,
                "options": [e.value for e in annotation],
                "description": description,
            }
        ]

    return [
        {
            "name": name,
            "type": get_type_name(annotation),
            "required": default == inspect.Parameter.empty,
            "default": default if default != inspect.Parameter.empty else None,
            "description": description,
        }
    ]


def normalize_value(value: Any, annotation: Any) -> Any:
    """
    Normalizes an incoming value (from JSON/API) to the expected Python type.
    """
    if value is None:
        return None

    # Handle Enums
    if is_enum(annotation):
        try:
            return annotation(value)
        except ValueError:
            # Fallback for name-based lookup
            if isinstance(value, str):
                try:
                    return annotation[value.upper()]
                except KeyError:
                    pass
            return value

    # Handle Pydantic Models
    if is_pydantic_model(annotation):
        if isinstance(value, dict):
            return annotation(**value)

    return value


def validate_model_parameters(
    model_class: Type[BaseModel], params: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """
    Validates a dictionary of parameters against a Pydantic model.
    Returns a dictionary of field-level error messages if validation fails, else None.
    """
    try:
        model_class(**params)
        return None
    except ValidationError as e:
        errors = {}
        for error in e.errors():
            # error['loc'] is a tuple of path components
            field_path = ".".join(str(p) for p in error["loc"])
            errors[field_path] = error["msg"]
        return errors


def get_missing_arguments(func, provided_args: Dict[str, Any]) -> List[str]:
    """
    Inspects a function signature to find missing required arguments.
    """
    sig = inspect.signature(func)
    missing = []
    for name, param in sig.parameters.items():
        if name == "self" or name == "cls":
            continue
        if param.default is inspect.Parameter.empty and name not in provided_args:
            missing.append(name)
    return missing
