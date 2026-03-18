import importlib
import inspect
import pkgutil
from enum import Enum
from typing import Any, Literal, Optional, Type, get_args, get_origin

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


def _inspect_single_param(
    name: str,
    annotation: Any,
    default: Any = inspect.Parameter.empty,
    description: str | None = None,
) -> list[dict[str, Any]]:
    """
    Inspects a single parameter and returns one or more parameter definitions.
    Flattens pydantic models.
    """
    if is_pydantic_model(annotation):
        parameters = []
        for field_name, field in annotation.model_fields.items():
            # Skip discriminator fields (usually named 'type' and typed as Literal)
            # These are internal implementation details for Pydantic and shouldn't be
            # exposed as user-tunable hyperparameters in the frontend.
            if field_name == "type" and get_origin(field.annotation) is Literal:
                continue

            # Recursive call could be done here if nested flattening is desired
            # For now, let's just do one level as requested ("the arg is just a namespace")
            parameters.extend(
                _inspect_single_param(
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


def extract_constructor_schema(cls: type) -> list[dict[str, Any]]:
    """
    Inspects cls.__init__ and returns a flat list of parameter definitions
    for frontend discovery. Flattens Pydantic model params into their fields.
    Skips Literal-typed 'type' discriminator fields.
    """
    sig = inspect.signature(cls.__init__)
    params = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        params.extend(
            _inspect_single_param(
                name=name,
                annotation=param.annotation,
                default=param.default,
            )
        )
    return params


def build_constructor_kwargs(cls: type, config: dict[str, Any]) -> dict[str, Any]:
    """
    Given a class and a flat config dict (from frontend/API), builds the
    correct kwargs for cls.__init__. Handles:
    - Pydantic model params: extracts matching fields, uses model_validate()
    - Enum params: normalizes string -> Enum
    - Simple params: passes through
    - Defaults: uses defaults for missing optional params
    Skips Literal-typed 'type' discriminator fields from config extraction.
    """
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        if is_pydantic_model(param.annotation):
            # Check if already provided as a nested dict under the param name
            if name in config and isinstance(config[name], dict):
                kwargs[name] = param.annotation.model_validate(config[name])
            else:
                # Extract matching fields from flat config
                model_fields = param.annotation.model_fields
                filtered = {
                    k: v
                    for k, v in config.items()
                    if k in model_fields
                    and not (
                        k == "type"
                        and get_origin(model_fields[k].annotation) is Literal
                    )
                }
                # If we found fields or if it's required, try to instantiate
                if filtered or param.default is inspect.Parameter.empty:
                    kwargs[name] = param.annotation.model_validate(filtered)
        else:
            if name in config:
                kwargs[name] = normalize_value(config[name], param.annotation)
            elif param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default
    return kwargs


def inspect_parameter(
    name: str,
    annotation: Any,
    default: Any = inspect.Parameter.empty,
    description: str | None = None,
) -> list[dict[str, Any]]:
    """Deprecated: Use extract_constructor_schema instead for whole classes."""
    return _inspect_single_param(name, annotation, default, description)


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
    model_class: Type[BaseModel], params: dict[str, Any]
) -> Optional[dict[str, str]]:
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


def get_missing_arguments(func, provided_args: dict[str, Any]) -> list[str]:
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


def discover_modules(package_name: str) -> list[str]:
    """
    Recursively discovers all dot-separated module paths within a package.

    Args:
        package_name (str): The dot-separated package path (e.g., 'src.api.routers.v1').

    Returns:
        list[str]: A list of dot-separated module paths.
    """
    package = importlib.import_module(package_name)
    package_path = package.__path__

    modules = []
    # Walk through all submodules in the package recursively
    for _, name, is_pkg in pkgutil.walk_packages(package_path, f"{package_name}."):
        # We target non-package modules for wiring (e.g., actual Python files)
        if not is_pkg:
            modules.append(name)
    return modules
