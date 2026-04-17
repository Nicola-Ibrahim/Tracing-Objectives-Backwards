import inspect
from typing import Any

from .type_utils import _is_discriminator, is_enum, is_pydantic_model


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
                    and not _is_discriminator(k, model_fields[k].annotation)
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
