import inspect
from typing import Any

from .type_utils import _is_discriminator, get_type_name, is_enum, is_pydantic_model


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
            if _is_discriminator(field_name, field.annotation):
                continue

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
