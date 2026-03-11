from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for FastAPI RequestValidationError to return clean JSON.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Invalid request parameters",
            "errors": exc.errors(),
        },
    )


async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """
    Custom handler for Pydantic ValidationError.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Data validation failed",
            "errors": exc.errors(),
        },
    )
