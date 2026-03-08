from typing import Any

from fastapi import HTTPException, status


def to_http_response(result: Any) -> Any:
    """
    Standardizes the mapping from Result objects to FastAPI responses.
    Returns the success value or raises an HTTPException.
    """
    if hasattr(result, "is_ok") and result.is_ok:
        return result.value

    if hasattr(result, "is_failure") and result.is_failure:
        error = result.error
        status_code = status.HTTP_400_BAD_REQUEST

        # Mapping domain error codes to HTTP status codes
        if error.code == "NOT_FOUND":
            status_code = status.HTTP_404_NOT_FOUND
        elif error.code == "INTERNAL_ERROR":
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif error.code == "UNAUTHORIZED":
            status_code = status.HTTP_401_UNAUTHORIZED
        elif error.code == "FORBIDDEN":
            status_code = status.HTTP_403_FORBIDDEN
        elif error.code == "VALIDATION_ERROR":
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

        raise HTTPException(
            status_code=status_code,
            detail={
                "message": error.message,
                "error_code": error.code,
                "details": error.details,
            },
        )

    return result
