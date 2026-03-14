from typing import Any, Callable, Generic, Self, TypeVar

from pydantic import BaseModel, Field

TResult = TypeVar("TResult")


class ErrorDetails(BaseModel):
    """
    Structured error information for standardized responses.
    """

    message: str = Field(..., description="High-level human-readable error message")
    details: str | dict[str, Any] | None = Field(
        None, description="Field-specific details"
    )
    code: str = Field("INTERNAL_ERROR", description="Machine-parsable error code")


TError = TypeVar("TError", bound=ErrorDetails)


class Result(Generic[TResult]):
    """
    Standardized wrapper for operation outcomes.
    """

    def __init__(
        self,
        is_ok: bool,
        value: TResult | None = None,
        error: TError | None = None,
    ):
        self._is_ok = is_ok
        self._value = value
        self._error = error

    @property
    def is_ok(self) -> bool:
        """Check if the result represents success."""
        return self._error is None

    @property
    def is_failure(self) -> bool:
        """Check if the result represents an error."""
        return self._error is not None

    @property
    def value(self) -> TResult:
        """Get the success value."""
        if self.is_failure:
            raise ValueError("Cannot access value on an error result.")
        return self._value

    @property
    def error(self) -> TError | None:
        if self.is_ok:
            raise ValueError(
                f"Cannot access error of a successful result: {self._value}"
            )
        return self._error

    @classmethod
    def ok(cls, value: TResult) -> Self:
        return cls(is_ok=True, value=value)

    @classmethod
    def fail(
        cls,
        message: str,
        details: dict[str, Any] | None = None,
        code: str = "INTERNAL_ERROR",
    ) -> Self:
        error = ErrorDetails(message=message, details=details, code=code)
        return cls(is_ok=False, error=error)

    def on_success(self, func: Callable[[TResult], Any]) -> Self:
        if self.is_ok:
            return Result.ok(func(self.value))
        return Result.fail(self.error)

    def on_failure(self, func: Callable[[TError], Any]) -> Self:
        if self.is_failure:
            return Result.fail(func(self.error))
        return Result.ok(self.value)

    def match(
        self, on_success: Callable[[TResult], Any], on_failure: Callable[[TError], Any]
    ) -> Any:
        """
        Execute appropriate function based on result type.

        Args:
            on_success (Callable): Function to handle success.
            on_failure (Callable): Function to handle error.

        Returns:
            Any: The return value of the called function.
        """
        if self.is_ok:
            return on_success(self.value)
        return on_failure(self.error)


# Standard error codes
VALIDATION_ERROR = "VALIDATION_ERROR"
NOT_FOUND = "NOT_FOUND"
INTERNAL_ERROR = "INTERNAL_ERROR"
UNAUTHORIZED = "UNAUTHORIZED"
FORBIDDEN = "FORBIDDEN"
