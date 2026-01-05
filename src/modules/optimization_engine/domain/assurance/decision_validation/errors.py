from ..shared.errors import AssuranceError


class ValidationError(AssuranceError):
    """Raised when validation inputs are malformed or inconsistent."""
