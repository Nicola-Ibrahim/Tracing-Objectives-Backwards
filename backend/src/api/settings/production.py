import os

from .base import BaseSettings


class ProductionSettings(BaseSettings):
    """
    Production environment settings.
    """

    DEBUG: bool = False
    ENV: str = "production"

    # In production, we typically want explicit CORS origins
    # These should be overridden by environment variables

    @property
    def secret_key(self) -> str:
        # Enforce strong secret key in production
        key = os.getenv("SECRET_KEY")
        if not key:
            raise ValueError(
                "SECRET_KEY environment variable is required in production"
            )
        return key

    # Add any production-specific overrides here
