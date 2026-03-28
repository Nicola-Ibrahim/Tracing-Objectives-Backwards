from .base import BaseSettings


class ProductionSettings(BaseSettings):
    """
    Production environment settings.
    """

    DEBUG: bool = False
    ENV: str = "production"

    # Add any production-specific overrides here

    # Add any production-specific overrides here
