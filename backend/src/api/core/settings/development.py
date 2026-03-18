from .base import BaseSettings


class DevelopmentSettings(BaseSettings):
    """
    Development environment settings.
    """

    DEBUG: bool = True
    ENV: str = "development"

    # Add any dev-specific overrides here
