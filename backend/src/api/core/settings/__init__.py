import os

from .base import BaseSettings
from .development import DevelopmentSettings
from .production import ProductionSettings


def get_settings() -> BaseSettings:
    """
    Factory function to return settings based on ENV environment variable.
    """
    env = os.getenv("ENV")

    if not env:
        # If no environment is specified, we stop.
        # This forces explicit configuration via Doppler/Shell.
        raise RuntimeError(
            "The 'ENV' environment variable is not set. "
            "Please set it to 'production' or 'development'."
        )

    env = env.lower()

    if env == "production":
        return ProductionSettings()
    elif env == "development":
        return DevelopmentSettings()

    raise ValueError(
        f"Invalid environment: '{env}'. "
        "Must be 'production' or 'development'."
    )


# Instantiate the settings based on the environment
settings = get_settings()
