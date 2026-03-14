import os

from .base import BaseSettings
from .development import DevelopmentSettings
from .production import ProductionSettings
from .test import TestSettings


def get_settings() -> BaseSettings:
    """
    Factory function to return settings based on ENV environment variable.
    """
    env = os.getenv("ENV", "development").lower()

    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestSettings()

    return DevelopmentSettings()


# Instantiate the settings based on the environment
settings = get_settings()
