from .base import BaseSettings


class TestSettings(BaseSettings):
    """
    Test environment settings.
    """

    DEBUG: bool = True
    ENV: str = "testing"

    # Use a separate storage for tests to avoid data pollution
    DATA_STORAGE_PATH: str = "./storage/test"

    # Add any test-specific overrides here
