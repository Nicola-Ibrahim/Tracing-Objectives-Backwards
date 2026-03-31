from .base import BaseSettings


class TestSettings(BaseSettings):
    """
    Test environment settings.
    """

    DEBUG: bool = True
    ENV: str = "testing"
