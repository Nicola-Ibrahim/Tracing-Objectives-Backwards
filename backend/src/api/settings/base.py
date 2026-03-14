import os
from typing import List

from pydantic import BaseModel, Field


class BaseSettings(BaseModel):
    """
    Shared application settings.
    """

    PROJECT_NAME: str = "Tracing Objectives Backwards API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Environment
    ENV: str = Field(default_factory=lambda: os.getenv("ENV", "development"))
    DEBUG: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true"
    )

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("BACKEND_CORS_ORIGINS", "*").split(",")
        ]
    )

    # Paths
    # src/api/settings/base.py -> src/api/ -> src/ -> project root
    ROOT_DIR: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    DATA_STORAGE_PATH: str = Field(
        default_factory=lambda: os.getenv("DATA_STORAGE_PATH", "./storage")
    )

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    @property
    def is_testing(self) -> bool:
        return self.ENV.lower() == "testing"
