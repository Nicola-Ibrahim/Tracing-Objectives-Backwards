import os
from typing import List

from pydantic import BaseModel, Field


class BaseSettings(BaseModel):
    """
    Shared application settings.
    """

    PROJECT_NAME: str = "Tracing Objectives Backwards API"
    VERSION: str = "1.0.0"
    
    # API Versioning
    API_VERSION: str = Field(default_factory=lambda: os.getenv("API_VERSION", "v1"))
    
    @property
    def API_V1_STR(self) -> str:
        return f"/api/{self.API_VERSION}"

    # Security
    SECRET_KEY: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY", "temporary-secret-key-for-dev")
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

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

    # Operations
    LOG_LEVEL: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    @property
    def is_testing(self) -> bool:
        return self.ENV.lower() == "testing"
