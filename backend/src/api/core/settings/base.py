import os
from typing import List

from pydantic import BaseModel, Field


class BaseSettings(BaseModel):
    """
    Shared application settings.
    """

    PROJECT_NAME: str = "Tracing Objectives Backwards API"
    VERSION: str = "1.0.0"

    # Security
    SECRET_KEY: str = Field(default_factory=lambda: os.getenv("SECRET_KEY"))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # Environment
    ENV: str = Field(default_factory=lambda: os.getenv("ENV"))
    DEBUG: bool = Field(
        default_factory=lambda: (os.getenv("DEBUG") or "false").lower() == "true"
    )

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in (os.getenv("BACKEND_CORS_ORIGINS") or "").split(",")
            if origin.strip()
        ]
    )

    # Paths
    # src/api/settings/base.py -> src/api/ -> src/ -> project root
    ROOT_DIR: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    # Operations
    LOG_LEVEL: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL"))

    # Redis
    REDIS_URL: str = Field(default_factory=lambda: os.getenv("REDIS_URL"))
