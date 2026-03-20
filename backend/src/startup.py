import os

from pydantic import BaseModel, Field

from .containers import RootContainer
from .modules.shared.infrastructure.discovery import (
    discover_modules,
)


class BackendConfig(BaseModel):
    """
    Centralized configuration for backend infrastructure.
    """

    redis_url: str = Field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    data_storage_path: str = Field(
        default_factory=lambda: os.getenv("DATA_STORAGE_PATH", "/app/storage")
    )
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true"
    )


class BackendStartUp:
    """
    Handles backend-wide initialization, module discovery, and DI wiring.
    Dependency Inversion: The caller (API or Worker) provides the BackendConfig.
    """

    def __init__(self, config: BackendConfig):
        self.container = RootContainer()
        self.container.config.from_dict(config.model_dump())

        # Discover all API routers and tasks for recursive wiring
        all_routers = discover_modules("src.api.routers.v1")
        all_tasks = discover_modules("src.modules.evaluation.infrastructure")
        all_wires = all_routers + [m for m in all_tasks if m.endswith(".tasks")]

        # Wire the RootContainer and all its composed containers
        self.container.wire(modules=all_wires)

    async def start(self):
        """Async infrastructure startup (e.g. connections)."""
        redis_conn = self.container.redis_connection()
        await redis_conn.connect()

    async def stop(self):
        """Gracefully shuts down all module-level resources."""
        redis_conn = self.container.redis_connection()
        await redis_conn.close()
        self.container.shutdown_resources()

    def wire_worker(self, ctx: dict):
        """
        Maps worker task identifiers to their required handlers in ARQ context.
        """
        ctx["run_diagnostics_service"] = (
            self.container.evaluation.run_diagnostics_service()
        )
        ctx["task_manager"] = self.container.task_manager()
