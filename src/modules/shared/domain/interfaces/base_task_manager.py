from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


class BaseTaskManager(ABC):
    """
    Unified port for all background task lifecycle operations.
    Combines enqueuing, status persistence, and real-time progress publishing/streaming.
    """

    # --- Task Enqueuing ---
    @abstractmethod
    async def enqueue(self, function_name: str, **kwargs: Any) -> Any:
        """
        Enqueues a job for execution by the background worker.
        """
        pass

    # --- Status Persistence ---
    @abstractmethod
    async def set_status(
        self, task_id: str, status_data: dict[str, Any], ttl: int = 3600
    ) -> None:
        """
        Stores or updates the current status of a task.
        """
        pass

    @abstractmethod
    async def get_status(self, task_id: str) -> dict[str, Any] | None:
        """
        Retrieves the preserved status of a task.
        """
        pass

    # --- Real-time Progress ---
    @abstractmethod
    async def publish(self, task_id: str, payload: dict[str, Any]) -> None:
        """
        Publishes a real-time progress update to the task-specific channel.
        """
        pass

    @abstractmethod
    async def subscribe(self, task_id: str) -> AsyncGenerator[str, None]:
        """
        Subscribes to a task channel and yields SSE-formatted progress strings.
        """
        pass
