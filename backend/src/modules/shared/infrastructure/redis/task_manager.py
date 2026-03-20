import json
from typing import Any, AsyncGenerator

from ...domain.interfaces.base_task_manager import BaseTaskManager
from ..serialization.json_encoder import serialize_diagnostics
from .connection import RedisConnection


class RedisTaskManager(BaseTaskManager):
    """
    Unified Redis adapter for all task lifecycle operations.
    Handles enqueuing (via ARQ), status storage (via Redis keys), 
    and progress publishing/streaming (via Redis Pub/Sub).
    """

    def __init__(self, connection: RedisConnection):
        self._conn = connection

    # --- Task Enqueuing ---
    async def enqueue(self, function_name: str, **kwargs: Any) -> Any:
        """
        Enqueues a job for the ARQ worker.
        """
        return await self._conn.enqueue_job(function_name, **kwargs)

    # --- Status Persistence ---
    async def set_status(
        self, task_id: str, status_data: dict[str, Any], ttl: int = 3600
    ) -> None:
        """
        Stores the current status of a task as a Redis key with TTL.
        Format: task:{task_id}:status
        """
        key = f"task:{task_id}:status"
        value = serialize_diagnostics(status_data)
        await self._conn.client.set(key, value, ex=ttl)

    async def get_status(self, task_id: str) -> dict[str, Any] | None:
        """
        Retrieves task status from Redis.
        """
        key = f"task:{task_id}:status"
        raw_data = await self._conn.client.get(key)
        if not raw_data:
            return None
        return json.loads(raw_data)

    # --- Real-time Progress ---
    async def publish(self, task_id: str, payload: dict[str, Any]) -> None:
        """
        Publishes a real-time progress update to the task-specific channel.
        Format: task:{task_id}
        """
        channel = f"task:{task_id}"
        message = serialize_diagnostics(payload)
        await self._conn.client.publish(channel, message)

    async def subscribe(self, task_id: str) -> AsyncGenerator[str, None]:
        """
        Subscribes to the task channel and yields SSE-formatted strings.
        """
        pubsub = self._conn.client.pubsub()
        channel = f"task:{task_id}"
        await pubsub.subscribe(channel)

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    # Extract the payload
                    data_raw = message["data"]
                    data_str = (
                        data_raw.decode("utf-8")
                        if isinstance(data_raw, bytes)
                        else str(data_raw)
                    )

                    # Format for SSE
                    yield f"data: {data_str}\n\n"

                    # Termination logic: if event is 'done' or 'error', stop the generator
                    try:
                        data = json.loads(data_str)
                        if data.get("event") in ["done", "error"]:
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
