from typing import Any

import arq
from arq.connections import RedisSettings

import redis.asyncio as redis


class RedisConnection:
    """
    Async Redis connection manager for Pub/Sub, caching, and ARQ pools.
    Stripped of business logic, focuses purely on connection lifecycle.
    """

    def __init__(self, url: str):
        self._url = url
        self._client: redis.Redis | None = None
        self._arq_pool: arq.ArqRedis | None = None

    async def connect(self) -> None:
        """
        Initializes connection pools. 
        Should be called during application or worker startup.
        """
        if not self._client:
            self._client = redis.from_url(self._url, decode_responses=False)

        if not self._arq_pool:
            self._arq_pool = await arq.create_pool(RedisSettings.from_dsn(self._url))

    async def close(self) -> None:
        """
        Gracefully closes all connection pools.
        """
        if self._arq_pool:
            await self._arq_pool.close()
            self._arq_pool = None

        if self._client:
            await self._client.aclose()  # type: ignore
            self._client = None

    @property
    def client(self) -> redis.Redis:
        """
        Returns the raw Redis client for Pub/Sub or low-level operations.
        """
        if self._client is None:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._client

    @property
    def arq_pool(self) -> arq.ArqRedis:
        """
        Returns the ARQ pool for worker job management.
        """
        if self._arq_pool is None:
            raise RuntimeError("ARQ pool not connected. Call connect() first.")
        return self._arq_pool

    async def enqueue_job(self, function_name: str, **kwargs: Any) -> Any:
        """
        Directly enqueues a job into the ARQ queue using the shared pool.
        """
        return await self.arq_pool.enqueue_job(function_name, **kwargs)
