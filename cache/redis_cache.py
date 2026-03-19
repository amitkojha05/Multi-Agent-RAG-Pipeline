"""cache/redis_cache.py

RedisCache wraps redis.asyncio for async cache operations.
"""
import hashlib
import json
import os
from typing import Any, Optional

import redis.asyncio as aioredis
import structlog

log = structlog.get_logger()


class RedisCache:
    def __init__(self) -> None:
        self._redis: Optional[aioredis.Redis] = None
        self.ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        self.channel = os.getenv("CACHE_INVALIDATE_CHANNEL", "cache:invalidate")

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(
            f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}",
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD") or None,
            decode_responses=True,
        )

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()

    def _key(self, query: str) -> str:
        return "scirag:" + hashlib.sha256(query.strip().lower().encode()).hexdigest()[
            :32
        ]

    async def get(self, query: str) -> Optional[dict]:
        if not self._redis:
            raise RuntimeError("RedisCache not connected")
        value = await self._redis.get(self._key(query))
        return json.loads(value) if value else None

    async def set(self, query: str, data: Any) -> None:
        if not self._redis:
            raise RuntimeError("RedisCache not connected")
        await self._redis.setex(
            self._key(query),
            self.ttl,
            json.dumps(data, default=str),
        )

    async def delete(self, query: str) -> None:
        if not self._redis:
            raise RuntimeError("RedisCache not connected")
        await self._redis.delete(self._key(query))

    async def flush_all(self) -> None:
        if not self._redis:
            raise RuntimeError("RedisCache not connected")
        keys = await self._redis.keys("scirag:*")
        if keys:
            await self._redis.delete(*keys)
        log.info("cache_flushed", keys_deleted=len(keys))

    async def publish_invalidation(self, source_id: str) -> None:
        if not self._redis:
            raise RuntimeError("RedisCache not connected")
        await self._redis.publish(self.channel, source_id)

    async def subscribe_invalidation(self) -> None:
        if not self._redis:
            raise RuntimeError("RedisCache not connected")
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self.channel)
        log.info("cache_invalidation_subscriber_started")
        async for message in pubsub.listen():
            if message["type"] == "message":
                source_id = message["data"]
                log.info("cache_invalidation_triggered", source=source_id)
                await self.flush_all()

