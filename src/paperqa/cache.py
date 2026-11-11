"""可关闭的 Redis 候选缓存，索引仍可独立查询。"""

import asyncio
import json
import math
from time import monotonic

from paperqa.utils import stable_id


class RedisCandidateCache:
    def __init__(self, url, ttl=300):
        from redis.asyncio import Redis
        from redis.backoff import NoBackoff
        from redis.exceptions import RedisError
        from redis.retry import Retry

        self.client = Redis.from_url(
            url, socket_timeout=0.15, socket_connect_timeout=0.15, retry=Retry(NoBackoff(), 0)
        )
        self.error = RedisError
        self.ttl = ttl
        self._failures = 0
        self._retry_at = 0.0
        self._probing = False
        self._state_version = 0

    def _key(self, key):
        return "pqa:candidates:v1:" + stable_id(json.dumps(key, ensure_ascii=False))

    async def _redis_call(self, operation, *args, **kwargs):
        """连续 3 次故障后跳过 Redis 30 秒，冷却后只放行一次探测。"""
        if self._probing or monotonic() < self._retry_at:
            return None
        # 隔离熔断前的在途请求，避免迟到成功误关熔断。
        version = self._state_version
        probe = self._retry_at > 0
        if probe:
            self._probing = True
        try:
            async with asyncio.timeout(0.2):
                result = await operation(*args, **kwargs)
        except (self.error, TimeoutError):
            if version == self._state_version:
                self._failures += 1
                if self._failures >= 3:
                    self._retry_at = monotonic() + 30
                    self._state_version += 1
            return None
        else:
            if version == self._state_version:
                self._failures = 0
                self._retry_at = 0.0
            return result
        finally:
            if probe:
                self._probing = False

    async def get(self, key, index, scope):
        try:
            raw = await self._redis_call(self.client.get, self._key(key))
            if raw is None:
                return None
            rows = json.loads(raw)
            if len(rows) > key[-1] * 6:
                return None
            ranked = [(index.texts[index.order[chunk]], float(score)) for chunk, score in rows]
            if any(t.doc.dockey not in scope or not math.isfinite(score) for t, score in ranked):
                return None
            return ranked
        except (KeyError, TypeError, ValueError):
            return None

    async def put(self, key, ranked):
        await self._redis_call(
            self.client.set,
            self._key(key), json.dumps([(t.chunk_id, score) for t, score in ranked]),
            ex=self.ttl,
        )

    async def close(self):
        await self.client.aclose()
