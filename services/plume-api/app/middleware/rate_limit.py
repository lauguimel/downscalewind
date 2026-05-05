"""In-memory token bucket rate limiter keyed by client IP.

Two limits: a daily quota (resets every 24 h) and a short-term burst limit
(tokens refill linearly). Good enough for a prototype; swap for Redis later.
"""

from __future__ import annotations

import threading
import time

from fastapi import HTTPException, Request

from ..config import settings


class _Bucket:
    __slots__ = ("day_count", "day_reset_ts", "tokens", "last_refill")

    def __init__(self) -> None:
        now = time.time()
        self.day_count = 0
        self.day_reset_ts = now + 86400
        self.tokens = float(settings.rate_limit_burst_per_min)
        self.last_refill = now


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, client_id: str) -> None:
        now = time.time()
        with self._lock:
            b = self._buckets.get(client_id)
            if b is None:
                b = _Bucket()
                self._buckets[client_id] = b

            # Reset daily counter
            if now >= b.day_reset_ts:
                b.day_count = 0
                b.day_reset_ts = now + 86400

            if b.day_count >= settings.rate_limit_per_day:
                retry = int(b.day_reset_ts - now)
                raise HTTPException(
                    status_code=429,
                    detail=f"daily quota exhausted ({settings.rate_limit_per_day}/day); retry in {retry}s",
                )

            # Refill burst tokens (1 token per (60 / burst_per_min) seconds)
            refill_rate = settings.rate_limit_burst_per_min / 60.0
            b.tokens = min(
                float(settings.rate_limit_burst_per_min),
                b.tokens + (now - b.last_refill) * refill_rate,
            )
            b.last_refill = now

            if b.tokens < 1.0:
                raise HTTPException(status_code=429, detail="too many requests; slow down")

            b.tokens -= 1.0
            b.day_count += 1


_limiter = RateLimiter()


def rate_limit_dep(request: Request) -> None:
    """FastAPI dependency — enforces rate limits keyed by client host."""
    client = request.client.host if request.client else "unknown"
    _limiter.check(client)
