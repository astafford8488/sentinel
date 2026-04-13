"""Sliding window rate limiter."""

from __future__ import annotations

import time
from collections import defaultdict


class RateLimiter:
    """Sliding window rate limiter for API gateway."""

    def __init__(
        self,
        requests_per_minute: int = 600,
        burst_size: int = 50,
    ) -> None:
        self.rpm = requests_per_minute
        self.burst = burst_size
        self.window = 60.0  # seconds

        self._requests: defaultdict[str, list[float]] = defaultdict(list)
        self._global_requests: list[float] = []

    def allow(self, client_id: str = "global") -> bool:
        """Check if request is allowed under rate limit."""
        now = time.monotonic()
        self._cleanup(now)

        # Global rate limit
        if len(self._global_requests) >= self.rpm:
            return False

        # Per-client burst limit
        client_recent = self._requests[client_id]
        if len(client_recent) >= self.burst:
            return False

        # Record request
        self._global_requests.append(now)
        self._requests[client_id].append(now)
        return True

    def _cleanup(self, now: float) -> None:
        """Remove expired entries."""
        cutoff = now - self.window
        self._global_requests = [t for t in self._global_requests if t > cutoff]

        expired_clients = []
        for client_id, times in self._requests.items():
            self._requests[client_id] = [t for t in times if t > cutoff]
            if not self._requests[client_id]:
                expired_clients.append(client_id)

        for client_id in expired_clients:
            del self._requests[client_id]

    @property
    def current_usage(self) -> int:
        self._cleanup(time.monotonic())
        return len(self._global_requests)

    def reset(self) -> None:
        self._global_requests.clear()
        self._requests.clear()
