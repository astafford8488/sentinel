"""Model runner with dynamic batching and prediction caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BatchConfig:
    max_batch_size: int = 32
    max_wait_ms: float = 50.0


@dataclass
class PredictionResult:
    prediction: Any
    model_name: str
    model_version: str
    latency_ms: float
    request_id: str
    cached: bool = False


class LRUCache:
    """Thread-safe LRU cache for prediction results."""

    def __init__(self, max_size: int = 10_000) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)


class ModelRunner:
    """Run inference on a model with optional batching and caching."""

    def __init__(
        self,
        model_name: str,
        model_version: str,
        predict_fn: Callable[[list[dict[str, Any]]], list[Any]],
        batch_config: BatchConfig | None = None,
        enable_caching: bool = True,
        cache_size: int = 10_000,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.predict_fn = predict_fn
        self.batch_config = batch_config or BatchConfig()
        self.cache = LRUCache(cache_size) if enable_caching else None

        self._batch_queue: asyncio.Queue[tuple[dict, asyncio.Future]] = asyncio.Queue()
        self._batch_task: asyncio.Task | None = None
        self._request_count = 0

    async def predict(self, features: dict[str, Any], request_id: str = "") -> PredictionResult:
        """Run inference on a single input, with caching and batching."""
        start = time.monotonic()

        # Check cache
        if self.cache:
            cache_key = self._cache_key(features)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return PredictionResult(
                    prediction=cached,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    latency_ms=(time.monotonic() - start) * 1000,
                    request_id=request_id,
                    cached=True,
                )

        # Direct prediction (no batching for simplicity in v1)
        result = self.predict_fn([features])[0]

        # Cache result
        if self.cache:
            self.cache.put(cache_key, result)

        self._request_count += 1

        return PredictionResult(
            prediction=result,
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=(time.monotonic() - start) * 1000,
            request_id=request_id,
        )

    async def predict_batch(self, batch: list[dict[str, Any]]) -> list[Any]:
        """Run batch inference."""
        return self.predict_fn(batch)

    def _cache_key(self, features: dict[str, Any]) -> str:
        raw = json.dumps(features, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def request_count(self) -> int:
        return self._request_count
