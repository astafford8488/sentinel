"""Real-time metrics collection for inference monitoring."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelMetrics:
    requests_total: int = 0
    errors_total: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    predictions: list[float] = field(default_factory=list)
    _window_start: float = field(default_factory=time.monotonic)

    @property
    def requests_per_second(self) -> float:
        elapsed = time.monotonic() - self._window_start
        return self.requests_total / max(elapsed, 1)

    @property
    def error_rate(self) -> float:
        return self.errors_total / max(self.requests_total, 1)

    @property
    def p50_latency_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_latency_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_latency_ms(self) -> float:
        return self._percentile(99)

    @property
    def prediction_mean(self) -> float:
        return sum(self.predictions) / len(self.predictions) if self.predictions else 0.0

    @property
    def prediction_std(self) -> float:
        if len(self.predictions) < 2:
            return 0.0
        mean = self.prediction_mean
        variance = sum((x - mean) ** 2 for x in self.predictions) / (len(self.predictions) - 1)
        return variance ** 0.5

    def _percentile(self, p: int) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests_total": self.requests_total,
            "requests_per_second": round(self.requests_per_second, 2),
            "errors_total": self.errors_total,
            "error_rate": round(self.error_rate, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "prediction_mean": round(self.prediction_mean, 4),
            "prediction_std": round(self.prediction_std, 4),
        }


class MetricsCollector:
    """Collect and aggregate inference metrics per model."""

    def __init__(self, max_history: int = 100_000) -> None:
        self._metrics: defaultdict[str, ModelMetrics] = defaultdict(ModelMetrics)
        self._max_history = max_history

    def record_request(
        self,
        model_name: str,
        latency_ms: float,
        prediction: float | None = None,
    ) -> None:
        m = self._metrics[model_name]
        m.requests_total += 1
        m.latencies_ms.append(latency_ms)

        if prediction is not None:
            m.predictions.append(prediction)

        # Trim history
        if len(m.latencies_ms) > self._max_history:
            m.latencies_ms = m.latencies_ms[-self._max_history:]
        if len(m.predictions) > self._max_history:
            m.predictions = m.predictions[-self._max_history:]

    def record_error(self, model_name: str, error: str) -> None:
        self._metrics[model_name].errors_total += 1

    def get_summary(self, model_name: str) -> dict[str, Any]:
        return self._metrics[model_name].to_dict()

    def get_global_summary(self) -> dict[str, Any]:
        total_requests = sum(m.requests_total for m in self._metrics.values())
        total_errors = sum(m.errors_total for m in self._metrics.values())
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "models_tracked": len(self._metrics),
        }

    def get_predictions(self, model_name: str, last_n: int = 1000) -> list[float]:
        return self._metrics[model_name].predictions[-last_n:]

    def reset(self, model_name: str | None = None) -> None:
        if model_name:
            self._metrics.pop(model_name, None)
        else:
            self._metrics.clear()
