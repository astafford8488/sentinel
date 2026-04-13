"""Latency and concurrency-aware auto-scaling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

from sentinel.utils.logging import get_logger

if TYPE_CHECKING:
    from sentinel.monitoring.metrics import MetricsCollector

logger = get_logger(__name__)


class ScalingMetric(str, Enum):
    LATENCY_P95 = "p95_latency_ms"
    LATENCY_P99 = "p99_latency_ms"
    ERROR_RATE = "error_rate"
    RPS = "requests_per_second"


@dataclass
class ScalingPolicy:
    metric: ScalingMetric = ScalingMetric.LATENCY_P95
    scale_up_threshold: float = 100.0     # ms for latency, rate for errors
    scale_down_threshold: float = 20.0
    min_replicas: int = 1
    max_replicas: int = 10
    scale_up_step: int = 1
    scale_down_step: int = 1
    cooldown_seconds: float = 60.0


@dataclass
class ScalingDecision:
    action: str              # "scale_up", "scale_down", "no_change"
    current_replicas: int
    target_replicas: int
    metric_value: float
    reason: str


class AutoScaler:
    """Evaluate scaling decisions based on metrics and policy."""

    def __init__(
        self,
        metrics: "MetricsCollector",
        policy: ScalingPolicy | None = None,
    ) -> None:
        self.metrics = metrics
        self.policy = policy or ScalingPolicy()
        self._current_replicas: dict[str, int] = {}

    def evaluate(self, model_name: str) -> ScalingDecision:
        """Evaluate whether to scale up, down, or hold."""
        summary = self.metrics.get_summary(model_name)
        current = self._current_replicas.get(model_name, self.policy.min_replicas)
        metric_value = summary.get(self.policy.metric.value, 0.0)

        if metric_value > self.policy.scale_up_threshold:
            target = min(current + self.policy.scale_up_step, self.policy.max_replicas)
            if target > current:
                self._current_replicas[model_name] = target
                return ScalingDecision(
                    action="scale_up",
                    current_replicas=current,
                    target_replicas=target,
                    metric_value=metric_value,
                    reason=f"{self.policy.metric.value} ({metric_value:.1f}) > threshold ({self.policy.scale_up_threshold})",
                )

        elif metric_value < self.policy.scale_down_threshold:
            target = max(current - self.policy.scale_down_step, self.policy.min_replicas)
            if target < current:
                self._current_replicas[model_name] = target
                return ScalingDecision(
                    action="scale_down",
                    current_replicas=current,
                    target_replicas=target,
                    metric_value=metric_value,
                    reason=f"{self.policy.metric.value} ({metric_value:.1f}) < threshold ({self.policy.scale_down_threshold})",
                )

        return ScalingDecision(
            action="no_change",
            current_replicas=current,
            target_replicas=current,
            metric_value=metric_value,
            reason="Within acceptable range",
        )

    def set_replicas(self, model_name: str, count: int) -> None:
        self._current_replicas[model_name] = count

    def get_replicas(self, model_name: str) -> int:
        return self._current_replicas.get(model_name, self.policy.min_replicas)
