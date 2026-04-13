"""Tests for auto-scaling."""

from sentinel.monitoring.metrics import MetricsCollector
from sentinel.scaling.autoscaler import AutoScaler, ScalingPolicy, ScalingMetric


class TestAutoScaler:
    def test_scale_up_on_high_latency(self):
        metrics = MetricsCollector()
        for _ in range(100):
            metrics.record_request("model-a", latency_ms=150.0)

        policy = ScalingPolicy(
            metric=ScalingMetric.LATENCY_P95,
            scale_up_threshold=100.0,
            min_replicas=1,
            max_replicas=5,
        )
        scaler = AutoScaler(metrics=metrics, policy=policy)
        decision = scaler.evaluate("model-a")

        assert decision.action == "scale_up"
        assert decision.target_replicas == 2

    def test_scale_down_on_low_latency(self):
        metrics = MetricsCollector()
        for _ in range(100):
            metrics.record_request("model-a", latency_ms=5.0)

        policy = ScalingPolicy(
            metric=ScalingMetric.LATENCY_P95,
            scale_down_threshold=20.0,
            min_replicas=1,
            max_replicas=5,
        )
        scaler = AutoScaler(metrics=metrics, policy=policy)
        scaler.set_replicas("model-a", 3)

        decision = scaler.evaluate("model-a")
        assert decision.action == "scale_down"
        assert decision.target_replicas == 2

    def test_no_change_within_range(self):
        metrics = MetricsCollector()
        for _ in range(100):
            metrics.record_request("model-a", latency_ms=50.0)

        policy = ScalingPolicy(
            scale_up_threshold=100.0,
            scale_down_threshold=20.0,
        )
        scaler = AutoScaler(metrics=metrics, policy=policy)
        decision = scaler.evaluate("model-a")

        assert decision.action == "no_change"

    def test_respects_max_replicas(self):
        metrics = MetricsCollector()
        for _ in range(100):
            metrics.record_request("model-a", latency_ms=200.0)

        policy = ScalingPolicy(
            scale_up_threshold=100.0,
            max_replicas=3,
        )
        scaler = AutoScaler(metrics=metrics, policy=policy)
        scaler.set_replicas("model-a", 3)

        decision = scaler.evaluate("model-a")
        assert decision.action == "no_change"
        assert decision.target_replicas == 3
