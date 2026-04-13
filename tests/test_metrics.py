"""Tests for metrics collection."""

from sentinel.monitoring.metrics import MetricsCollector, ModelMetrics


class TestModelMetrics:
    def test_empty_metrics(self):
        m = ModelMetrics()
        assert m.requests_total == 0
        assert m.p50_latency_ms == 0.0
        assert m.error_rate == 0.0

    def test_latency_percentiles(self):
        m = ModelMetrics()
        m.latencies_ms = list(range(1, 101))  # 1 to 100
        m.requests_total = 100
        assert m.p50_latency_ms == 50
        assert m.p95_latency_ms == 95
        assert m.p99_latency_ms == 99

    def test_prediction_stats(self):
        m = ModelMetrics()
        m.predictions = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert abs(m.prediction_mean - 0.3) < 0.01
        assert m.prediction_std > 0


class TestMetricsCollector:
    def test_record_request(self):
        collector = MetricsCollector()
        collector.record_request("model-a", latency_ms=10.5, prediction=0.8)
        summary = collector.get_summary("model-a")
        assert summary["requests_total"] == 1
        assert summary["p50_latency_ms"] == 10.5

    def test_record_error(self):
        collector = MetricsCollector()
        collector.record_request("model-a", latency_ms=5.0)
        collector.record_error("model-a", "timeout")
        summary = collector.get_summary("model-a")
        assert summary["errors_total"] == 1
        assert summary["error_rate"] == 1.0  # 1 error out of 1 request

    def test_global_summary(self):
        collector = MetricsCollector()
        collector.record_request("model-a", 10.0)
        collector.record_request("model-b", 20.0)
        global_summary = collector.get_global_summary()
        assert global_summary["total_requests"] == 2
        assert global_summary["models_tracked"] == 2

    def test_history_trimming(self):
        collector = MetricsCollector(max_history=10)
        for i in range(20):
            collector.record_request("model-a", float(i))
        preds = collector.get_predictions("model-a")
        assert len(preds) == 0  # No predictions recorded

    def test_reset(self):
        collector = MetricsCollector()
        collector.record_request("model-a", 10.0)
        collector.reset("model-a")
        summary = collector.get_summary("model-a")
        assert summary["requests_total"] == 0
