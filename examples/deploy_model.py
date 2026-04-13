"""Example: Register and serve a model with monitoring."""

from sentinel import Sentinel, ModelConfig
from sentinel.monitoring import AlertManager, AlertRule, AlertSeverity


def main() -> None:
    platform = Sentinel()

    # Register model with custom predict function
    config = ModelConfig(
        name="fraud-detector",
        version="1.2.0",
        model_path="./models/fraud_v1.2",
        framework="xgboost",
        metadata={"accuracy": 0.94, "trained_on": "2025-03-15"},
    )
    platform.register(config)

    # Set up alerting
    alerts = AlertManager()
    alerts.add_rule(AlertRule(
        name="high-latency",
        model_name="fraud-detector",
        metric="p95_latency_ms",
        threshold=100.0,
        severity=AlertSeverity.WARNING,
    ))
    alerts.add_rule(AlertRule(
        name="high-error-rate",
        model_name="fraud-detector",
        metric="error_rate",
        threshold=0.05,
        severity=AlertSeverity.CRITICAL,
    ))

    # Serve
    platform.serve(models=["fraud-detector:1.2.0"], port=8000)


if __name__ == "__main__":
    main()
