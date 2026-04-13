"""Alerting system with threshold and anomaly-based rules."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    LOG = "log"
    WEBHOOK = "webhook"
    SLACK = "slack"


@dataclass
class Alert:
    rule_name: str
    model_name: str
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    name: str
    model_name: str
    metric: str                        # "error_rate", "p95_latency_ms", "drift_psi"
    threshold: float
    comparison: str = "gt"             # "gt", "lt", "gte", "lte"
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: float = 300.0    # Min time between alerts for same rule


class AlertManager:
    """Manage alert rules and dispatch notifications."""

    def __init__(self) -> None:
        self._rules: list[AlertRule] = []
        self._handlers: dict[AlertChannel, Callable[[Alert], None]] = {
            AlertChannel.LOG: self._log_handler,
        }
        self._last_fired: dict[str, float] = {}
        self._alert_history: list[Alert] = []

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def add_handler(self, channel: AlertChannel, handler: Callable[[Alert], None]) -> None:
        self._handlers[channel] = handler

    def check(self, model_name: str, metrics: dict[str, float]) -> list[Alert]:
        """Check all rules against current metrics and fire alerts."""
        fired: list[Alert] = []

        for rule in self._rules:
            if rule.model_name != model_name and rule.model_name != "*":
                continue

            value = metrics.get(rule.metric)
            if value is None:
                continue

            if self._should_alert(rule, value):
                if self._in_cooldown(rule.name):
                    continue

                alert = Alert(
                    rule_name=rule.name,
                    model_name=model_name,
                    severity=rule.severity,
                    message=f"{rule.metric} = {value:.4f} {rule.comparison} {rule.threshold}",
                    metadata={"metric": rule.metric, "value": value, "threshold": rule.threshold},
                )

                self._fire(alert)
                fired.append(alert)
                self._last_fired[rule.name] = time.time()

        return fired

    def _should_alert(self, rule: AlertRule, value: float) -> bool:
        match rule.comparison:
            case "gt":
                return value > rule.threshold
            case "lt":
                return value < rule.threshold
            case "gte":
                return value >= rule.threshold
            case "lte":
                return value <= rule.threshold
            case _:
                return False

    def _in_cooldown(self, rule_name: str) -> bool:
        last = self._last_fired.get(rule_name, 0)
        rule = next((r for r in self._rules if r.name == rule_name), None)
        cooldown = rule.cooldown_seconds if rule else 300
        return (time.time() - last) < cooldown

    def _fire(self, alert: Alert) -> None:
        self._alert_history.append(alert)
        for handler in self._handlers.values():
            try:
                handler(alert)
            except Exception as e:
                logger.error("Alert handler failed", error=str(e))

    def _log_handler(self, alert: Alert) -> None:
        logger.warning(
            f"ALERT [{alert.severity.value}] {alert.rule_name}: {alert.message}",
            model=alert.model_name,
        )

    def get_history(self, limit: int = 100) -> list[Alert]:
        return self._alert_history[-limit:]
