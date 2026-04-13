"""A/B testing with statistical significance testing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from scipy import stats


@dataclass
class TrafficSplit:
    control: int = 80
    variant: int = 20

    def __post_init__(self) -> None:
        if self.control + self.variant != 100:
            raise ValueError("Traffic split must sum to 100")


@dataclass
class ABTest:
    name: str
    control: str          # "model:version"
    variant: str          # "model:version"
    split: TrafficSplit = field(default_factory=TrafficSplit)
    min_samples: int = 1000
    confidence_level: float = 0.95


@dataclass
class ABResults:
    name: str
    control_predictions: list[float] = field(default_factory=list)
    variant_predictions: list[float] = field(default_factory=list)

    def record(self, prediction: Any, is_control: bool) -> None:
        value = float(prediction) if isinstance(prediction, (int, float)) else 0.0
        if is_control:
            self.control_predictions.append(value)
        else:
            self.variant_predictions.append(value)

    @property
    def samples_control(self) -> int:
        return len(self.control_predictions)

    @property
    def samples_variant(self) -> int:
        return len(self.variant_predictions)

    @property
    def control_mean(self) -> float:
        return sum(self.control_predictions) / len(self.control_predictions) if self.control_predictions else 0.0

    @property
    def variant_mean(self) -> float:
        return sum(self.variant_predictions) / len(self.variant_predictions) if self.variant_predictions else 0.0

    def significance_test(self, confidence: float = 0.95) -> dict[str, Any]:
        """Run Welch's t-test for statistical significance."""
        if len(self.control_predictions) < 2 or len(self.variant_predictions) < 2:
            return {
                "p_value": 1.0,
                "significant": False,
                "recommendation": "Not enough samples",
            }

        t_stat, p_value = stats.ttest_ind(
            self.control_predictions,
            self.variant_predictions,
            equal_var=False,  # Welch's t-test
        )

        alpha = 1 - confidence
        significant = p_value < alpha

        if not significant:
            recommendation = "Continue experiment — not yet significant"
        elif self.variant_mean > self.control_mean:
            recommendation = "Variant wins — promote to production"
        else:
            recommendation = "Control wins — keep current version"

        return {
            "control_mean": round(self.control_mean, 4),
            "variant_mean": round(self.variant_mean, 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
            "significant": significant,
            "samples_control": self.samples_control,
            "samples_variant": self.samples_variant,
            "recommendation": recommendation,
        }

    def effect_size(self) -> float:
        """Cohen's d effect size."""
        if len(self.control_predictions) < 2 or len(self.variant_predictions) < 2:
            return 0.0

        c_std = _std(self.control_predictions)
        v_std = _std(self.variant_predictions)
        pooled_std = math.sqrt((c_std**2 + v_std**2) / 2)

        if pooled_std == 0:
            return 0.0

        return (self.variant_mean - self.control_mean) / pooled_std


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)
