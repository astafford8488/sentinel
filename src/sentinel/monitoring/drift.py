"""Statistical drift detection: PSI, KS test, Jensen-Shannon divergence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class DriftReport:
    psi: float = 0.0
    ks_statistic: float = 0.0
    ks_p_value: float = 1.0
    js_divergence: float = 0.0
    is_drifted: bool = False
    features_drifted: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """Detect distribution drift using multiple statistical tests."""

    def __init__(
        self,
        reference_data: np.ndarray | list[float] | None = None,
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.05,
        js_threshold: float = 0.1,
        n_bins: int = 10,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.js_threshold = js_threshold
        self.n_bins = n_bins

        self.reference: np.ndarray | None = None
        if reference_data is not None:
            self.set_reference(reference_data)

    def set_reference(self, data: np.ndarray | list[float]) -> None:
        """Set the reference distribution (typically from training data)."""
        self.reference = np.array(data, dtype=np.float64)

    def check(
        self,
        current_data: np.ndarray | list[float],
        feature_name: str = "prediction",
    ) -> DriftReport:
        """Run all drift tests against current data."""
        if self.reference is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        current = np.array(current_data, dtype=np.float64)

        # PSI
        psi = self._compute_psi(self.reference, current)

        # KS Test
        ks_stat, ks_p = scipy_stats.ks_2samp(self.reference, current)

        # Jensen-Shannon Divergence
        js = self._compute_js_divergence(self.reference, current)

        # Determine if drifted
        is_drifted = (
            psi > self.psi_threshold
            or ks_p < self.ks_alpha
            or js > self.js_threshold
        )

        features_drifted = []
        if is_drifted:
            features_drifted.append(feature_name)

        return DriftReport(
            psi=round(float(psi), 4),
            ks_statistic=round(float(ks_stat), 4),
            ks_p_value=round(float(ks_p), 4),
            js_divergence=round(float(js), 4),
            is_drifted=is_drifted,
            features_drifted=features_drifted,
            details={
                "reference_size": len(self.reference),
                "current_size": len(current),
                "reference_mean": round(float(np.mean(self.reference)), 4),
                "current_mean": round(float(np.mean(current)), 4),
                "reference_std": round(float(np.std(self.reference)), 4),
                "current_std": round(float(np.std(current)), 4),
            },
        )

    def check_features(
        self,
        reference_features: dict[str, list[float]],
        current_features: dict[str, list[float]],
    ) -> DriftReport:
        """Check drift across multiple features."""
        all_drifted: list[str] = []
        max_psi = 0.0
        max_js = 0.0

        for feature_name in reference_features:
            if feature_name not in current_features:
                continue

            ref = np.array(reference_features[feature_name])
            cur = np.array(current_features[feature_name])

            self.set_reference(ref)
            report = self.check(cur, feature_name=feature_name)

            if report.is_drifted:
                all_drifted.append(feature_name)
            max_psi = max(max_psi, report.psi)
            max_js = max(max_js, report.js_divergence)

        return DriftReport(
            psi=max_psi,
            js_divergence=max_js,
            is_drifted=len(all_drifted) > 0,
            features_drifted=all_drifted,
        )

    def _compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Population Stability Index."""
        # Create bins from reference distribution
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, self.n_bins + 1)

        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        # Normalize to proportions
        ref_pct = ref_hist / max(ref_hist.sum(), 1) + 1e-10
        cur_pct = cur_hist / max(cur_hist.sum(), 1) + 1e-10

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def _compute_js_divergence(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Jensen-Shannon Divergence."""
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins = np.linspace(min_val, max_val, self.n_bins + 1)

        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)

        # Normalize
        ref_hist = ref_hist / max(ref_hist.sum(), 1e-10)
        cur_hist = cur_hist / max(cur_hist.sum(), 1e-10)

        # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
        m = 0.5 * (ref_hist + cur_hist)
        m = np.maximum(m, 1e-10)

        kl_pm = np.sum(ref_hist * np.log(np.maximum(ref_hist, 1e-10) / m))
        kl_qm = np.sum(cur_hist * np.log(np.maximum(cur_hist, 1e-10) / m))

        return float(0.5 * kl_pm + 0.5 * kl_qm)
