"""Tests for drift detection."""

import numpy as np
import pytest

from sentinel.monitoring.drift import DriftDetector


class TestDriftDetector:
    def test_no_drift_same_distribution(self):
        np.random.seed(42)
        ref = np.random.normal(0.5, 0.1, 1000)
        cur = np.random.normal(0.5, 0.1, 1000)

        detector = DriftDetector(reference_data=ref)
        report = detector.check(cur)

        assert report.psi < 0.2
        assert report.ks_p_value > 0.05
        assert report.is_drifted is False

    def test_significant_drift(self):
        np.random.seed(42)
        ref = np.random.normal(0.3, 0.1, 1000)
        cur = np.random.normal(0.7, 0.2, 1000)

        detector = DriftDetector(reference_data=ref)
        report = detector.check(cur)

        assert report.is_drifted is True
        assert report.ks_p_value < 0.05

    def test_psi_computation(self):
        np.random.seed(42)
        ref = np.random.uniform(0, 1, 5000)
        # Shifted distribution
        cur = np.random.uniform(0.2, 1.2, 5000)

        detector = DriftDetector(reference_data=ref)
        report = detector.check(cur)

        assert report.psi > 0

    def test_no_reference_raises(self):
        detector = DriftDetector()
        with pytest.raises(ValueError):
            detector.check([0.1, 0.2, 0.3])

    def test_details_populated(self):
        ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        cur = np.array([0.2, 0.3, 0.4, 0.5, 0.6])

        detector = DriftDetector(reference_data=ref)
        report = detector.check(cur)

        assert "reference_mean" in report.details
        assert "current_mean" in report.details
        assert report.details["reference_size"] == 5

    def test_multi_feature_drift(self):
        np.random.seed(42)
        ref_features = {
            "amount": np.random.normal(100, 20, 1000).tolist(),
            "hour": np.random.normal(12, 4, 1000).tolist(),
        }
        cur_features = {
            "amount": np.random.normal(100, 20, 1000).tolist(),  # Same
            "hour": np.random.normal(20, 2, 1000).tolist(),      # Drifted
        }

        detector = DriftDetector()
        report = detector.check_features(ref_features, cur_features)

        assert "hour" in report.features_drifted
