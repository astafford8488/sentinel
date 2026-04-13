"""Tests for A/B testing."""

import pytest

from sentinel.gateway.ab_test import ABTest, ABResults, TrafficSplit


class TestTrafficSplit:
    def test_valid_split(self):
        split = TrafficSplit(control=70, variant=30)
        assert split.control == 70

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError):
            TrafficSplit(control=60, variant=60)


class TestABResults:
    def test_record_and_means(self):
        results = ABResults(name="test")
        for i in range(100):
            results.record(0.5 + i * 0.001, is_control=True)
            results.record(0.6 + i * 0.001, is_control=False)

        assert results.samples_control == 100
        assert results.samples_variant == 100
        assert results.control_mean < results.variant_mean

    def test_significance_not_enough_samples(self):
        results = ABResults(name="test")
        results.record(0.5, is_control=True)
        test = results.significance_test()
        assert test["significant"] is False
        assert "Not enough" in test["recommendation"]

    def test_significance_with_data(self):
        results = ABResults(name="test")
        import random
        random.seed(42)

        for _ in range(500):
            results.record(random.gauss(0.5, 0.1), is_control=True)
            results.record(random.gauss(0.8, 0.1), is_control=False)

        test = results.significance_test()
        assert test["significant"] is True
        assert test["p_value"] < 0.05

    def test_effect_size(self):
        results = ABResults(name="test")
        for _ in range(100):
            results.record(0.5, is_control=True)
            results.record(0.9, is_control=False)

        d = results.effect_size()
        assert d > 0  # Variant is higher


class TestABTest:
    def test_creation(self):
        ab = ABTest(
            name="test",
            control="model:1.0",
            variant="model:2.0",
            split=TrafficSplit(control=90, variant=10),
        )
        assert ab.min_samples == 1000
        assert ab.confidence_level == 0.95
