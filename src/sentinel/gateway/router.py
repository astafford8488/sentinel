"""API gateway with multi-model routing, A/B testing, and rate limiting."""

from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from sentinel.gateway.ab_test import ABTest, ABResults
from sentinel.gateway.rate_limiter import RateLimiter
from sentinel.utils.logging import get_logger

if TYPE_CHECKING:
    from sentinel.monitoring.metrics import MetricsCollector
    from sentinel.versioning.registry import ModelRegistry

logger = get_logger(__name__)


class APIGateway:
    """Route inference requests with A/B testing, rate limiting, and version management."""

    def __init__(
        self,
        registry: "ModelRegistry",
        metrics: "MetricsCollector",
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.registry = registry
        self.metrics = metrics
        self.rate_limiter = rate_limiter or RateLimiter()

        self._active_versions: dict[str, str] = {}
        self._ab_tests: dict[str, ABTest] = {}
        self._ab_results: dict[str, ABResults] = {}

    def set_active_version(self, model_name: str, version: str) -> None:
        """Set the active version for a model."""
        self._active_versions[model_name] = version
        logger.info("Active version set", model=model_name, version=version)

    def get_active_version(self, model_name: str) -> str | None:
        """Get the current active version for a model."""
        return self._active_versions.get(model_name)

    def create_ab_test(self, ab_test: ABTest) -> None:
        """Create a new A/B test."""
        self._ab_tests[ab_test.name] = ab_test
        self._ab_results[ab_test.name] = ABResults(name=ab_test.name)
        logger.info(
            "A/B test created",
            name=ab_test.name,
            control=ab_test.control,
            variant=ab_test.variant,
            split=f"{ab_test.split.control}/{ab_test.split.variant}",
        )

    def get_ab_results(self, test_name: str) -> ABResults | None:
        return self._ab_results.get(test_name)

    async def route_request(
        self,
        model_name: str,
        features: dict[str, Any],
        version: str | None = None,
        request_id: str = "",
    ) -> dict[str, Any]:
        """Route a prediction request through the gateway."""
        # Rate limiting
        if not self.rate_limiter.allow(request_id):
            raise RuntimeError("Rate limit exceeded")

        # Determine version — A/B test overrides explicit version
        resolved_version = version
        ab_test = self._find_ab_test(model_name)

        if ab_test and not version:
            resolved_version = self._ab_route(ab_test)
        elif not resolved_version:
            resolved_version = self._active_versions.get(model_name)

        if not resolved_version:
            # Fall back to latest
            versions = self.registry.get_versions(model_name)
            if not versions:
                raise KeyError(f"No versions registered for '{model_name}'")
            resolved_version = versions[-1].version

        # Get model and predict
        model_entry = self.registry.get(model_name, resolved_version)
        if model_entry is None:
            raise KeyError(f"Model '{model_name}:{resolved_version}' not found")

        # Execute prediction via the model's predict function
        prediction = model_entry.predict(features)

        # Record A/B result
        if ab_test:
            is_control = resolved_version == ab_test.control.split(":")[-1]
            results = self._ab_results.get(ab_test.name)
            if results:
                results.record(prediction, is_control=is_control)

        return {
            "prediction": prediction,
            "version": resolved_version,
            "cached": False,
        }

    def _find_ab_test(self, model_name: str) -> ABTest | None:
        """Find active A/B test for a model."""
        for test in self._ab_tests.values():
            if model_name in test.control or model_name in test.variant:
                return test
        return None

    def _ab_route(self, ab_test: ABTest) -> str:
        """Route to control or variant based on traffic split."""
        if random.randint(1, 100) <= ab_test.split.control:
            return ab_test.control.split(":")[-1]
        return ab_test.variant.split(":")[-1]
