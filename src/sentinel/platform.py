"""Main SENTINEL orchestrator — ties serving, monitoring, and versioning together."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sentinel.serving.server import InferenceServer
from sentinel.gateway.router import APIGateway
from sentinel.monitoring.metrics import MetricsCollector
from sentinel.versioning.registry import ModelRegistry
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    name: str
    version: str
    model_path: str
    framework: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)


class Sentinel:
    """Production ML inference and observability platform."""

    def __init__(self) -> None:
        self.registry = ModelRegistry()
        self.metrics = MetricsCollector()
        self.gateway = APIGateway(registry=self.registry, metrics=self.metrics)
        self._server: InferenceServer | None = None

    def register(self, config: ModelConfig) -> None:
        """Register a model version."""
        self.registry.register(
            name=config.name,
            version=config.version,
            model_path=config.model_path,
            metadata={**config.metadata, "framework": config.framework},
        )
        logger.info("Model registered", name=config.name, version=config.version)

    def serve(
        self,
        models: list[str] | None = None,
        port: int = 8000,
        workers: int = 1,
    ) -> None:
        """Start the inference server."""
        self._server = InferenceServer(
            registry=self.registry,
            gateway=self.gateway,
            metrics=self.metrics,
        )
        logger.info("Starting SENTINEL", port=port, models=models)
        self._server.run(port=port, workers=workers)

    def rollback(self, model_name: str, to_version: str) -> None:
        """Roll back a model to a previous version."""
        self.gateway.set_active_version(model_name, to_version)
        logger.info("Rolled back", model=model_name, version=to_version)

    def status(self) -> dict[str, Any]:
        """Get platform status."""
        models = self.registry.list_models()
        return {
            "models_registered": len(models),
            "models": {
                name: {
                    "versions": [v.version for v in versions],
                    "active": self.gateway.get_active_version(name),
                }
                for name, versions in models.items()
            },
            "metrics_summary": self.metrics.get_global_summary(),
        }
