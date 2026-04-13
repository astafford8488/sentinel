"""Model registry with versioning, metadata, and lineage tracking."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelEntry:
    name: str
    version: str
    model_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    _predict_fn: Callable[[dict[str, Any]], Any] | None = None

    def predict(self, features: dict[str, Any]) -> Any:
        """Run prediction using the registered model."""
        if self._predict_fn:
            return self._predict_fn(features)

        # Default: load and predict based on framework
        framework = self.metadata.get("framework", "generic")
        return self._generic_predict(features, framework)

    def _generic_predict(self, features: dict[str, Any], framework: str) -> Any:
        """Generic prediction for common frameworks."""
        import importlib
        import json
        from pathlib import Path

        model_path = Path(self.model_path)

        if framework == "sklearn" or (model_path / "model.pkl").exists():
            import pickle
            with open(model_path / "model.pkl", "rb") as f:
                model = pickle.load(f)
            import numpy as np
            X = np.array([list(features.values())])
            return float(model.predict(X)[0])

        if framework == "xgboost" or (model_path / "model.json").exists():
            # XGBoost JSON model
            import numpy as np
            return float(np.random.random())  # Placeholder

        # Fallback: return features hash as mock prediction
        return sum(float(v) for v in features.values() if isinstance(v, (int, float))) / max(len(features), 1)

    def set_predict_fn(self, fn: Callable[[dict[str, Any]], Any]) -> None:
        """Set a custom prediction function."""
        self._predict_fn = fn


class ModelRegistry:
    """Registry for model versions with metadata and lineage."""

    def __init__(self) -> None:
        self._models: defaultdict[str, list[ModelEntry]] = defaultdict(list)

    def register(
        self,
        name: str,
        version: str,
        model_path: str,
        metadata: dict[str, Any] | None = None,
        predict_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> ModelEntry:
        """Register a new model version."""
        # Check for duplicate
        for entry in self._models[name]:
            if entry.version == version:
                raise ValueError(f"Version {version} already registered for {name}")

        entry = ModelEntry(
            name=name,
            version=version,
            model_path=model_path,
            metadata=metadata or {},
        )
        if predict_fn:
            entry.set_predict_fn(predict_fn)

        self._models[name].append(entry)
        logger.info("Registered", model=name, version=version)
        return entry

    def get(self, name: str, version: str) -> ModelEntry | None:
        """Get a specific model version."""
        for entry in self._models.get(name, []):
            if entry.version == version:
                return entry
        return None

    def get_latest(self, name: str) -> ModelEntry | None:
        """Get the latest registered version."""
        versions = self._models.get(name, [])
        return versions[-1] if versions else None

    def get_versions(self, name: str) -> list[ModelEntry]:
        """Get all versions of a model, ordered by registration time."""
        return list(self._models.get(name, []))

    def list_models(self) -> dict[str, list[ModelEntry]]:
        """List all registered models and their versions."""
        return dict(self._models)

    def delete_version(self, name: str, version: str) -> bool:
        """Remove a specific version."""
        entries = self._models.get(name, [])
        for i, entry in enumerate(entries):
            if entry.version == version:
                entries.pop(i)
                logger.info("Deleted", model=name, version=version)
                return True
        return False

    def get_lineage(self, name: str) -> list[dict[str, Any]]:
        """Get version history with metadata."""
        return [
            {
                "version": e.version,
                "registered_at": e.registered_at,
                "metadata": e.metadata,
                "model_path": e.model_path,
            }
            for e in self._models.get(name, [])
        ]
