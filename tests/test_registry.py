"""Tests for model registry."""

import pytest

from sentinel.versioning.registry import ModelRegistry


class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        reg.register("model-a", "1.0.0", "/path/to/model")
        entry = reg.get("model-a", "1.0.0")
        assert entry is not None
        assert entry.version == "1.0.0"

    def test_duplicate_version_raises(self):
        reg = ModelRegistry()
        reg.register("model-a", "1.0.0", "/path")
        with pytest.raises(ValueError):
            reg.register("model-a", "1.0.0", "/path")

    def test_get_latest(self):
        reg = ModelRegistry()
        reg.register("model-a", "1.0.0", "/path")
        reg.register("model-a", "2.0.0", "/path")
        latest = reg.get_latest("model-a")
        assert latest is not None
        assert latest.version == "2.0.0"

    def test_get_versions(self):
        reg = ModelRegistry()
        reg.register("model-a", "1.0.0", "/path")
        reg.register("model-a", "1.1.0", "/path")
        reg.register("model-a", "2.0.0", "/path")
        versions = reg.get_versions("model-a")
        assert len(versions) == 3

    def test_delete_version(self):
        reg = ModelRegistry()
        reg.register("model-a", "1.0.0", "/path")
        assert reg.delete_version("model-a", "1.0.0") is True
        assert reg.get("model-a", "1.0.0") is None

    def test_lineage(self):
        reg = ModelRegistry()
        reg.register("model-a", "1.0.0", "/path/v1", metadata={"accuracy": 0.9})
        reg.register("model-a", "2.0.0", "/path/v2", metadata={"accuracy": 0.95})
        lineage = reg.get_lineage("model-a")
        assert len(lineage) == 2
        assert lineage[0]["metadata"]["accuracy"] == 0.9

    def test_custom_predict_fn(self):
        reg = ModelRegistry()
        entry = reg.register(
            "model-a", "1.0.0", "/path",
            predict_fn=lambda features: features.get("x", 0) * 2,
        )
        assert entry.predict({"x": 5}) == 10

    def test_list_models(self):
        reg = ModelRegistry()
        reg.register("model-a", "1.0.0", "/path")
        reg.register("model-b", "1.0.0", "/path")
        models = reg.list_models()
        assert "model-a" in models
        assert "model-b" in models
