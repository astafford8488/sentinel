"""FastAPI inference server with multi-model routing."""

from __future__ import annotations

import time
import uuid
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from sentinel.gateway.router import APIGateway
    from sentinel.monitoring.metrics import MetricsCollector
    from sentinel.versioning.registry import ModelRegistry


class PredictRequest(BaseModel):
    features: dict[str, Any]
    model_version: str | None = None


class PredictResponse(BaseModel):
    prediction: Any
    model: str
    version: str
    latency_ms: float
    request_id: str
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    uptime_seconds: float


class InferenceServer:
    """FastAPI inference server with metrics instrumentation."""

    def __init__(
        self,
        registry: "ModelRegistry",
        gateway: "APIGateway",
        metrics: "MetricsCollector",
    ) -> None:
        self.registry = registry
        self.gateway = gateway
        self.metrics = metrics
        self._start_time = time.monotonic()

    def create_app(self) -> Any:
        from fastapi import FastAPI, HTTPException

        app = FastAPI(title="SENTINEL Inference API", version="0.1.0")

        @app.get("/health", response_model=HealthResponse)
        async def health() -> HealthResponse:
            return HealthResponse(
                status="healthy",
                models_loaded=len(self.registry.list_models()),
                uptime_seconds=time.monotonic() - self._start_time,
            )

        @app.post("/predict/{model_name}", response_model=PredictResponse)
        async def predict(model_name: str, request: PredictRequest) -> PredictResponse:
            request_id = f"req_{uuid.uuid4().hex[:8]}"
            start = time.monotonic()

            # Route through gateway (handles A/B, rate limiting, etc.)
            try:
                result = await self.gateway.route_request(
                    model_name=model_name,
                    features=request.features,
                    version=request.model_version,
                    request_id=request_id,
                )
            except KeyError:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
            except Exception as e:
                self.metrics.record_error(model_name, str(e))
                raise HTTPException(status_code=500, detail=str(e))

            latency = (time.monotonic() - start) * 1000
            self.metrics.record_request(model_name, latency)

            return PredictResponse(
                prediction=result["prediction"],
                model=model_name,
                version=result.get("version", "unknown"),
                latency_ms=round(latency, 2),
                request_id=request_id,
                cached=result.get("cached", False),
            )

        @app.get("/models")
        async def list_models() -> dict[str, Any]:
            models = self.registry.list_models()
            return {
                name: {
                    "versions": [v.version for v in versions],
                    "active": self.gateway.get_active_version(name),
                }
                for name, versions in models.items()
            }

        @app.get("/metrics/{model_name}")
        async def get_metrics(model_name: str) -> dict[str, Any]:
            return self.metrics.get_summary(model_name)

        return app

    def run(self, port: int = 8000, workers: int = 1) -> None:
        import uvicorn
        app = self.create_app()
        uvicorn.run(app, host="0.0.0.0", port=port, workers=workers)
