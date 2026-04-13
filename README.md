<div align="center">

# SENTINEL

### Production ML Inference & Observability Platform

[![CI](https://github.com/astafford8488/sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/astafford8488/sentinel/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Ship ML models that stay healthy in production.**

[Quick Start](#quick-start) | [Architecture](#architecture) | [Monitoring](#monitoring) | [A/B Testing](#ab-testing)

</div>

---

## The Problem

Getting a model to work in a notebook is 10% of the job. The other 90% is:
- Serving it behind a reliable API with proper latency
- Knowing when it's degrading before users tell you
- Rolling out new versions without breaking production
- Scaling up at 2am when traffic spikes
- Figuring out *why* predictions changed after a deploy

Most ML teams build this from scratch every time. SENTINEL is the platform so you don't have to.

## What SENTINEL Does

SENTINEL is a **production ML serving and observability platform** that handles everything between "the model works" and "the model works reliably at scale."

| Component | What It Does |
|-----------|-------------|
| **Model Serving** | FastAPI inference gateway with batching, async, and multi-model routing |
| **API Gateway** | Rate limiting, auth, request validation, circuit breakers, and load balancing |
| **Model Registry** | Version management with metadata, lineage tracking, and instant rollback |
| **A/B Testing** | Traffic splitting with statistical significance testing and automatic winner selection |
| **Monitoring** | Real-time latency, throughput, error rate, and prediction drift dashboards |
| **Drift Detection** | Statistical tests (PSI, KS, Jensen-Shannon) on input features and prediction distributions |
| **Auto-scaling** | Concurrency and latency-aware scaling with configurable policies |
| **Alerting** | Threshold and anomaly-based alerts via webhook, Slack, or PagerDuty |

## Quick Start

```bash
git clone https://github.com/astafford8488/sentinel.git
cd sentinel
pip install -e ".[dev]"
```

### Deploy a Model in 10 Lines

```python
from sentinel import Sentinel, ModelConfig

platform = Sentinel()

# Register a model
platform.registry.register(
    name="fraud-detector",
    version="1.2.0",
    model_path="./models/fraud_v1.2",
    metadata={"framework": "xgboost", "accuracy": 0.94, "trained_on": "2025-03-15"},
)

# Serve it
platform.serve(
    models=["fraud-detector:1.2.0"],
    port=8000,
    workers=4,
)
```

### Or Use the CLI

```bash
# Register & serve
sentinel register --name fraud-detector --version 1.2.0 --path ./models/fraud_v1.2
sentinel serve --models fraud-detector:1.2.0 --port 8000

# Monitor
sentinel monitor --dashboard        # Open live dashboard
sentinel status                     # Health check all models

# A/B test
sentinel ab-test create \
    --name "fraud-v1.2-vs-v1.3" \
    --control fraud-detector:1.2.0 \
    --variant fraud-detector:1.3.0 \
    --split 80/20

# Rollback
sentinel rollback fraud-detector --to 1.1.0
```

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              API Gateway                  │
                    │  Rate Limit │ Auth │ Validate │ Circuit   │
                    └──────────────────┬───────────────────────┘
                                       │
                    ┌──────────────────┼───────────────────────┐
                    │           Load Balancer / Router          │
                    │     A/B Split │ Canary │ Blue-Green       │
                    └───────┬───────────────────┬──────────────┘
                            │                   │
                   ┌────────▼────────┐ ┌────────▼────────┐
                   │   Model A       │ │   Model B       │
                   │   (control)     │ │   (variant)     │
                   │                 │ │                 │
                   │  Batch │ Async  │ │  Batch │ Async  │
                   │  Cache │ GPU    │ │  Cache │ GPU    │
                   └────────┬────────┘ └────────┬────────┘
                            │                   │
                    ┌───────▼───────────────────▼──────────────┐
                    │            Observability Layer            │
                    │                                          │
                    │  Metrics │ Drift │ Alerts │ Traces       │
                    │  Latency │ PSI   │ Slack  │ Request Log  │
                    │  Throughput │ KS  │ PD    │ Predictions  │
                    └──────────────────────────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────────┐
                    │           Model Registry                  │
                    │  Versions │ Metadata │ Lineage │ Rollback │
                    └──────────────────────────────────────────┘
```

## Model Serving

SENTINEL's inference server is designed for production workloads:

```python
from sentinel.serving import InferenceServer, ModelRunner, BatchConfig

server = InferenceServer(
    batch_config=BatchConfig(
        max_batch_size=32,       # Dynamic batching
        max_wait_ms=50,          # Max wait for batch fill
    ),
    max_concurrent=100,          # Concurrent request limit
    request_timeout=30.0,        # Per-request timeout
    enable_caching=True,         # LRU prediction cache
    cache_size=10_000,
)
```

### Request Format

```bash
curl -X POST http://localhost:8000/predict/fraud-detector \
  -H "Content-Type: application/json" \
  -d '{"features": {"amount": 500.0, "merchant": "online", "hour": 3}}'

# Response
{
    "prediction": 0.87,
    "model": "fraud-detector",
    "version": "1.2.0",
    "latency_ms": 12.4,
    "request_id": "req_a1b2c3"
}
```

## Monitoring

Every prediction is instrumented. No setup required — monitoring starts automatically.

```python
from sentinel.monitoring import MetricsCollector, DriftDetector

# Access real-time metrics
metrics = platform.metrics
print(metrics.get_summary("fraud-detector"))
# ModelMetrics(
#     requests_total=142857,
#     requests_per_second=47.6,
#     p50_latency_ms=8.2,
#     p95_latency_ms=24.1,
#     p99_latency_ms=89.3,
#     error_rate=0.002,
#     prediction_mean=0.34,
#     prediction_std=0.21,
# )

# Drift detection
drift = DriftDetector(reference_data=training_distribution)
report = drift.check(recent_predictions)
print(report)
# DriftReport(
#     psi=0.08,           # Population Stability Index
#     ks_statistic=0.12,  # Kolmogorov-Smirnov
#     js_divergence=0.04, # Jensen-Shannon
#     is_drifted=False,
#     features_drifted=["hour", "amount"],
# )
```

## A/B Testing

Run controlled experiments with statistical rigor:

```python
from sentinel.gateway import ABTest, TrafficSplit

ab = ABTest(
    name="fraud-v1.2-vs-v1.3",
    control="fraud-detector:1.2.0",
    variant="fraud-detector:1.3.0",
    split=TrafficSplit(control=80, variant=20),
    min_samples=1000,
    confidence_level=0.95,
)

platform.gateway.create_ab_test(ab)

# Check results
results = platform.gateway.get_ab_results("fraud-v1.2-vs-v1.3")
print(results)
# ABResults(
#     control_mean=0.342, variant_mean=0.338,
#     p_value=0.23, significant=False,
#     samples_control=4200, samples_variant=1050,
#     recommendation="Continue experiment — not yet significant"
# )
```

## Drift Detection

Three statistical tests run continuously on prediction distributions:

| Test | What It Catches | Threshold |
|------|----------------|-----------|
| **PSI** (Population Stability Index) | Distribution shift in binned features | > 0.2 = significant drift |
| **KS** (Kolmogorov-Smirnov) | Maximum distance between CDFs | p < 0.05 = drift detected |
| **JS** (Jensen-Shannon Divergence) | Symmetric divergence between distributions | > 0.1 = notable drift |

## Project Structure

```
sentinel/
├── src/sentinel/
│   ├── platform.py         # Main Sentinel orchestrator
│   ├── serving/            # Inference server, batching, caching
│   ├── gateway/            # API gateway, routing, A/B testing
│   ├── monitoring/         # Metrics, drift detection, alerting
│   ├── scaling/            # Auto-scaling policies
│   ├── versioning/         # Model registry & rollback
│   └── utils/              # Logging, config
├── configs/                 # Default configurations
├── dashboard/               # Monitoring dashboard templates
├── examples/                # Ready-to-run examples
├── tests/                   # Test suite
└── .github/workflows/       # CI pipeline
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
mypy src/sentinel/
ruff check src/ tests/
```

## Roadmap

- [ ] Prometheus + Grafana export
- [ ] GPU utilization tracking
- [ ] Shadow mode (mirror traffic to new model without serving)
- [ ] Feature store integration
- [ ] Model explanation logging (SHAP/LIME per request)
- [ ] Kubernetes operator for auto-deployment

## License

MIT — see [LICENSE](LICENSE)
