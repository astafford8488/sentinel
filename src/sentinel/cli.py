"""CLI entry point for SENTINEL."""

from __future__ import annotations

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(package_name="sentinel-ml")
def main() -> None:
    """SENTINEL — Production ML Inference & Observability Platform."""
    pass


@main.command()
@click.option("--name", "-n", required=True, help="Model name")
@click.option("--version", "-v", required=True, help="Model version")
@click.option("--path", "-p", required=True, help="Path to model artifacts")
@click.option("--framework", default="generic", help="Model framework (sklearn, xgboost, torch)")
def register(name: str, version: str, path: str, framework: str) -> None:
    """Register a model version."""
    from sentinel import Sentinel, ModelConfig

    platform = Sentinel()
    platform.register(ModelConfig(name=name, version=version, model_path=path, framework=framework))
    console.print(f"Registered {name}:{version}")


@main.command()
@click.option("--models", "-m", required=True, help="Comma-separated model:version pairs")
@click.option("--port", "-p", default=8000, type=int)
@click.option("--workers", "-w", default=1, type=int)
def serve(models: str, port: int, workers: int) -> None:
    """Start the inference server."""
    from sentinel import Sentinel

    platform = Sentinel()
    model_list = [m.strip() for m in models.split(",")]
    console.print(f"Starting SENTINEL on port {port} with models: {model_list}")
    platform.serve(models=model_list, port=port, workers=workers)


@main.command()
def status() -> None:
    """Show platform status."""
    from sentinel import Sentinel

    platform = Sentinel()
    info = platform.status()
    console.print(info)


@main.command()
@click.option("--name", required=True, help="A/B test name")
@click.option("--control", required=True, help="Control model:version")
@click.option("--variant", required=True, help="Variant model:version")
@click.option("--split", default="80/20", help="Traffic split (control/variant)")
def ab_test(name: str, control: str, variant: str, split: str) -> None:
    """Create an A/B test."""
    from sentinel.gateway import ABTest, TrafficSplit

    parts = split.split("/")
    ts = TrafficSplit(control=int(parts[0]), variant=int(parts[1]))
    test = ABTest(name=name, control=control, variant=variant, split=ts)
    console.print(f"A/B test '{name}' created: {control} ({ts.control}%) vs {variant} ({ts.variant}%)")


@main.command()
@click.argument("model_name")
@click.option("--to", "to_version", required=True, help="Version to roll back to")
def rollback(model_name: str, to_version: str) -> None:
    """Roll back a model to a previous version."""
    console.print(f"Rolling back {model_name} to {to_version}")


@main.command()
@click.option("--dashboard", is_flag=True, help="Open live dashboard")
def monitor(dashboard: bool) -> None:
    """Monitor model health and metrics."""
    if dashboard:
        console.print("Dashboard: http://localhost:8000/metrics")
    else:
        console.print("Use --dashboard to open the monitoring dashboard")


if __name__ == "__main__":
    main()
