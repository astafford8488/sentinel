"""Example: Run an A/B test between two model versions."""

import random

from sentinel import Sentinel, ModelConfig
from sentinel.gateway import ABTest, TrafficSplit


def main() -> None:
    platform = Sentinel()

    # Register two versions
    for version, accuracy in [("1.2.0", 0.94), ("1.3.0", 0.96)]:
        platform.register(ModelConfig(
            name="fraud-detector",
            version=version,
            model_path=f"./models/fraud_v{version}",
            metadata={"accuracy": accuracy},
        ))

    # Create A/B test
    ab = ABTest(
        name="fraud-v1.2-vs-v1.3",
        control="fraud-detector:1.2.0",
        variant="fraud-detector:1.3.0",
        split=TrafficSplit(control=80, variant=20),
        min_samples=1000,
        confidence_level=0.95,
    )
    platform.gateway.create_ab_test(ab)

    # Simulate traffic
    for _ in range(2000):
        features = {"amount": random.uniform(10, 5000), "hour": random.randint(0, 23)}
        # In production, this would go through platform.gateway.route_request()

    # Check results
    results = platform.gateway.get_ab_results("fraud-v1.2-vs-v1.3")
    if results:
        print(results.significance_test())


if __name__ == "__main__":
    main()
