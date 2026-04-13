"""Example: Detect prediction drift between training and production distributions."""

import numpy as np

from sentinel.monitoring import DriftDetector


def main() -> None:
    # Simulated training distribution
    np.random.seed(42)
    training_predictions = np.random.normal(loc=0.35, scale=0.15, size=10000)

    # Simulated production predictions (with drift)
    production_predictions = np.random.normal(loc=0.42, scale=0.18, size=5000)

    # Detect drift
    detector = DriftDetector(
        reference_data=training_predictions,
        psi_threshold=0.2,
        ks_alpha=0.05,
        js_threshold=0.1,
    )

    report = detector.check(production_predictions)

    print(f"PSI:              {report.psi}")
    print(f"KS Statistic:     {report.ks_statistic}")
    print(f"KS p-value:       {report.ks_p_value}")
    print(f"JS Divergence:    {report.js_divergence}")
    print(f"Drift Detected:   {report.is_drifted}")
    print(f"Details:          {report.details}")


if __name__ == "__main__":
    main()
