"""
Basic usage example for financial-data-validation.

Demonstrates how to validate synthetic price paths with default settings.
"""

import numpy as np

from financial_data_validation import validate_paths


def generate_garch_paths(n_paths=1000, n_timesteps=252):
    """Generate GARCH(1,1) price paths for demonstration."""
    s0 = 100.0
    omega = 0.000005
    alpha = 0.12
    beta = 0.85

    paths = np.zeros((n_paths, n_timesteps))
    vol = np.zeros((n_paths, n_timesteps))

    paths[:, 0] = s0
    vol[:, 0] = 0.02

    for t in range(1, n_timesteps):
        z = np.random.randn(n_paths)
        returns = vol[:, t - 1] * z
        paths[:, t] = paths[:, t - 1] * np.exp(returns)

        vol_squared = omega + alpha * returns**2 + beta * vol[:, t - 1] ** 2
        vol[:, t] = np.sqrt(np.maximum(vol_squared, 1e-8))

    return paths


def main():
    """Run basic validation example."""
    print("Generating 1,000 GARCH(1,1) price paths...")
    np.random.seed(42)
    paths = generate_garch_paths(n_paths=1000, n_timesteps=252)

    print(f"Paths shape: {paths.shape}")
    print(f"Price range: ${paths.min():.2f} - ${paths.max():.2f}")

    print("\nRunning validation...")
    report = validate_paths(paths, frequency="daily")

    # Print full report
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Access individual scores
    print("\nDetailed breakdown:")
    print(f"  Ljung-Box:      {report.ljung_box_score:.3f} (no autocorrelation)")
    print(f"  ARCH:           {report.arch_score:.3f} (volatility clustering)")
    print(f"  Jarque-Bera:    {report.jarque_bera_score:.3f} (distribution moments)")
    print(f"  KS:             {report.ks_score:.3f} (distribution shape)")
    print(f"  Variance Ratio: {report.variance_ratio_score:.3f} (random walk)")
    print(f"  Runs:           {report.runs_score:.3f} (sign randomness)")

    # Access detailed test results
    print("\nTest details:")
    lb_details = report.details["ljung_box"]
    print(f"  Ljung-Box: {lb_details['n_passed']}/{lb_details['n_paths']} paths passed")

    arch_details = report.details["arch"]
    print(f"  ARCH: {arch_details['n_passed']}/{arch_details['n_paths']} paths show clustering")

    # Check if validation passed
    if report.passed:
        print("\n✓ Data is suitable for backtesting and analysis!")
    else:
        print("\n✗ Data quality concerns - review individual test scores")


if __name__ == "__main__":
    main()
