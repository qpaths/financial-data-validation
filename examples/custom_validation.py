"""
Custom validation with adjusted weights and thresholds.

Shows how to customize the validation for specific use cases.
"""

import numpy as np

from financial_data_validation import validate_paths


def generate_paths(n_paths=1000, n_timesteps=252):
    """Generate GARCH paths."""
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
    """Demonstrate custom validation settings."""
    np.random.seed(42)
    paths = generate_paths()

    print("=" * 60)
    print("DEFAULT VALIDATION")
    print("=" * 60)

    # Default weights and threshold
    report_default = validate_paths(paths, frequency="daily")
    print(report_default)

    print("\n" + "=" * 60)
    print("CUSTOM VALIDATION: Emphasize ARCH Effects")
    print("=" * 60)

    # Custom weights: prioritize volatility clustering
    custom_weights = {
        "ljung_box": 0.15,  # Reduced
        "arch": 0.40,  # Increased (very important)
        "jarque_bera": 0.15,  # Reduced
        "ks": 0.10,  # Reduced
        "variance_ratio": 0.10,
        "runs": 0.10,
    }

    report_custom = validate_paths(paths, frequency="daily", weights=custom_weights)

    print(f"Quality Score: {report_custom.quality_score:.1f}/100")
    print(f"Status: {'PASSED' if report_custom.passed else 'FAILED'}")
    print("\nARCH weight: 40% (vs 25% default)")
    print(f"ARCH score: {report_custom.arch_score:.3f}")

    print("\n" + "=" * 60)
    print("CUSTOM VALIDATION: Strict Threshold")
    print("=" * 60)

    # Higher threshold for production use
    report_strict = validate_paths(
        paths,
        frequency="daily",
        threshold=85.0,  # Stricter than default 70
    )

    print(f"Quality Score: {report_strict.quality_score:.1f}/100")
    print("Threshold: 85 (vs 70 default)")
    print(f"Status: {'PASSED' if report_strict.passed else 'FAILED'}")

    print("\n" + "=" * 60)
    print("CUSTOM VALIDATION: Hourly Data")
    print("=" * 60)

    # Different frequency affects lag selection
    report_hourly = validate_paths(
        paths,
        frequency="hourly",  # Uses 24 lags instead of 20
    )

    print("Frequency: hourly")
    print("Lags tested: 24 (1 trading day)")
    print(f"Quality Score: {report_hourly.quality_score:.1f}/100")

    print("\n" + "=" * 60)
    print("USE CASE RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. High-frequency trading:
   - Use frequency="minute" 
   - Emphasize ARCH and runs tests
   
2. Risk management:
   - Use threshold=85
   - Emphasize Jarque-Bera and KS tests (tail behavior)
   
3. Academic research:
   - Use default settings
   - Report all individual test scores
   
4. Production backtesting:
   - Use threshold=80
   - Emphasize ARCH (volatility clustering critical)
    """)


if __name__ == "__main__":
    main()
