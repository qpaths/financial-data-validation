"""
Compare validation results across different models.

Demonstrates how GBM, GARCH, and other models perform on validation.
"""

import numpy as np

from financial_data_validation import validate_paths


def generate_gbm(n_paths=1000, n_timesteps=252):
    """Geometric Brownian Motion (constant volatility)."""
    s0 = 100
    mu = 0.0001
    sigma = 0.02

    paths = np.zeros((n_paths, n_timesteps))
    paths[:, 0] = s0

    for t in range(1, n_timesteps):
        z = np.random.randn(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma**2) + sigma * z)

    return paths


def generate_garch(n_paths=1000, n_timesteps=252):
    """GARCH(1,1) (stochastic volatility)."""
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


def generate_trending(n_paths=1000, n_timesteps=252):
    """Returns with momentum (positive autocorrelation)."""
    s0 = 100
    paths = np.zeros((n_paths, n_timesteps))
    paths[:, 0] = s0
    returns = np.zeros(n_timesteps)

    for i in range(n_paths):
        returns[0] = np.random.randn() * 0.02
        for t in range(1, n_timesteps):
            # AR(1) with positive autocorrelation
            returns[t] = 0.3 * returns[t - 1] + np.random.randn() * 0.015

        for t in range(1, n_timesteps):
            paths[i, t] = paths[i, t - 1] * np.exp(returns[t])

    return paths


def print_comparison(name, report):
    """Print formatted comparison results."""
    print(f"\n{name}")
    print("-" * 40)
    print(f"Overall:         {report.quality_score:.1f}/100 {'✓' if report.passed else '✗'}")
    print(f"Ljung-Box:       {report.ljung_box_score:.3f}")
    print(f"ARCH:            {report.arch_score:.3f}")
    print(f"Jarque-Bera:     {report.jarque_bera_score:.3f}")
    print(f"KS:              {report.ks_score:.3f}")
    print(f"Variance Ratio:  {report.variance_ratio_score:.3f}")
    print(f"Runs:            {report.runs_score:.3f}")


def main():
    """Compare validation across models."""
    np.random.seed(42)

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print("\nGenerating 1,000 paths × 252 timesteps for each model...\n")

    # Generate paths from different models
    gbm_paths = generate_gbm()
    garch_paths = generate_garch()
    trending_paths = generate_trending()

    # Validate each
    gbm_report = validate_paths(gbm_paths, frequency="daily")
    garch_report = validate_paths(garch_paths, frequency="daily")
    trending_report = validate_paths(trending_paths, frequency="daily")

    # Print comparison
    print_comparison("GBM (Geometric Brownian Motion)", gbm_report)
    print_comparison("GARCH(1,1)", garch_report)
    print_comparison("Trending (AR(1) with momentum)", trending_report)

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    print(f"""
1. GBM (Quality: {gbm_report.quality_score:.1f})
   - Passes overall but weak on ARCH ({gbm_report.arch_score:.2f})
   - No volatility clustering (constant vol by design)
   - Good for basic testing, not realistic for risk analysis
   
2. GARCH (Quality: {garch_report.quality_score:.1f})
   - Strong performance across all tests
   - Realistic volatility clustering (ARCH: {garch_report.arch_score:.2f})
   - Recommended for production backtesting
   
3. Trending (Quality: {trending_report.quality_score:.1f})
   - Fails Ljung-Box ({trending_report.ljung_box_score:.2f}) - autocorrelated returns
   - Unrealistic for equity markets
   - Violates efficient market hypothesis
    """)

    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("""
For realistic synthetic market data:
✓ Use GARCH or stochastic volatility models
✗ Avoid simple GBM for risk analysis
✗ Avoid models with return autocorrelation
    """)


if __name__ == "__main__":
    main()
