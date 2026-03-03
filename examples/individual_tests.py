"""
Using individual diagnostic tests directly.

Shows how to run each test independently for fine-grained analysis.
"""

import numpy as np

from financial_data_validation.diagnostics.arch import arch_test
from financial_data_validation.diagnostics.jarque_bera import jarque_bera_test
from financial_data_validation.diagnostics.kolmogorov_smirnov import kolmogoronv_smirnov_test
from financial_data_validation.diagnostics.ljung_box import ljung_box_test
from financial_data_validation.diagnostics.runs import runs_test
from financial_data_validation.diagnostics.variance_ratio import variance_ratio_test
from financial_data_validation.utils import compute_returns


def generate_gbm_paths(n_paths=1000, n_timesteps=252, mu=0.0001, sigma=0.02):
    """Generate Geometric Brownian Motion paths."""
    s0 = 100
    paths = np.zeros((n_paths, n_timesteps))
    paths[:, 0] = s0

    for t in range(1, n_timesteps):
        z = np.random.randn(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma**2) + sigma * z)

    return paths


def main():
    """Run individual tests on GBM paths."""
    print("Generating GBM price paths...")
    np.random.seed(42)
    paths = generate_gbm_paths(n_paths=1000, n_timesteps=252)

    # Convert to returns for testing
    returns = compute_returns(paths)
    print(f"Returns shape: {returns.shape}")

    print("\n" + "=" * 60)
    print("INDIVIDUAL TEST RESULTS")
    print("=" * 60)

    # Test 1: Ljung-Box (Autocorrelation)
    print("\n1. LJUNG-BOX TEST (Autocorrelation)")
    print("-" * 40)
    lb_score, lb_details = ljung_box_test(returns, lags=20)
    print(f"Score: {lb_score:.3f}")
    print(f"Mean p-value: {lb_details['mean_p_value']:.4f}")
    print(f"Pass rate: {lb_details['pass_rate']:.1%}")
    print(
        f"Interpretation: {
            '✓ No autocorrelation' if lb_details['passed'] else '✗ Autocorrelation detected'
        }"
    )

    # Test 2: ARCH (Volatility Clustering)
    print("\n2. ARCH TEST (Volatility Clustering)")
    print("-" * 40)
    arch_score, arch_details = arch_test(returns, lags=20)
    print(f"Score: {arch_score:.3f}")
    print(f"Mean p-value: {arch_details['mean_p_value']:.4f}")
    print(f"Pass rate: {arch_details['pass_rate']:.1%}")
    print(
        f"Interpretation: {
            '✓ Volatility clustering present'
            if arch_details['passed']
            else '✗ No volatility clustering (constant vol)'
        }"
    )

    # Test 3: Jarque-Bera (Normality)
    print("\n3. JARQUE-BERA TEST (Distribution Moments)")
    print("-" * 40)
    jb_score, jb_details = jarque_bera_test(returns)
    print(f"Score: {jb_score:.3f}")
    print(f"Mean skewness: {jb_details['skewness']['mean']:.4f}")
    print(f"Mean kurtosis: {jb_details['kurtosis']['mean']:.4f}")
    print(f"Pass rate: {jb_details['pass_rate']:.1%}")
    print(
        f"Interpretation: {
            '✓ Reasonable distribution'
            if jb_details['passed']
            else '✗ Extreme skewness or kurtosis'
        }"
    )

    # Test 4: Kolmogorov-Smirnov (Distribution Shape)
    print("\n4. KOLMOGOROV-SMIRNOV TEST (Distribution Shape)")
    print("-" * 40)
    ks_score, ks_details = kolmogoronv_smirnov_test(returns)
    print(f"Score: {ks_score:.3f}")
    print(f"Mean KS statistic: {ks_details['mean_ks_statistic']:.4f}")
    print(f"Pass rate: {ks_details['pass_rate']:.1%}")
    print(
        f"Interpretation: {
            '✓ Good fit to normal' if ks_details['passed'] else '✗ Poor fit to normal'
        }"
    )

    # Test 5: Variance Ratio (Random Walk)
    print("\n5. VARIANCE RATIO TEST (Random Walk)")
    print("-" * 40)
    vr_score, vr_details = variance_ratio_test(returns, lags=[2, 5, 10])
    print(f"Score: {vr_score:.3f}")
    for lag in [2, 5, 10]:
        vr = vr_details["variance_ratios"][lag]["mean_vr"]
        print(f"  VR({lag}): {vr:.3f}")
    print(
        f"Interpretation: {
            '✓ Random walk behavior' if vr_details['passed'] else '✗ Mean reversion or momentum'
        }"
    )

    # Test 6: Runs Test (Sign Randomness)
    print("\n6. RUNS TEST (Sign Randomness)")
    print("-" * 40)
    runs_score, runs_details = runs_test(returns)
    print(f"Score: {runs_score:.3f}")
    print(f"Mean runs: {runs_details['mean_runs']:.1f}")
    print(f"Expected runs: {runs_details['mean_expected_runs']:.1f}")
    print(f"Pass rate: {runs_details['pass_rate']:.1%}")
    print(
        f"Interpretation: {
            '✓ Random sign sequencing' if runs_details['passed'] else '✗ Non-random patterns'
        }"
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("GBM (constant volatility) passes most tests except ARCH.")
    print("This is expected - GBM has no volatility clustering.")


if __name__ == "__main__":
    main()
