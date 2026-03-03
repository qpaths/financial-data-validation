"""Variance Ratio test for mean reversion and momentum."""

import numpy as np
from scipy import stats


def variance_ratio_test(
    returns: np.ndarray, lags: list[int] | None = None, significance: float = 0.05
) -> tuple[float, dict]:
    """
    Variance Ratio test for random walk hypothesis.

    Tests whether returns follow a random walk by comparing variance of
    k-period returns to k times the variance of 1-period returns.

    Mathematical formulation:
        VR(k) = Var(r_t + r_{t-1} + ... + r_{t-k+1}) / (k * Var(r_t))

    Under random walk hypothesis:
        VR(k) = 1 for all k

    Real markets typically show:
        VR(k) < 1 → Mean reversion (negative autocorrelation)
        VR(k) > 1 → Momentum (positive autocorrelation)

    Equity markets often show VR < 1 at horizons of 2-10 days (weak mean reversion).

    Args:
        returns: Array of shape (n_paths, n_timesteps)
        lags: List of holding periods to test (default: [2, 5, 10])
        significance: Significance level (default: 0.05)

    Returns:
        (score, details) where:
            score: Quality score based on VR values (0-1)
            details: Dict with VR statistics for each lag

    References:
        Lo, A.W. and MacKinlay, A.C. (1988). Stock Market Prices Do Not Follow
        Random Walks: Evidence from a Simple Specification Test.
    """
    if returns.ndim != 2:
        raise ValueError(f"Expected 2D array, got {returns.shape}")

    n_paths, n_obs = returns.shape

    if lags is None:
        lags = [2, 5, 10]

    max_lag = max(lags)
    if n_obs < max_lag * 3:
        raise ValueError(f"Need at least {max_lag * 3} observations, got {n_obs}")

    vr_results = {}

    for lag in lags:
        vr_values = []
        z_stats = []
        p_values = []

        for returns_path in returns:
            # 1-period variance
            var_1 = np.var(returns_path, ddof=1)

            if var_1 == 0:
                vr_values.append(1.0)
                z_stats.append(0.0)
                p_values.append(1.0)
                continue

            # k-period returns (overlapping)
            k_period_returns = []
            for i in range(n_obs - lag + 1):
                k_return = np.sum(returns_path[i : i + lag])
                k_period_returns.append(k_return)

            # k-period variance
            var_k = np.var(k_period_returns, ddof=1)

            # Variance ratio
            vr = var_k / (lag * var_1)
            vr_values.append(vr)

            # Test statistic (Lo-MacKinlay, 1988)
            # Under H0 (random walk), VR = 1
            # Asymptotic variance for overlapping returns
            nq = len(k_period_returns)

            # Simplified asymptotic variance (homoskedastic case)
            # θ(q) ≈ 2(2q-1)(q-1) / (3q)
            theta = 2 * (2 * lag - 1) * (lag - 1) / (3 * lag)

            # Test statistic
            z_stat = (vr - 1) / np.sqrt(theta / nq)
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))  # Two-tailed

            z_stats.append(z_stat)
            p_values.append(p_value)

        vr_values = np.array(vr_values)
        z_stats = np.array(z_stats)
        p_values = np.array(p_values)

        vr_results[lag] = {
            "mean_vr": float(np.mean(vr_values)),
            "median_vr": float(np.median(vr_values)),
            "std_vr": float(np.std(vr_values)),
            "mean_z_stat": float(np.mean(z_stats)),
            "mean_p_value": float(np.mean(p_values)),
            "vr_range": (float(np.min(vr_values)), float(np.max(vr_values))),
        }

    # Scoring: penalize extreme deviations from VR = 1
    # Realistic range for equity markets: VR ∈ [0.7, 1.1]
    # Score based on mean VR at lag=5 (most informative)
    primary_lag = 5 if 5 in lags else lags[len(lags) // 2]
    mean_vr = vr_results[primary_lag]["mean_vr"]

    # Score calculation
    if 0.85 <= mean_vr <= 1.05:
        score = 1.0  # Excellent
    elif 0.70 <= mean_vr <= 1.15:
        # Linear decay outside excellent range
        if mean_vr < 0.85:
            score = 1.0 - (0.85 - mean_vr) / (0.85 - 0.70) * 0.3
        else:
            score = 1.0 - (mean_vr - 1.05) / (1.15 - 1.05) * 0.3
    elif 0.50 <= mean_vr <= 1.30:
        # Further decay
        if mean_vr < 0.70:
            score = 0.7 - (0.70 - mean_vr) / (0.70 - 0.50) * 0.5
        else:
            score = 0.7 - (mean_vr - 1.15) / (1.30 - 1.15) * 0.5
    else:
        score = 0.0  # Extreme deviation

    details = {
        "test": "variance_ratio",
        "null_hypothesis": "Returns follow random walk (VR = 1)",
        "interpretation": "VR < 1: mean reversion, VR > 1: momentum, VR ≈ 1: random walk",
        "lags_tested": lags,
        "primary_lag": primary_lag,
        "n_paths": n_paths,
        "n_observations": n_obs,
        "variance_ratios": vr_results,
        "passed": 0.80 <= mean_vr <= 1.10,  # Reasonable range
    }

    return score, details
