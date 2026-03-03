"""Ljung-Box test for autocorrelation in returns."""

import numpy as np
from scipy import stats

from financial_data_validation.utils import compute_vectorized_acf


def ljung_box_test(returns: np.ndarray, lags: int = 20, significance: float = 0.05) -> tuple:
    """
    Ljung-Box test for autocorrelation in returns.

    Tests whether returns exhibit significant autocorrelation. For efficient
    markets, returns should be unpredictable from past returns (no autocorrelation).

    Mathematical formulation:
        Q = n(n+2) * Σ[ρ²ₖ / (n-k)] for k=1 to h

    Where:
        n = number of observations
        h = number of lags
        ρₖ = sample autocorrelation at lag k

    Under H₀ (no autocorrelation), Q ~ χ²(h)

    Args:
        returns: Array of shape (n_paths, n_timesteps) with log returns
        lags: Number of lags to test (default: 20 for daily data ≈ 1 month)
        significance: Significance level (default: 0.05)

    Returns:
        (score, details) where:
            score: Proportion of paths passing test (0-1, higher is better)
            details: Dict with test statistics and diagnostics

    Example:
        >>> returns = np.random.randn(1000, 251)  # Random returns (should pass)
        >>> score, details = ljung_box_test(returns)
        >>> score > 0.90  # Most paths should show no autocorrelation
        True
    """
    if returns.ndim != 2:
        raise ValueError(f"Expected 2D array (n_paths, n_timesteps), got {returns.shape}")

    n_paths, n_obs = returns.shape

    if n_obs < 2 * lags:
        raise ValueError(f"Need at least {2 * lags} observations for {lags} lags, got {n_obs}")

    # Demean the series
    x = returns - np.mean(returns, axis=1, keepdims=True)

    # Compute autocorrelation function
    acf = compute_vectorized_acf(x, nlags=lags)

    # Ljung-Box Q Statistic
    # Q = n(n+2) * sum(rho_k^2 / (n-k)) for k=1 to h
    rho = acf[:, 1 : lags + 1]
    k = np.arange(1, lags + 1)
    q_stats = n_obs * (n_obs + 2) * np.sum(rho**2 / (n_obs - k), axis=1)

    # P-value from chi-squared distribution with h degrees of freedom
    p_values = 1 - stats.chi2.cdf(q_stats, df=lags)

    # Score: proportion passing (high p-value = no autocorrelation = good)
    pass_mask = p_values > significance
    score = np.mean(pass_mask)

    details = {
        "test": "ljung_box",
        "null_hypothesis": "No autocorrelation in returns up to lag h",
        "interpretation": "High p-value is good (returns are unpredictable)",
        "lags": lags,
        "significance_level": significance,
        "n_paths": n_paths,
        "n_observations": n_obs,
        "mean_p_value": float(np.mean(p_values)),
        "median_p_value": float(np.median(p_values)),
        "std_p_value": float(np.std(p_values)),
        "min_p_value": float(np.min(p_values)),
        "max_p_value": float(np.max(p_values)),
        "mean_q_statistic": float(np.mean(q_stats)),
        "pass_rate": float(score),
        "n_passed": int(np.sum(pass_mask)),
        "n_failed": int(np.sum(~pass_mask)),
        "passed": score >= 0.90,  # At least 90% of paths should pass
    }

    return (score, details)
