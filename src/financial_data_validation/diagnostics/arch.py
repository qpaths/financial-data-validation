"""ARCH test for volatility clustering."""

import numpy as np
from scipy import stats

from financial_data_validation.utils import compute_vectorized_acf


def arch_test(
    returns: np.ndarray, lags: int = 20, significance: float = 0.05
) -> tuple[float, dict]:
    """
    ARCH test for volatility clustering (heteroskedasticity).

    Tests for ARCH effects by checking if squared returns exhibit autocorrelation.
    Good financial data SHOULD show volatility clustering (a stylized fact of
    real markets).

    Mathematical formulation:
        1. Square the returns: ε²ₜ
        2. Test for autocorrelation in ε²ₜ using Engle's LM test
        3. LM = n * R² ≈ n * Σ(ρ²ₖ) where ρₖ is ACF of squared returns

    Under H₀ (no ARCH effects), LM ~ χ²(q)

    Args:
        returns: Array of shape (n_paths, n_timesteps)
        lags: Number of lags to test (default: 20 for daily data)
        significance: Significance level (default: 0.05)

    Returns:
        (score, details) where:
            score: Proportion of paths showing ARCH effects (0-1, higher is better)
            details: Dict with test statistics

    Example:
        >>> # Returns with volatility clustering (should show ARCH effects)
        >>> returns = generate_garch_returns(1000, 252)
        >>> score, details = arch_test(returns)
        >>> score > 0.70  # Most paths should show clustering
        True
    """
    if returns.ndim != 2:
        raise ValueError(f"Expected 2D array, got {returns.shape}")

    n_paths, n_obs = returns.shape

    if n_obs < 2 * lags:
        raise ValueError(f"Need at least {2 * lags} observations, got {n_obs}")

    # Square the returns (all paths at once)
    squared_returns = returns**2

    # Demean squared returns
    x = squared_returns - np.mean(squared_returns, axis=1, keepdims=True)

    # Compute ACF for all paths vectorized
    acf = compute_vectorized_acf(x, nlags=lags)

    # Engle's LM statistic for all paths
    # LM ≈ n * sum(rho_k^2) for k=1 to q
    rho = acf[:, 1 : lags + 1]  # Shape: (n_paths, lags)
    lm_stats = n_obs * np.sum(rho**2, axis=1)  # Shape: (n_paths,)

    # P-values for all paths
    p_values = 1 - stats.chi2.cdf(lm_stats, df=lags)

    # Score: proportion with ARCH effects (p < significance is good)
    # Lower p-value = evidence of ARCH effects = volatility clustering = good
    pass_mask = p_values < significance
    score = np.mean(pass_mask)

    details = {
        "test": "arch_effects",
        "null_hypothesis": "No ARCH effects (homoskedasticity)",
        "interpretation": "Low p-value is good (volatility clustering present)",
        "lags": lags,
        "significance_level": significance,
        "n_paths": n_paths,
        "n_observations": n_obs,
        "mean_p_value": float(np.mean(p_values)),
        "median_p_value": float(np.median(p_values)),
        "std_p_value": float(np.std(p_values)),
        "min_p_value": float(np.min(p_values)),
        "max_p_value": float(np.max(p_values)),
        "mean_lm_statistic": float(np.mean(lm_stats)),
        "pass_rate": float(score),
        "n_passed": int(np.sum(pass_mask)),
        "n_failed": int(np.sum(~pass_mask)),
        "passed": score >= 0.70,  # At least 70% should show ARCH effects
    }

    return score, details
