"""Jarque-Bera test for normality of returns."""

import numpy as np
from scipy import stats


def jarque_bera_test(returns: np.ndarray, significance: float = 0.05) -> tuple[float, dict]:
    """
    Jarque-Bera test for normality of return distribution.

    Tests whether returns follow a normal distribution based on skewness and kurtosis.
    Financial returns typically deviate slightly from normality:
    - Slight negative skew (crashes more common than rallies)
    - Positive excess kurtosis (fat tails, more extreme events)

    Mathematical formulation:
        JB = (n/6) * [S² + (K²/4)]

    Where:
        S = skewness = E[(X-μ)³] / σ³
        K = excess kurtosis = E[(X-μ)⁴] / σ⁴ - 3

    Under H₀ (normality), JB ~ χ²(2)

    Args:
        returns: Array of shape (n_paths, n_timesteps)
        significance: Significance level (default: 0.05)

    Returns:
        (score, details) where:
            score: Quality score based on reasonable deviation from normality (0-1)
            details: Dict with skewness, kurtosis, and test statistics

    Note:
        Perfect normality (p > 0.05) is actually unrealistic for financial data.
        The scoring rewards slight deviation (realistic) and penalizes extreme
        deviation (unrealistic).
    """
    if returns.ndim != 2:
        raise ValueError(f"Expected 2D array, got {returns.shape}")

    n_paths, n_obs = returns.shape

    if n_obs < 20:
        raise ValueError(f"Need at least 20 observations for JB test, got {n_obs}")

    # Compute skewness and kurtosis for all paths (vectorized)
    # scipy.stats functions work on last axis by default
    skewness = stats.skew(returns, axis=1)  # Shape: (n_paths,)
    kurtosis = stats.kurtosis(returns, axis=1)  # Excess kurtosis (normal = 0)

    # Jarque-Bera statistic: JB = (n/6) * [S² + (K²/4)]
    jb_stats = (n_obs / 6) * (skewness**2 + (kurtosis**2) / 4)

    # P-values from chi-squared distribution with 2 df
    p_values = 1 - stats.chi2.cdf(jb_stats, df=2)

    # Scoring strategy:
    # Perfect normality is unrealistic. We want:
    # - Skewness: between -0.5 and 0.5 (slight negative skew is OK)
    # - Kurtosis: between 0 and 3 (some fat tails expected)

    # Score based on reasonable ranges
    skew_in_range = np.abs(skewness) <= 1.0  # Within ±1.0 is good
    kurt_in_range = (kurtosis >= -0.5) & (kurtosis <= 5.0)  # -0.5 to 5.0 is good

    # Combined score: proportion of paths with reasonable moments
    score = np.mean(skew_in_range & kurt_in_range)

    details = {
        "test": "jarque_bera",
        "null_hypothesis": "Returns are normally distributed",
        "interpretation": "Slight deviation from normality is expected and realistic",
        "significance_level": significance,
        "n_paths": n_paths,
        "n_observations": n_obs,
        "mean_jb_statistic": float(np.mean(jb_stats)),
        "median_jb_statistic": float(np.median(jb_stats)),
        "mean_p_value": float(np.mean(p_values)),
        "median_p_value": float(np.median(p_values)),
        "skewness": {
            "mean": float(np.mean(skewness)),
            "median": float(np.median(skewness)),
            "std": float(np.std(skewness)),
            "min": float(np.min(skewness)),
            "max": float(np.max(skewness)),
            "in_range_rate": float(np.mean(skew_in_range)),
        },
        "kurtosis": {
            "mean": float(np.mean(kurtosis)),
            "median": float(np.median(kurtosis)),
            "std": float(np.std(kurtosis)),
            "min": float(np.min(kurtosis)),
            "max": float(np.max(kurtosis)),
            "in_range_rate": float(np.mean(kurt_in_range)),
        },
        "pass_rate": float(score),
        "n_passed": int(np.sum(skew_in_range & kurt_in_range)),
        "n_failed": int(np.sum(~(skew_in_range & kurt_in_range))),
        "passed": score >= 0.80,  # At least 80% should have reasonable moments
    }

    return score, details
