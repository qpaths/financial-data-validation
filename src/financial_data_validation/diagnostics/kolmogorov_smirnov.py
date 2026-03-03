"""Kolmogorov-Smirnov test for distribution matching."""

import numpy as np
from scipy import stats


def kolmogoronv_smirnov_test(returns: np.ndarray, significance: float = 0.05) -> tuple[float, dict]:
    """
    Kolmogorov-Smirnov test comparing return distribution to normal.

    [... same docstring ...]
    """
    if returns.ndim != 2:
        raise ValueError(f"Expected 2D array, got {returns.shape}")

    n_paths, n_obs = returns.shape

    if n_obs < 20:
        raise ValueError(f"Need at least 20 observations for KS test, got {n_obs}")

    ks_statistics = []
    p_values = []

    for returns_path in returns:
        mean = np.mean(returns_path)
        std = np.std(returns_path, ddof=1)

        if std == 0:
            ks_statistics.append(0.0)
            p_values.append(1.0)
            continue

        standardized = (returns_path - mean) / std
        ks_stat, p_value = stats.kstest(standardized, "norm")

        ks_statistics.append(ks_stat)
        p_values.append(p_value)

    ks_statistics = np.array(ks_statistics)
    p_values = np.array(p_values)

    mean_ks = np.mean(ks_statistics)

    # Calibrated scoring based on empirical results
    if mean_ks <= 0.04:
        score = 1.0
    elif mean_ks <= 0.065:
        score = 1.0 - (mean_ks - 0.04) / 0.025 * 0.4
    elif mean_ks <= 0.08:
        score = 0.6 - (mean_ks - 0.065) / 0.015 * 0.4
    elif mean_ks <= 0.12:
        score = 0.2 - (mean_ks - 0.08) / 0.04 * 0.2
    else:
        score = 0.0

    pass_mask = ks_statistics < 0.08
    pass_rate = np.mean(pass_mask)

    details = {
        "test": "kolmogorov_smirnov",
        "null_hypothesis": "Returns follow normal distribution",
        "interpretation": "Lower D statistic is better (closer to normal)",
        "significance_level": significance,
        "n_paths": n_paths,
        "n_observations": n_obs,
        "mean_ks_statistic": float(mean_ks),
        "median_ks_statistic": float(np.median(ks_statistics)),
        "std_ks_statistic": float(np.std(ks_statistics)),
        "min_ks_statistic": float(np.min(ks_statistics)),
        "max_ks_statistic": float(np.max(ks_statistics)),
        "mean_p_value": float(np.mean(p_values)),
        "median_p_value": float(np.median(p_values)),
        "ks_range": {"min": float(np.min(ks_statistics)), "max": float(np.max(ks_statistics))},
        "pass_rate": float(pass_rate),
        "n_passed": int(np.sum(pass_mask)),
        "n_failed": int(np.sum(~pass_mask)),
        "passed": mean_ks < 0.08,
    }

    return score, details
