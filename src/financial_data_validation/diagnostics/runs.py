"""Runs test for randomness of return signs."""

import numpy as np
from scipy import stats


def runs_test(returns: np.ndarray, significance: float = 0.05) -> tuple[float, dict]:
    """
    Runs test for randomness of positive/negative returns.

    A "run" is a sequence of consecutive returns with the same sign.
    Example: [+, +, -, -, -, +, +] has 3 runs.

    Tests whether the number of runs is consistent with random sequencing.

    Mathematical formulation:
        Under H0 (random sequence):
            E[R] = (2*n_pos*n_neg)/(n_pos + n_neg) + 1
            Var[R] = (2*n_pos*n_neg*(2*n_pos*n_neg - n_pos - n_neg)) /
                     ((n_pos + n_neg)²*(n_pos + n_neg - 1))

        Test statistic:
            Z = (R - E[R]) / sqrt(Var[R])

        Under H0, Z ~ N(0,1)

    Args:
        returns: Array of shape (n_paths, n_timesteps)
        significance: Significance level (default: 0.05)

    Returns:
        (score, details) where:
            score: Proportion of paths passing the test (0-1)
            details: Dict with test statistics

    Notes:
        Too few runs → trending/momentum
        Too many runs → mean reversion/oscillation
        Expected runs → randomness (good)
    """
    if returns.ndim != 2:
        raise ValueError(f"Expected 2D array, got {returns.shape}")

    n_paths, n_obs = returns.shape

    if n_obs < 20:
        raise ValueError(f"Need at least 20 observations for runs test, got {n_obs}")

    n_runs_list = []
    expected_runs_list = []
    z_stats = []
    p_values = []

    for returns_path in returns:
        # Get signs (positive = 1, negative = -1, ignore zeros)
        signs = np.sign(returns_path)
        signs = signs[signs != 0]  # Remove zeros

        if len(signs) < 10:
            # Too few non-zero returns
            n_runs_list.append(np.nan)
            expected_runs_list.append(np.nan)
            z_stats.append(np.nan)
            p_values.append(np.nan)
            continue

        # Count runs
        runs = 1  # Start with 1 run
        for i in range(1, len(signs)):
            if signs[i] != signs[i - 1]:
                runs += 1

        # Count positive and negative returns
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        n = n_pos + n_neg

        # Expected number of runs under H0
        expected_runs = (2 * n_pos * n_neg) / n + 1

        # Variance of runs under H0
        var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))

        if var_runs <= 0:
            n_runs_list.append(runs)
            expected_runs_list.append(expected_runs)
            z_stats.append(0.0)
            p_values.append(1.0)
            continue

        # Test statistic (with continuity correction)
        if runs > expected_runs:
            z_stat = (runs - 0.5 - expected_runs) / np.sqrt(var_runs)
        else:
            z_stat = (runs + 0.5 - expected_runs) / np.sqrt(var_runs)

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        n_runs_list.append(runs)
        expected_runs_list.append(expected_runs)
        z_stats.append(z_stat)
        p_values.append(p_value)

    # Filter out NaNs
    valid_mask = ~np.isnan(p_values)
    p_values_valid = np.array(p_values)[valid_mask]

    if len(p_values_valid) == 0:
        # All paths failed
        score = 0.0
        pass_rate = 0.0
    else:
        # Score: proportion passing (p > significance)
        pass_mask = p_values_valid > significance
        score = np.mean(pass_mask)
        pass_rate = score

    details = {
        "test": "runs_test",
        "null_hypothesis": "Return signs are randomly sequenced",
        "interpretation": "Too few runs → trending, too many runs → mean reversion",
        "significance_level": significance,
        "n_paths": n_paths,
        "n_observations": n_obs,
        "mean_runs": float(np.nanmean(n_runs_list)),
        "mean_expected_runs": float(np.nanmean(expected_runs_list)),
        "mean_z_stat": float(np.nanmean(z_stats)),
        "mean_p_value": float(np.nanmean(p_values)),
        "pass_rate": float(pass_rate),
        "n_passed": int(np.sum(p_values_valid > significance)),
        "n_failed": int(np.sum(p_values_valid <= significance)),
        "passed": pass_rate >= 0.90,  # At least 90% should pass
    }

    return score, details
