"""Utility functions for financial data validation"""

import numpy as np


def compute_returns(paths: np.ndarray) -> np.ndarray:
    """
    Compute log returns from price paths.

    Args:
        paths: Array of shape (n_paths, n_timesteps)

    Returns:
        Array of shape (n_paths, n_timesteps - 1) with log returns
    """
    if paths.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {paths.shape}")

    if np.any(paths <= 0):
        raise ValueError("All prices must be positive for log returns")

    return np.diff(np.log(paths), axis=1)


def demean(x: np.ndarray) -> np.ndarray:
    """Remove mean from series."""
    return x - np.mean(x)


def compute_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """
    Compute autocorrelation function for a single path.

    Args:
        x: 1D time series, already demeaned
        nlags: Number of lags

    Returns:
        Array of length nlags+1 with ACF values (includes lag 0 = 1.0)
    """
    n = len(x)
    c0 = np.dot(x, x) / n

    if c0 == 0:
        return np.ones(nlags + 1)

    acf_values = np.ones(nlags + 1)
    for k in range(1, nlags + 1):
        c_k = np.dot(x[:-k], x[k:]) / n
        acf_values[k] = c_k / c0

    return acf_values


def compute_vectorized_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """
    Compute ACF for multiple paths simultaneously.

    Args:
        x: Array of shape (n_paths, n_timesteps), already demeaned
        nlags: Number of lags

    Returns:
        Array of shape (n_paths, nlags+1) with ACF values
    """
    n_paths, n_obs = x.shape

    # Calculate variance
    c0 = np.sum(x**2, axis=1) / n_obs

    # Handle zero variance
    zero_var_mask = c0 == 0
    c0[zero_var_mask] = 1.0

    # Initialize ACF array
    acf = np.ones((n_paths, nlags + 1))

    for k in range(1, nlags + 1):
        # Autocovariance at lag k: c(k) = (1/n) * sum(x_t * x_{t-k})
        c_k = np.sum(x[:, :-k] * x[:, k:], axis=1) / n_obs

        # Autocorrelation: rho(k) = c(k) / c(0)
        acf[:, k] = c_k / c0

    acf[zero_var_mask, :] = 1.0

    return acf
