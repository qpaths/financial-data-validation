"""Tests for ARCH effects test."""

import numpy as np
import pytest

from financial_data_validation.diagnostics.arch import arch_test


class TestARCH:
    """Tests for arch_test function."""

    def test_constant_volatility_fails(self):
        """Constant volatility (GBM) should fail ARCH test."""
        np.random.seed(42)
        returns = np.random.randn(1000, 252) * 0.02

        score, details = arch_test(returns, lags=20)

        # Should not show ARCH effects
        assert score < 0.15
        assert details["mean_p_value"] > 0.30  # High p-values (no ARCH)

    def test_garch_passes(self):
        """GARCH returns should pass (show ARCH effects)."""
        np.random.seed(42)
        n_paths = 1000
        n_obs = 252

        # GARCH(1,1)
        omega = 0.000005
        alpha = 0.12
        beta = 0.85

        returns = np.zeros((n_paths, n_obs))
        vol = np.ones((n_paths, n_obs)) * 0.02

        for i in range(n_paths):
            for t in range(1, n_obs):
                z = np.random.randn()
                returns[i, t] = vol[i, t - 1] * z
                vol_sq = omega + alpha * returns[i, t - 1] ** 2 + beta * vol[i, t - 1] ** 2
                vol[i, t] = np.sqrt(max(vol_sq, 1e-8))

        score, details = arch_test(returns, lags=20)

        # Should show ARCH effects
        assert score > 0.70
        assert details["passed"]
        assert details["mean_p_value"] < 0.10

    def test_wrong_dimensions(self):
        """Test error on wrong dimensions."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            arch_test(np.random.randn(100), lags=10)

    def test_too_few_observations(self):
        """Test error when too few observations."""
        returns = np.random.randn(100, 30)
        with pytest.raises(ValueError, match="Need at least"):
            arch_test(returns, lags=20)
