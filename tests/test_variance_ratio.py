"""Tests for Variance Ratio test."""

import numpy as np
import pytest

from financial_data_validation.diagnostics.variance_ratio import variance_ratio_test


class TestVarianceRatio:
    """Tests for variance_ratio_test function."""

    def test_random_walk_passes(self):
        """Random walk should have VR ≈ 1."""
        np.random.seed(42)
        returns = np.random.randn(1000, 252) * 0.02

        score, details = variance_ratio_test(returns, lags=[2, 5, 10])

        # Should pass with VR near 1
        assert score > 0.85
        assert details["passed"]

        # Check VR values
        vr_5 = details["variance_ratios"][5]["mean_vr"]
        assert 0.90 < vr_5 < 1.10

    def test_mean_reversion_detected(self):
        """Mean reverting returns should have VR < 1."""
        np.random.seed(42)
        n_paths = 1000
        n_obs = 252

        # AR(1) with negative autocorrelation (mean reversion)
        returns = np.zeros((n_paths, n_obs))
        for i in range(n_paths):
            returns[i, 0] = np.random.randn() * 0.02
            for t in range(1, n_obs):
                returns[i, t] = -0.3 * returns[i, t - 1] + np.random.randn() * 0.02

        score, details = variance_ratio_test(returns, lags=[2, 5, 10])

        # VR should be less than 1
        vr_5 = details["variance_ratios"][5]["mean_vr"]
        assert vr_5 < 0.85

    def test_momentum_detected(self):
        """Momentum returns should have VR > 1."""
        np.random.seed(42)
        n_paths = 1000
        n_obs = 252

        # AR(1) with positive autocorrelation (momentum)
        returns = np.zeros((n_paths, n_obs))
        for i in range(n_paths):
            returns[i, 0] = np.random.randn() * 0.02
            for t in range(1, n_obs):
                returns[i, t] = 0.3 * returns[i, t - 1] + np.random.randn() * 0.02

        score, details = variance_ratio_test(returns, lags=[2, 5, 10])

        # VR should be greater than 1
        vr_5 = details["variance_ratios"][5]["mean_vr"]
        assert vr_5 > 1.10

    def test_wrong_dimensions(self):
        """Test error on wrong dimensions."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            variance_ratio_test(np.random.randn(100))

    def test_too_few_observations(self):
        """Test error when too few observations."""
        returns = np.random.randn(100, 20)
        with pytest.raises(ValueError, match="Need at least"):
            variance_ratio_test(returns, lags=[10])
