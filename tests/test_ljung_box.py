"""Tests for Ljung-Box test."""

import numpy as np
import pytest

from financial_data_validation.diagnostics.ljung_box import ljung_box_test


class TestLjungBox:
    """Tests for ljung_box_test function."""

    def test_white_noise_passes(self):
        """White noise should pass (no autocorrelation)."""
        np.random.seed(42)
        returns = np.random.randn(1000, 252) * 0.02

        score, details = ljung_box_test(returns, lags=20)

        # Most paths should pass
        assert score > 0.85
        assert details["passed"]
        assert details["mean_p_value"] > 0.20  # High p-values

    def test_autocorrelated_fails(self):
        """Autocorrelated returns should fail."""
        np.random.seed(42)
        n_paths = 1000
        n_obs = 252

        # AR(1) with high autocorrelation
        returns = np.zeros((n_paths, n_obs))
        for i in range(n_paths):
            returns[i, 0] = np.random.randn() * 0.02
            for t in range(1, n_obs):
                returns[i, t] = 0.7 * returns[i, t - 1] + np.random.randn() * 0.015

        score, details = ljung_box_test(returns, lags=20)

        # Should fail (low score, low p-values)
        assert score < 0.20
        assert not details["passed"]
        assert details["mean_p_value"] < 0.05

    def test_wrong_dimensions(self):
        """Test error on wrong dimensions."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            ljung_box_test(np.random.randn(100), lags=10)

    def test_too_few_observations(self):
        """Test error when too few observations."""
        returns = np.random.randn(100, 30)
        with pytest.raises(ValueError, match="Need at least"):
            ljung_box_test(returns, lags=20)

    def test_output_structure(self):
        """Test output structure is correct."""
        returns = np.random.randn(100, 100) * 0.02
        score, details = ljung_box_test(returns, lags=10)

        # Check score is in valid range
        assert 0 <= score <= 1

        # Check details has required keys
        assert "test" in details
        assert "mean_p_value" in details
        assert "pass_rate" in details
        assert "passed" in details
        assert details["test"] == "ljung_box"
