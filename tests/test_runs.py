"""Tests for Runs test."""

import numpy as np
import pytest

from financial_data_validation.diagnostics.runs import runs_test


class TestRunsTest:
    """Tests for runs_test function."""

    def test_random_signs_pass(self):
        """Random signs should pass."""
        np.random.seed(42)
        returns = np.random.randn(1000, 252) * 0.02

        score, details = runs_test(returns)

        # Most paths should pass
        assert score > 0.85
        assert details["passed"]

    def test_trending_fails(self):
        """Trending (too few runs) should fail."""
        np.random.seed(42)
        n_paths = 1000
        n_obs = 252

        # Create trending returns (positive autocorrelation = fewer runs)
        returns = np.zeros((n_paths, n_obs))
        for i in range(n_paths):
            returns[i, 0] = np.random.randn() * 0.02
            for t in range(1, n_obs):
                # High positive autocorrelation
                returns[i, t] = 0.6 * returns[i, t - 1] + np.random.randn() * 0.01

        score, details = runs_test(returns)

        # Should fail (too few runs)
        assert score < 0.30
        assert details["mean_runs"] < details["mean_expected_runs"] * 0.8

    def test_oscillating_fails(self):
        """Oscillating (too many runs) should fail."""
        np.random.seed(42)
        n_paths = 1000
        n_obs = 252

        # Create oscillating returns (negative autocorrelation = more runs)
        returns = np.zeros((n_paths, n_obs))
        for i in range(n_paths):
            returns[i, 0] = np.random.randn() * 0.02
            for t in range(1, n_obs):
                # Negative autocorrelation
                returns[i, t] = -0.5 * returns[i, t - 1] + np.random.randn() * 0.01

        score, details = runs_test(returns)

        # Should fail (too many runs)
        assert score < 0.30
        assert details["mean_runs"] > details["mean_expected_runs"] * 1.2

    def test_wrong_dimensions(self):
        """Test error on wrong dimensions."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            runs_test(np.random.randn(100))

    def test_too_few_observations(self):
        """Test error when too few observations."""
        returns = np.random.randn(100, 15)
        with pytest.raises(ValueError, match="Need at least"):
            runs_test(returns)
