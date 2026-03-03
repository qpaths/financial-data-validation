"""Tests for Jarque-Bera test."""

import numpy as np
import pytest

from financial_data_validation.diagnostics.jarque_bera import jarque_bera_test


class TestJarqueBera:
    """Tests for jarque_bera_test function."""

    def test_normal_passes(self):
        """Normal returns should pass."""
        np.random.seed(42)
        returns = np.random.randn(10000, 252) * 0.02

        score, details = jarque_bera_test(returns)

        # Should pass with good score
        assert score > 0.90
        assert details["passed"]
        assert np.abs(details["skewness"]["mean"]) < 0.2
        assert np.abs(details["kurtosis"]["mean"]) < 0.5

    def test_extreme_skew_fails(self):
        """Highly skewed returns should fail."""
        np.random.seed(42)
        # Exponential distribution (high positive skew)
        returns = np.random.exponential(0.02, size=(10000, 252)) - 0.02

        score, details = jarque_bera_test(returns)

        # Should fail
        assert score < 0.10
        assert not details["passed"]
        assert details["skewness"]["mean"] > 1.5

    def test_fat_tails_acceptable(self):
        """Student's t (fat tails) should be acceptable."""
        np.random.seed(42)
        # t-distribution with df=5 (fat tails but not extreme)
        returns = np.random.standard_t(df=5, size=(10000, 252)) * 0.02

        score, details = jarque_bera_test(returns)

        # Should get reasonable score
        assert score > 0.70
        assert 2 < details["kurtosis"]["mean"] < 5

    def test_wrong_dimensions(self):
        """Test error on wrong dimensions."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            jarque_bera_test(np.random.randn(100))

    def test_too_few_observations(self):
        """Test error when too few observations."""
        returns = np.random.randn(100, 15)
        with pytest.raises(ValueError, match="Need at least"):
            jarque_bera_test(returns)
