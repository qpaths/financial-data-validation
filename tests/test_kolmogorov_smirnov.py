"""Tests for Kolmogorov-Smirnov test."""

import numpy as np
import pytest

from financial_data_validation.diagnostics.kolmogorov_smirnov import kolmogoronv_smirnov_test


class TestKolmogorovSmirnov:
    """Tests for kolmogoronv_smirnov_test function."""

    def test_normal_passes(self):
        """Normal returns should pass."""
        np.random.seed(42)
        returns = np.random.randn(10000, 252) * 0.02

        score, details = kolmogoronv_smirnov_test(returns)

        # Should pass with excellent score
        assert score > 0.90
        assert details["passed"]
        assert details["mean_ks_statistic"] < 0.05

    def test_uniform_fails(self):
        """Uniform distribution should score low."""
        np.random.seed(42)
        returns = np.random.uniform(-0.05, 0.05, size=(10000, 252))

        score, details = kolmogoronv_smirnov_test(returns)

        # Should get low score
        assert score < 0.30
        assert details["mean_ks_statistic"] > 0.07

    def test_exponential_fails(self):
        """Exponential distribution should fail completely."""
        np.random.seed(42)
        returns = np.random.exponential(0.02, size=(10000, 252)) - 0.02

        score, details = kolmogoronv_smirnov_test(returns)

        # Should fail
        assert score < 0.10
        assert not details["passed"]
        assert details["mean_ks_statistic"] > 0.12

    def test_fat_tails_acceptable(self):
        """Fat-tailed returns should be acceptable."""
        np.random.seed(42)
        returns = np.random.standard_t(df=5, size=(10000, 252)) * 0.02

        score, details = kolmogoronv_smirnov_test(returns)

        # Should get decent score
        assert score > 0.50

    def test_wrong_dimensions(self):
        """Test error on wrong dimensions."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            kolmogoronv_smirnov_test(np.random.randn(100))

    def test_too_few_observations(self):
        """Test error when too few observations."""
        returns = np.random.randn(100, 15)
        with pytest.raises(ValueError, match="Need at least"):
            kolmogoronv_smirnov_test(returns)
