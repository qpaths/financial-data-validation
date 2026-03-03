"""Tests for utility functions."""

import numpy as np
import pytest

from financial_data_validation.utils import (
    compute_acf,
    compute_returns,
    compute_vectorized_acf,
    demean,
)


class TestComputeReturns:
    """Tests for compute_returns function."""

    def test_basic_returns(self):
        """Test basic log returns calculation."""
        paths = np.array([[100, 110, 121], [100, 90, 81]])
        returns = compute_returns(paths)

        # Log returns: ln(110/100) ≈ 0.0953, ln(121/110) ≈ 0.0953
        assert returns.shape == (2, 2)
        assert np.allclose(returns[0, 0], np.log(110 / 100), atol=1e-6)
        assert np.allclose(returns[0, 1], np.log(121 / 110), atol=1e-6)

    def test_wrong_dimensions(self):
        """Test error on wrong dimensions."""
        with pytest.raises(ValueError, match="Expected 2D array"):
            compute_returns(np.array([100, 110, 121]))

    def test_negative_prices(self):
        """Test error on negative prices."""
        paths = np.array([[100, -110, 121]])
        with pytest.raises(ValueError, match="must be positive"):
            compute_returns(paths)

    def test_zero_prices(self):
        """Test error on zero prices."""
        paths = np.array([[100, 0, 121]])
        with pytest.raises(ValueError, match="must be positive"):
            compute_returns(paths)


class TestDemean:
    """Tests for demean function."""

    def test_removes_mean(self):
        """Test that demean removes mean."""
        x = np.array([1, 2, 3, 4, 5])
        result = demean(x)
        assert np.isclose(np.mean(result), 0.0)
        assert np.allclose(result, [-2, -1, 0, 1, 2])

    def test_already_zero_mean(self):
        """Test on already demeaned data."""
        x = np.array([-2, -1, 0, 1, 2])
        result = demean(x)
        assert np.allclose(result, x)


class TestComputeACF:
    """Tests for compute_acf function (single path)."""

    def test_white_noise_acf(self):
        """Test ACF of white noise should be near zero."""
        np.random.seed(42)
        x = np.random.randn(1000)
        x = demean(x)

        acf = compute_acf(x, nlags=10)

        # Lag 0 should be 1.0
        assert np.isclose(acf[0], 1.0)

        # Other lags should be small (< 0.1 for large sample)
        assert np.all(np.abs(acf[1:]) < 0.1)

    def test_constant_series(self):
        """Test ACF of constant series."""
        x = np.ones(100)
        x = demean(x)  # All zeros after demeaning

        acf = compute_acf(x, nlags=5)

        # Should return all 1.0 (convention for zero variance)
        assert np.allclose(acf, 1.0)

    def test_ar1_process(self):
        """Test ACF of AR(1) process."""
        np.random.seed(42)
        n = 1000
        phi = 0.7

        # Generate AR(1): x_t = phi * x_{t-1} + epsilon_t
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + np.random.randn()

        x = demean(x)
        acf = compute_acf(x, nlags=5)

        # ACF of AR(1) should be approximately phi^k
        expected_acf = phi ** np.arange(6)
        assert np.allclose(acf, expected_acf, atol=0.1)


class TestComputeVectorizedACF:
    """Tests for compute_vectorized_acf function (multiple paths)."""

    def test_matches_single_path(self):
        """Test vectorized matches single path computation."""
        np.random.seed(42)
        n_paths = 10
        n_obs = 200

        x = np.random.randn(n_paths, n_obs)
        x = x - np.mean(x, axis=1, keepdims=True)  # Demean each path

        # Compute vectorized
        acf_vec = compute_vectorized_acf(x, nlags=10)

        # Compute individually and compare
        for i in range(n_paths):
            acf_single = compute_acf(x[i], nlags=10)
            assert np.allclose(acf_vec[i], acf_single, atol=1e-10)

    def test_white_noise_multiple_paths(self):
        """Test vectorized ACF on white noise."""
        np.random.seed(42)
        x = np.random.randn(1000, 500)
        x = x - np.mean(x, axis=1, keepdims=True)

        acf = compute_vectorized_acf(x, nlags=20)

        # All paths should have lag 0 = 1.0
        assert np.allclose(acf[:, 0], 1.0)

        # Mean ACF at other lags should be near 0
        mean_acf = np.mean(acf[:, 1:], axis=0)
        assert np.all(np.abs(mean_acf) < 0.05)

    def test_zero_variance_paths(self):
        """Test handling of zero variance paths."""
        x = np.zeros((5, 100))  # All constant

        acf = compute_vectorized_acf(x, nlags=10)

        # Should return all 1.0
        assert np.allclose(acf, 1.0)

    def test_shape_output(self):
        """Test output shape is correct."""
        n_paths = 100
        n_obs = 252
        nlags = 20

        x = np.random.randn(n_paths, n_obs)
        x = x - np.mean(x, axis=1, keepdims=True)

        acf = compute_vectorized_acf(x, nlags=nlags)

        assert acf.shape == (n_paths, nlags + 1)
