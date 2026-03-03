"""Tests for main validator."""

import numpy as np
import pytest

from financial_data_validation import validate_paths


class TestValidator:
    """Tests for validate_paths function."""

    def test_garch_passes(self):
        """GARCH model should pass validation."""
        np.random.seed(42)
        n_paths = 5000
        n_timesteps = 252

        # Generate GARCH paths
        paths = self._generate_garch_paths(n_paths, n_timesteps)

        report = validate_paths(paths, frequency="daily")

        # Should pass
        assert report.passed
        assert report.quality_score >= 70
        assert 0 <= report.quality_score <= 100

    def test_gbm_partial_pass(self):
        """GBM should get partial score (no ARCH effects)."""
        np.random.seed(42)
        n_paths = 5000
        n_timesteps = 252

        # Generate GBM paths
        s0 = 100
        mu = 0.0001
        sigma = 0.02

        paths = np.zeros((n_paths, n_timesteps))
        paths[:, 0] = s0

        for t in range(1, n_timesteps):
            z = np.random.randn(n_paths)
            paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma**2) + sigma * z)

        report = validate_paths(paths, frequency="daily", threshold=75.0)

        # Should fail overall (no ARCH)
        assert not report.passed
        assert report.quality_score < 75

        # But individual tests should mostly pass except ARCH
        assert report.ljung_box_score > 0.85
        assert report.arch_score < 0.15
        assert report.jarque_bera_score > 0.90

    def test_too_few_paths(self):
        """Test error with too few paths."""
        paths = np.random.randn(50, 252) * 0.02 + 100
        with pytest.raises(ValueError, match="at least 100 paths"):
            validate_paths(paths)

    def test_too_few_timesteps(self):
        """Test error with too few timesteps."""
        paths = np.random.randn(1000, 50) * 0.02 + 100
        with pytest.raises(ValueError, match="at least 100 timesteps"):
            validate_paths(paths)

    def test_custom_weights(self):
        """Test custom weighting scheme."""
        paths = self._generate_garch_paths(1000, 252)

        custom_weights = {
            "ljung_box": 0.3,
            "arch": 0.3,
            "jarque_bera": 0.2,
            "ks": 0.1,
            "variance_ratio": 0.05,
            "runs": 0.05,
        }

        report = validate_paths(paths, weights=custom_weights)

        assert report.details["weights"] == custom_weights

    def test_invalid_weights(self):
        """Test error on invalid weights."""
        paths = self._generate_garch_paths(1000, 252)

        bad_weights = {
            "ljung_box": 0.5,
            "arch": 0.3,
            "jarque_bera": 0.1,
            "ks": 0.1,
            "variance_ratio": 0.05,
            "runs": 0.05,
        }  # Sum = 1.1

        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_paths(paths, weights=bad_weights)

    def test_report_string_format(self):
        """Test report string formatting."""
        paths = self._generate_garch_paths(1000, 252)
        report = validate_paths(paths)

        report_str = str(report)

        # Check key elements are in the string
        assert "Quality Score" in report_str
        assert "Ljung-Box" in report_str
        assert "ARCH" in report_str
        assert ("✓ PASSED" in report_str) or ("✗ FAILED" in report_str)

    @staticmethod
    def _generate_garch_paths(n_paths, n_timesteps):
        """Helper to generate GARCH(1,1) paths."""
        s0 = 100.0
        omega = 0.000005
        alpha = 0.12
        beta = 0.85

        paths = np.zeros((n_paths, n_timesteps))
        vol = np.zeros((n_paths, n_timesteps))

        paths[:, 0] = s0
        vol[:, 0] = 0.02

        for t in range(1, n_timesteps):
            z = np.random.randn(n_paths)
            returns = vol[:, t - 1] * z
            paths[:, t] = paths[:, t - 1] * np.exp(returns)

            vol_squared = omega + alpha * returns**2 + beta * vol[:, t - 1] ** 2
            vol[:, t] = np.sqrt(np.maximum(vol_squared, 1e-8))

        return paths
