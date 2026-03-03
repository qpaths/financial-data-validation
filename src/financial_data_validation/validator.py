"""Main validation orchestrator."""

from dataclasses import dataclass

import numpy as np

from financial_data_validation.diagnostics.arch import arch_test
from financial_data_validation.diagnostics.jarque_bera import jarque_bera_test
from financial_data_validation.diagnostics.kolmogorov_smirnov import kolmogoronv_smirnov_test
from financial_data_validation.diagnostics.ljung_box import ljung_box_test
from financial_data_validation.diagnostics.runs import runs_test
from financial_data_validation.diagnostics.variance_ratio import variance_ratio_test
from financial_data_validation.utils import compute_returns


@dataclass
class ValidationReport:
    """Results from validating a price path dataset."""

    ljung_box_score: float
    arch_score: float
    jarque_bera_score: float
    ks_score: float
    variance_ratio_score: float
    runs_score: float
    quality_score: float
    passed: bool
    details: dict

    def __str__(self) -> str:
        """Human-readable report."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return f"""
Financial Data Validation Report {status}
{"=" * 50}
Overall Quality Score: {self.quality_score:.1f}/100

Individual Test Scores:
  Ljung-Box (autocorrelation):       {self.ljung_box_score:.2f}
  ARCH (volatility clustering):      {self.arch_score:.2f}
  Jarque-Bera (normality):           {self.jarque_bera_score:.2f}
  Kolmogorov-Smirnov (distribution): {self.ks_score:.2f}
  Variance Ratio (mean reversion):   {self.variance_ratio_score:.2f}
  Runs Test (randomness):            {self.runs_score:.2f}

Interpretation:
  90-100: Excellent - indistinguishable from real markets
  80-89:  Good - suitable for most applications
  70-79:  Acceptable - passes minimum requirements
  <70:    Poor - may produce unreliable results
        """.strip()


def validate_paths(
    paths: np.ndarray,
    frequency: str = "daily",
    threshold: float = 70.0,
    weights: dict | None = None,
) -> ValidationReport:
    """
    Validate synthetic financial time series paths.

    Runs 4 statistical tests:
    1. Ljung-Box: Tests for autocorrelation (should be absent)
    2. ARCH: Tests for volatility clustering (should be present)
    3. Jarque-Bera: Tests for reasonable distribution moments
    4. Kolmogorov-Smirnov: Tests distribution shape vs normal

    Args:
        paths: Array of shape (n_paths, n_timesteps) containing price paths
        frequency: "daily", "hourly", or "minute" (affects lag selection)
        threshold: Minimum quality score to pass (default: 70)
        weights: Custom test weights dict, e.g. {"ljung_box": 0.3, ...}
                 Default: {"ljung_box": 0.25, "arch": 0.30, "jarque_bera": 0.25, "ks": 0.20}

    Returns:
        ValidationReport with test scores and overall quality assessment

    Example:
        >>> import numpy as np
        >>> paths = np.random.lognormal(0, 0.02, size=(1000, 252))
        >>> report = validate_paths(paths)
        >>> print(f"Quality score: {report.quality_score:.1f}")
        >>> if report.passed:
        ...     print("✓ Paths pass statistical validation")
        >>> print(report)  # Full report
    """
    if paths.ndim != 2:
        raise ValueError(f"Expected 2D array (n_paths, n_timesteps), got shape {paths.shape}")

    n_paths, n_timesteps = paths.shape

    if n_paths < 100:
        raise ValueError(f"Need at least 100 paths for reliable validation, got {n_paths}")

    if n_timesteps < 100:
        raise ValueError(f"Need at least 100 timesteps for reliable validation, got {n_timesteps}")

    # Default weights
    if weights is None:
        weights = {
            "ljung_box": 0.20,
            "arch": 0.25,
            "jarque_bera": 0.20,
            "ks": 0.15,
            "variance_ratio": 0.10,
            "runs": 0.10,
        }

    # Validate weights sum to 1.0
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.3f}")

    # Compute returns from prices
    returns = compute_returns(paths)

    # Determine lags based on frequency
    if frequency == "daily":
        lags = 20  # ~1 month
    elif frequency == "hourly":
        lags = 24  # 1 trading day
    elif frequency == "minute":
        lags = 60  # 1 hour
    else:
        raise ValueError(f"Invalid frequency '{frequency}', must be 'daily', 'hourly', or 'minute'")

    # Run all tests
    lb_score, lb_details = ljung_box_test(returns, lags=lags)
    arch_score, arch_details = arch_test(returns, lags=lags)
    jb_score, jb_details = jarque_bera_test(returns)
    ks_score, ks_details = kolmogoronv_smirnov_test(returns)
    vr_score, vr_details = variance_ratio_test(returns)
    runs_score, runs_details = runs_test(returns)

    # Compute composite quality score
    quality_score = (
        weights["ljung_box"] * lb_score * 100
        + weights["arch"] * arch_score * 100
        + weights["jarque_bera"] * jb_score * 100
        + weights["ks"] * ks_score * 100
        + weights["variance_ratio"] * vr_score * 100
        + weights["runs"] * runs_score * 100
    )

    return ValidationReport(
        ljung_box_score=lb_score,
        arch_score=arch_score,
        jarque_bera_score=jb_score,
        ks_score=ks_score,
        variance_ratio_score=vr_score,  # NEW
        runs_score=runs_score,  # NEW
        quality_score=quality_score,
        passed=quality_score >= threshold,
        details={
            "ljung_box": lb_details,
            "arch": arch_details,
            "jarque_bera": jb_details,
            "kolmogorov_smirnov": ks_details,
            "variance_ratio": vr_details,  # NEW
            "runs": runs_details,  # NEW
            "n_paths": n_paths,
            "n_timesteps": n_timesteps,
            "frequency": frequency,
            "threshold": threshold,
            "weights": weights,
        },
    )
