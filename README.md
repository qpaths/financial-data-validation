# financial-data-validation

[![PyPI](https://img.shields.io/pypi/v/financial-data-validation)](https://pypi.org/project/financial-data-validation/)
[![Python](https://img.shields.io/pypi/pyversions/financial-data-validation)](https://pypi.org/project/financial-data-validation/)
[![Tests](https://github.com/qpaths/financial-data-validation/actions/workflows/test.yml/badge.svg)](https://github.com/qpaths/financial-data-validation/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/qpaths/financial-data-validation/blob/main/LICENSE)

Statistical validation for synthetic financial time series.

## Installation

```bash
pip install finanial-data-validation
```

## Quick Start

```python
from financial_data_validation import validate_paths
import numpy as np

# Your synthetic price paths (n_paths, n_timesteps)
paths = np.random.lognormal(0, 0.02, size=(1000, 252))

# Validate
report = validate_paths(paths)

print(f"Quality Score: {report.quality_score:.1f}/100")
print(f"Passed: {'✓' if report.passed else '✗'}")
```

## What It Tests

| Test                   | Validates                              | Pass Criteria |
| ---------------------- | -------------------------------------- | ------------- |
| **Ljung-Box**          | No spurious autocorrelation in returns | p > 0.05      |
| **ARCH**               | Volatility clustering present          | p < 0.05      |
| **Jarque-Bera**        | Returns approximately normal           | p > 0.01      |
| **Kolmogorov-Smirnov** | Distribution shape matches expectation | D < 0.05      |

## Quality Scores

- **90-100**: Excellent — indistinguishable from real markets
- **80-89**: Good — suitable for most applications
- **70-79**: Acceptable — passes minimum requirements
- **< 70**: Poor — may produce unreliable results

## Why This Exists

Synthetic market data is only useful if it's statistically realistic. This package validates whether generated price paths exhibit the properties of real financial markets: volatility clustering, fat tails, and proper correlation structure.

Built by [QPaths](https://qpaths.io) — we use this to validate every dataset we generate.

## Example Output

```python
ValidationReport(
    ljung_box_score=0.87,
    arch_score=0.92,
    jarque_bera_score=0.81,
    ks_score=0.85,
    quality_score=86.4,
    passed=True
)
```

## Use Cases

- Validate synthetic data before backtesting trading strategies
- Quality-check Monte Carlo simulations
- Verify stochastic model implementations
- Test financial data generation pipelines

## Documentation

Full documentation: [github.com/qpaths/financial-data-validation](https://github.com/qpaths/financial-data-validation)

## License

MIT

---

**Part of the QPaths ecosystem** — Generate validated synthetic market data at [qpaths.io](https://qpaths.io)
