# Examples

Practical examples for using `financial-data-validation`.

## Quick Start

```bash
# Run basic validation example
python examples/basic_usage.py

# Run individual tests
python examples/individual_tests.py

# Custom validation settings
python examples/custom_validation.py

# Compare different models
python examples/comparing_models.py
```

## Files

### `basic_usage.py`

Complete workflow: generate paths → validate → interpret results.
**Best for:** Getting started quickly.

### `individual_tests.py`

Run each diagnostic test independently with detailed output.
**Best for:** Understanding what each test measures.

### `custom_validation.py`

Customize weights, thresholds, and frequency settings.
**Best for:** Production use cases with specific requirements.

### `comparing_models.py`

Compare GBM, GARCH, and other models side-by-side.
**Best for:** Choosing the right model for your use case.

## Common Use Cases

### Validate Your Own Data

```python
import numpy as np
from financial_data_validation import validate_paths

# Your price paths (n_paths, n_timesteps)
paths = your_simulation_function()

# Validate
report = validate_paths(paths, frequency="daily")

if report.passed:
    print("✓ Data quality verified")
else:
    print("✗ Quality issues detected")
    print(f"Score: {report.quality_score:.1f}/100")
```

### Run One Specific Test

```python
from financial_data_validation.utils import compute_returns
from financial_data_validation.diagnostics.arch import arch_test

returns = compute_returns(paths)
score, details = arch_test(returns, lags=20)

print(f"ARCH score: {score:.3f}")
print(f"Volatility clustering: {'Yes' if details['passed'] else 'No'}")
```

### Custom Quality Threshold

```python
# Stricter validation for production
report = validate_paths(
    paths,
    frequency="daily",
    threshold=85.0  # Default is 70
)
```

### Emphasize Specific Tests

```python
# Risk management: emphasize tail behavior
weights = {
    "ljung_box": 0.15,
    "arch": 0.20,
    "jarque_bera": 0.30,  # Fat tails
    "ks": 0.25,           # Distribution shape
    "variance_ratio": 0.05,
    "runs": 0.05
}

report = validate_paths(paths, weights=weights)
```

## Expected Output

All examples print detailed validation reports showing:

- Overall quality score (0-100)
- Individual test scores (0-1)
- Pass/fail status
- Interpretation guidance

## Need Help?

- Package documentation: [GitHub](https://github.com/qpaths/financial-data-validation)
- Report issues: [Issues](https://github.com/qpaths/financial-data-validation/issues)
