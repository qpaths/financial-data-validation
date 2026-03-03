# Contributing to financial-data-validation

Thanks for your interest in contributing!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/qpaths/financial-data-validation.git
cd financial-data-validation

# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=financial_data_validation

# Lint
uv run ruff check src/
uv run ruff format src/
```

## Project Structure

```
financial-data-validation/
├── src/financial_data_validation/
│   ├── diagnostics/          # Individual test implementations
│   │   ├── ljung_box.py
│   │   ├── arch.py
│   │   ├── jarque_bera.py
│   │   ├── kolmogorov_smirnov.py
│   │   ├── variance_ratio.py
│   │   └── runs.py
│   ├── utils.py              # Utility functions
│   └── validator.py          # Main validation orchestrator
├── tests/                    # Unit tests
├── examples/                 # Usage examples
└── docs/                     # Documentation
```

## Adding a New Test

1. Create test file in `src/financial_data_validation/diagnostics/`
2. Implement test function returning `(score, details)` tuple
3. Add comprehensive unit tests in `tests/`
4. Update `validator.py` to include new test
5. Add example usage to `examples/individual_tests.py`
6. Update documentation

## Test Guidelines

- All tests must return `(score: float, details: dict)`
- Score must be in range [0, 1]
- Details must include: `test`, `null_hypothesis`, `interpretation`, `passed`
- Vectorize operations where possible (avoid Python loops)
- Add docstrings with mathematical formulation

## Code Style

- Use `ruff` for formatting (configured in `pyproject.toml`)
- Type hints for public functions
- Docstrings for all public functions
- Keep functions focused (single responsibility)

## Pull Request Process

1. Fork the repo and create a branch
2. Make your changes
3. Add/update tests
4. Ensure all tests pass: `uv run pytest`
5. Format code: `uv run ruff format src/`
6. Submit PR with clear description

## Questions?

Open an issue or reach out to hello@qpaths.io
