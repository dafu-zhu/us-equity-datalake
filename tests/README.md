# Test Suite for US Equity Data Lake

This directory contains comprehensive unit and integration tests for the US Equity Data Lake project.

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (including test dependencies)
uv sync --all-extras
```

## Test Structure

```
tests/
├── unit/                    # Unit tests (fast, isolated, no external dependencies)
│   ├── collection/          # Tests for data collection modules
│   │   ├── test_models.py   # Dataclass and enum tests
│   │   └── ...
│   ├── derived/             # Tests for derived metrics computation
│   │   ├── test_metrics.py  # Derived fundamental metrics
│   │   └── test_ttm.py      # Trailing twelve months computation
│   ├── universe/            # Tests for stock universe management
│   │   ├── test_current.py  # Current universe from Nasdaq
│   │   ├── test_historical.py
│   │   └── test_manager.py
│   ├── master/              # Tests for security master
│   │   └── ...
│   └── storage/             # Tests for storage layer
│       ├── test_config_loader.py
│       ├── test_rate_limiter.py
│       └── ...
└── integration/             # Integration tests (slower, may require external services)
```

## Running Tests

### Run All Tests

```bash
# Using uv
uv run pytest
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
uv run pytest -m unit

# Integration tests only (slower)
uv run pytest -m integration

# Specific module
uv run pytest tests/unit/collection/
uv run pytest tests/unit/derived/test_metrics.py

# Specific test function
uv run pytest tests/unit/storage/test_rate_limiter.py::TestRateLimiter::test_thread_safety
```

### Run with Coverage

```bash
# Generate coverage report
uv run pytest --cov=src/quantdl --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Tests in Parallel

```bash
# Install pytest-xdist (if not already in pyproject.toml)
uv add --dev pytest-xdist

# Run tests in parallel
uv run pytest -n auto
```

## Test Markers

Tests are marked with the following markers (defined in `pytest.ini`):

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (require external services)
- `@pytest.mark.slow`: Slow tests
- `@pytest.mark.external`: Tests requiring external API access (WRDS, SEC EDGAR, Alpaca, S3)

## Writing New Tests

### Test File Naming

- Test files should be named `test_*.py`
- Test classes should be named `Test*`
- Test functions should be named `test_*`

### Example Test

```python
"""
Unit tests for mymodule
Tests functionality of MyClass
"""
import pytest
from quantdl.mymodule import MyClass


class TestMyClass:
    """Test MyClass"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return {"key": "value"}

    def test_basic_functionality(self, sample_data):
        """Test basic functionality"""
        obj = MyClass(sample_data)
        assert obj.process() == expected_result

    def test_edge_case(self):
        """Test edge case handling"""
        obj = MyClass(None)
        with pytest.raises(ValueError):
            obj.process()
```

### Mocking External Dependencies

Use `unittest.mock` to mock external dependencies:

```python
from unittest.mock import Mock, patch

@patch('quantdl.collection.fundamental.requests.get')
def test_api_call(mock_get):
    """Test API call with mocked response"""
    mock_get.return_value.json.return_value = {"data": "test"}
    result = fetch_data()
    assert result == {"data": "test"}
```

## Test Guidelines

1. **Unit Tests Should Be Fast**: Aim for < 1 second per test
2. **Isolate Dependencies**: Mock external services (APIs, databases, S3)
3. **Test Edge Cases**: Empty inputs, null values, errors
4. **Use Fixtures**: Share test data setup across tests
5. **Clear Assertions**: Use descriptive assertion messages
6. **Test One Thing**: Each test should verify one behavior
7. **Arrange-Act-Assert**: Follow AAA pattern
   ```python
   def test_example():
       # Arrange
       data = create_test_data()

       # Act
       result = process(data)

       # Assert
       assert result == expected
   ```

## Continuous Integration

Tests run automatically on:
- Every push to main branch
- Every pull request
- Scheduled nightly runs (for integration tests)

## Test Coverage Goals

- Overall: > 80%
- Critical modules (storage, derived): > 90%
- Collection modules: > 75%

## Troubleshooting

### Tests Failing Locally

```bash
# Clear pytest cache
uv run pytest --cache-clear

# Run in verbose mode
uv run pytest -vv

# Show print statements
uv run pytest -s
```

### Import Errors

```bash
# Sync dependencies to ensure everything is installed
uv sync

# Or install package in development mode
uv pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/us-equity-datalake/src:$PYTHONPATH
```

### Missing Test Dependencies

```bash
# Install with dev dependencies
uv sync --group dev
```

### Fixture Not Found

Ensure fixtures are:
1. Defined in the same test file, OR
2. Defined in `conftest.py` at the appropriate level

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Effective Python Testing With Pytest](https://realpython.com/pytest-python-testing/)
