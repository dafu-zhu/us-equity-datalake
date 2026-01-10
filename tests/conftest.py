"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_ticker():
    """Provide a sample ticker for testing"""
    return "AAPL"


@pytest.fixture
def sample_tickers():
    """Provide a list of sample tickers for testing"""
    return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]


@pytest.fixture
def sample_date():
    """Provide a sample date for testing"""
    return "2024-06-30"


@pytest.fixture
def sample_date_range():
    """Provide a sample date range for testing"""
    return {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }


@pytest.fixture
def sample_year():
    """Provide a sample year for testing"""
    return 2024


@pytest.fixture
def sample_cik():
    """Provide a sample CIK for testing"""
    return "0000320193"  # Apple Inc.


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (slower)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external API access"
    )

def pytest_collection_modifyitems(config, items):
    """
    Auto-apply markers based on test file location.

    Convention:
      - tests/unit/**         => @pytest.mark.unit
      - tests/integration/**  => @pytest.mark.integration
    """
    root = Path(str(config.rootpath)).resolve()

    unit_dir = (root / "tests" / "unit").resolve()
    integration_dir = (root / "tests" / "integration").resolve()

    for item in items:
        p = Path(str(item.fspath)).resolve()

        if unit_dir in p.parents:
            item.add_marker(pytest.mark.unit)

        if integration_dir in p.parents:
            item.add_marker(pytest.mark.integration)
