"""
Unit tests for collection.models module
Tests dataclasses and enums used across the collection layer
"""
import pytest
import datetime
from quantdl.collection.models import (
    FndDataPoint,
    TickDataPoint,
    TickField,
    DataSource
)


class TestFndDataPoint:
    """Test FndDataPoint dataclass"""

    def test_creation_with_all_fields(self):
        """Test creating FndDataPoint with all fields"""
        point = FndDataPoint(
            timestamp=datetime.date(2024, 3, 31),
            value=1000000.0,
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 3, 31),
            frame="CY2024Q1",
            is_instant=False,
            form="10-Q",
            accn="0001234567-24-000001"
        )

        assert point.timestamp == datetime.date(2024, 3, 31)
        assert point.value == 1000000.0
        assert point.start_date == datetime.date(2024, 1, 1)
        assert point.end_date == datetime.date(2024, 3, 31)
        assert point.frame == "CY2024Q1"
        assert point.is_instant is False
        assert point.form == "10-Q"
        assert point.accn == "0001234567-24-000001"

    def test_creation_with_optional_none(self):
        """Test creating FndDataPoint with optional fields as None"""
        point = FndDataPoint(
            timestamp=datetime.date(2024, 3, 31),
            value=500000.0,
            start_date=None,
            end_date=datetime.date(2024, 3, 31),
            frame=None,
            is_instant=True,
            form="10-K",
            accn=None
        )

        assert point.start_date is None
        assert point.frame is None
        assert point.accn is None
        assert point.is_instant is True


class TestTickDataPoint:
    """Test TickDataPoint dataclass"""

    def test_creation_with_all_fields(self):
        """Test creating TickDataPoint with all fields"""
        point = TickDataPoint(
            timestamp="2024-01-15T09:30:00Z",
            open=150.25,
            high=152.75,
            low=149.50,
            close=151.00,
            volume=1000000,
            num_trades=5000,
            vwap=150.80
        )

        assert point.timestamp == "2024-01-15T09:30:00Z"
        assert point.open == 150.25
        assert point.high == 152.75
        assert point.low == 149.50
        assert point.close == 151.00
        assert point.volume == 1000000
        assert point.num_trades == 5000
        assert point.vwap == 150.80

    def test_price_validation_logic(self):
        """Test that high >= low and close is within range"""
        point = TickDataPoint(
            timestamp="2024-01-15T09:30:00Z",
            open=150.00,
            high=155.00,
            low=148.00,
            close=152.00,
            volume=500000,
            num_trades=2500,
            vwap=151.00
        )

        # Validate OHLC relationships
        assert point.high >= point.low
        assert point.close >= point.low
        assert point.close <= point.high


class TestTickField:
    """Test TickField enum"""

    def test_enum_values(self):
        """Test that enum has correct values"""
        assert TickField.CLOSE.value == 'c'
        assert TickField.HIGH.value == 'h'
        assert TickField.LOW.value == 'l'
        assert TickField.NUM_TRADES.value == 'n'
        assert TickField.OPEN.value == 'o'
        assert TickField.TIMESTAMP.value == 't'
        assert TickField.VOLUME.value == 'v'
        assert TickField.VWAP.value == 'vw'

    def test_enum_members(self):
        """Test that all expected members exist"""
        expected_members = {
            'CLOSE', 'HIGH', 'LOW', 'NUM_TRADES',
            'OPEN', 'TIMESTAMP', 'VOLUME', 'VWAP'
        }
        actual_members = set(TickField.__members__.keys())
        assert actual_members == expected_members

    def test_enum_access_by_name(self):
        """Test accessing enum by name"""
        assert TickField['CLOSE'] == TickField.CLOSE
        assert TickField['VOLUME'] == TickField.VOLUME


class TestDataSource:
    """Test DataSource abstract base class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that DataSource cannot be instantiated directly"""
        with pytest.raises(TypeError):
            DataSource()

    def test_must_implement_abstract_methods(self):
        """Test that subclasses must implement all abstract methods"""

        # Missing all methods
        with pytest.raises(TypeError):
            class IncompleteSource(DataSource):
                pass
            IncompleteSource()

        # Missing some methods
        with pytest.raises(TypeError):
            class PartialSource(DataSource):
                def supports_concept(self, concept: str) -> bool:
                    return True
            PartialSource()

    def test_valid_implementation(self):
        """Test that a complete implementation works"""

        class ValidSource(DataSource):
            def supports_concept(self, concept: str) -> bool:
                return concept in ["revenue", "net_income"]

            def extract_concept(self, concept: str):
                if concept == "revenue":
                    return [
                        FndDataPoint(
                            timestamp=datetime.date(2024, 3, 31),
                            value=1000000.0,
                            start_date=datetime.date(2024, 1, 1),
                            end_date=datetime.date(2024, 3, 31),
                            frame="CY2024Q1",
                            is_instant=False,
                            form="10-Q",
                            accn="test"
                        )
                    ]
                return None

            def get_coverage_period(self):
                return ("2020-01-01", "2024-12-31")

        # Should instantiate successfully
        source = ValidSource()
        assert source.supports_concept("revenue") is True
        assert source.supports_concept("unknown") is False
        assert source.get_coverage_period() == ("2020-01-01", "2024-12-31")

        data = source.extract_concept("revenue")
        assert len(data) == 1
        assert data[0].value == 1000000.0
