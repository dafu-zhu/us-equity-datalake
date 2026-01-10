"""
Unit tests for derived.ttm module
Tests Trailing Twelve Months (TTM) computation
"""
import pytest
import polars as pl
import datetime as dt
from quantdl.derived.ttm import compute_ttm_long


class TestComputeTTMLong:
    """Test compute_ttm_long function"""

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        empty_df = pl.DataFrame()
        result = compute_ttm_long(empty_df)

        assert result.is_empty()

    def test_missing_required_columns(self):
        """Test with missing required columns"""
        df = pl.DataFrame({
            'symbol': ['AAPL'],
            'value': [1000000.0]
            # Missing: as_of_date, concept, frame
        })

        result = compute_ttm_long(df)
        assert result.is_empty()

    def test_basic_ttm_computation(self):
        """Test basic TTM computation with 4 quarters"""
        # Create data for 4 consecutive quarters
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 4,
            'as_of_date': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
            'accn': ['acc1', 'acc2', 'acc3', 'acc4'],
            'form': ['10-Q', '10-Q', '10-Q', '10-K'],
            'concept': ['rev'] * 4,
            'value': [100.0, 110.0, 120.0, 130.0],
            'start': ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'],
            'end': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
            'frame': ['CY2024Q1', 'CY2024Q2', 'CY2024Q3', 'CY2024Q4']
        })

        result = compute_ttm_long(df, duration_concepts={'rev'})

        # Should have 1 TTM value (for the 4th quarter)
        assert len(result) == 1

        # TTM should be sum of all 4 quarters
        assert result['value'][0] == 460.0  # 100 + 110 + 120 + 130

        # Should have correct symbol and concept
        assert result['symbol'][0] == 'AAPL'
        assert result['concept'][0] == 'rev'

    def test_multiple_concepts(self):
        """Test TTM computation with multiple concepts"""
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 8,
            'as_of_date': [
                '2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31',
                '2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'
            ],
            'accn': ['acc1', 'acc2', 'acc3', 'acc4'] * 2,
            'form': ['10-Q', '10-Q', '10-Q', '10-K'] * 2,
            'concept': ['rev'] * 4 + ['net_inc'] * 4,
            'value': [100.0, 110.0, 120.0, 130.0, 10.0, 15.0, 20.0, 25.0],
            'start': ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'] * 2,
            'end': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'] * 2,
            'frame': ['CY2024Q1', 'CY2024Q2', 'CY2024Q3', 'CY2024Q4'] * 2
        })

        result = compute_ttm_long(df, duration_concepts={'rev', 'net_inc'})

        # Should have 2 TTM values (one for each concept)
        assert len(result) == 2

        # Check revenue TTM
        rev_row = result.filter(pl.col('concept') == 'rev')
        assert len(rev_row) == 1
        assert rev_row['value'][0] == 460.0

        # Check net income TTM
        ni_row = result.filter(pl.col('concept') == 'net_inc')
        assert len(ni_row) == 1
        assert ni_row['value'][0] == 70.0  # 10 + 15 + 20 + 25

    def test_multiple_symbols(self):
        """Test TTM computation with multiple symbols"""
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 4 + ['MSFT'] * 4,
            'as_of_date': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'] * 2,
            'accn': ['acc1', 'acc2', 'acc3', 'acc4'] * 2,
            'form': ['10-Q'] * 8,
            'concept': ['rev'] * 8,
            'value': [100.0, 110.0, 120.0, 130.0, 200.0, 210.0, 220.0, 230.0],
            'start': ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'] * 2,
            'end': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'] * 2,
            'frame': ['CY2024Q1', 'CY2024Q2', 'CY2024Q3', 'CY2024Q4'] * 2
        })

        result = compute_ttm_long(df, duration_concepts={'rev'})

        # Should have 2 TTM values (one for each symbol)
        assert len(result) == 2

        # Check AAPL TTM
        aapl_row = result.filter(pl.col('symbol') == 'AAPL')
        assert len(aapl_row) == 1
        assert aapl_row['value'][0] == 460.0

        # Check MSFT TTM
        msft_row = result.filter(pl.col('symbol') == 'MSFT')
        assert len(msft_row) == 1
        assert msft_row['value'][0] == 860.0

    def test_insufficient_quarters(self):
        """Test with fewer than 4 quarters (should not produce TTM)"""
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 3,
            'as_of_date': ['2024-03-31', '2024-06-30', '2024-09-30'],
            'accn': ['acc1', 'acc2', 'acc3'],
            'form': ['10-Q'] * 3,
            'concept': ['rev'] * 3,
            'value': [100.0, 110.0, 120.0],
            'start': ['2024-01-01', '2024-04-01', '2024-07-01'],
            'end': ['2024-03-31', '2024-06-30', '2024-09-30'],
            'frame': ['CY2024Q1', 'CY2024Q2', 'CY2024Q3']
        })

        result = compute_ttm_long(df, duration_concepts={'rev'})

        # Should be empty (need at least 4 quarters for TTM)
        assert result.is_empty()

    def test_rolling_ttm(self):
        """Test rolling TTM computation over multiple periods"""
        # 6 quarters of data
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 6,
            'as_of_date': [
                '2024-03-31', '2024-06-30', '2024-09-30',
                '2024-12-31', '2025-03-31', '2025-06-30'
            ],
            'accn': [f'acc{i}' for i in range(1, 7)],
            'form': ['10-Q'] * 6,
            'concept': ['rev'] * 6,
            'value': [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            'start': [
                '2024-01-01', '2024-04-01', '2024-07-01',
                '2024-10-01', '2025-01-01', '2025-04-01'
            ],
            'end': [
                '2024-03-31', '2024-06-30', '2024-09-30',
                '2024-12-31', '2025-03-31', '2025-06-30'
            ],
            'frame': ['CY2024Q1', 'CY2024Q2', 'CY2024Q3', 'CY2024Q4', 'CY2025Q1', 'CY2025Q2']
        })

        result = compute_ttm_long(df, duration_concepts={'rev'})

        # Should have 3 TTM values (for quarters 4, 5, 6)
        assert len(result) == 3

        # First TTM (Q4 2024): Q1+Q2+Q3+Q4 = 100+110+120+130 = 460
        ttm1 = result.filter(pl.col('as_of_date') == '2024-12-31')
        assert ttm1['value'][0] == 460.0

        # Second TTM (Q1 2025): Q2+Q3+Q4+Q1 = 110+120+130+140 = 500
        ttm2 = result.filter(pl.col('as_of_date') == '2025-03-31')
        assert ttm2['value'][0] == 500.0

        # Third TTM (Q2 2025): Q3+Q4+Q1+Q2 = 120+130+140+150 = 540
        ttm3 = result.filter(pl.col('as_of_date') == '2025-06-30')
        assert ttm3['value'][0] == 540.0

    def test_null_values_in_window(self):
        """Test that windows with null values are skipped"""
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 4,
            'as_of_date': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
            'accn': ['acc1', 'acc2', 'acc3', 'acc4'],
            'form': ['10-Q'] * 4,
            'concept': ['rev'] * 4,
            'value': [100.0, None, 120.0, 130.0],  # Q2 is None
            'start': ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'],
            'end': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
            'frame': ['CY2024Q1', 'CY2024Q2', 'CY2024Q3', 'CY2024Q4']
        })

        result = compute_ttm_long(df, duration_concepts={'rev'})

        # Should be empty because window has null value
        assert result.is_empty()

    def test_filter_by_duration_concepts(self):
        """Test that only duration concepts are processed"""
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 8,
            'as_of_date': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'] * 2,
            'accn': ['acc1', 'acc2', 'acc3', 'acc4'] * 2,
            'form': ['10-Q'] * 8,
            'concept': ['rev'] * 4 + ['ta'] * 4,  # rev is duration, ta is instant
            'value': [100.0, 110.0, 120.0, 130.0, 1000.0, 1100.0, 1200.0, 1300.0],
            'start': ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'] * 2,
            'end': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'] * 2,
            'frame': ['CY2024Q1', 'CY2024Q2', 'CY2024Q3', 'CY2024Q4'] * 2
        })

        # Only process 'rev' as duration concept
        result = compute_ttm_long(df, duration_concepts={'rev'})

        # Should only have TTM for rev, not ta
        assert len(result) == 1
        assert result['concept'][0] == 'rev'

    def test_preserves_metadata(self):
        """Test that TTM preserves accn, form, start, end, frame from latest quarter"""
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 4,
            'as_of_date': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
            'accn': ['acc1', 'acc2', 'acc3', 'acc4'],
            'form': ['10-Q', '10-Q', '10-Q', '10-K'],
            'concept': ['rev'] * 4,
            'value': [100.0, 110.0, 120.0, 130.0],
            'start': ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'],
            'end': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
            'frame': ['CY2024Q1', 'CY2024Q2', 'CY2024Q3', 'CY2024Q4']
        })

        result = compute_ttm_long(df, duration_concepts={'rev'})

        # Metadata should come from the latest quarter (Q4)
        assert result['accn'][0] == 'acc4'
        assert result['form'][0] == '10-K'
        assert result['frame'][0] == 'CY2024Q4'

        # TTM start should be from first quarter, end from last quarter
        assert result['start'][0] == '2024-01-01'
        assert result['end'][0] == '2024-12-31'

    def test_invalid_date_formats(self):
        """Test handling of invalid date formats"""
        df = pl.DataFrame({
            'symbol': ['AAPL'] * 4,
            'as_of_date': ['2024-03-31', 'invalid-date', '2024-09-30', '2024-12-31'],
            'accn': ['acc1', 'acc2', 'acc3', 'acc4'],
            'form': ['10-Q'] * 4,
            'concept': ['rev'] * 4,
            'value': [100.0, 110.0, 120.0, 130.0],
            'start': ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'],
            'end': ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
            'frame': ['CY2024Q1', 'CY2024Q2', 'CY2024Q3', 'CY2024Q4']
        })

        result = compute_ttm_long(df, duration_concepts={'rev'})

        # Invalid date should be skipped, not enough valid quarters for TTM
        # (Only 3 valid quarters: Q1, Q3, Q4)
        assert result.is_empty()
