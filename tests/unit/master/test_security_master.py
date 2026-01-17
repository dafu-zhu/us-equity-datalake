"""
Unit tests for master.security_master module
Tests symbol normalization and security master functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import polars as pl
import pandas as pd
import datetime as dt
import os
import io
import requests
from quantdl.master.security_master import SymbolNormalizer, SecurityMaster


class TestSymbolNormalizer:
    """Test SymbolNormalizer class"""

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_initialization(self, mock_logger, mock_fetch):
        """Test SymbolNormalizer initialization"""
        # Mock current stock list
        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', 'BRK.B', 'GOOGL', 'ABC.D']
        })
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Verify fetch was called
        mock_fetch.assert_called_once_with(with_filter=True, refresh=False)

        # Verify symbol map was created
        assert 'AAPL' in normalizer.sym_map
        assert 'BRKB' in normalizer.sym_map  # Dots removed
        assert 'GOOGL' in normalizer.sym_map
        assert 'ABCD' in normalizer.sym_map  # Dots removed

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_initialization_with_security_master(self, mock_logger, mock_fetch):
        """Test initialization with SecurityMaster instance"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        normalizer = SymbolNormalizer(security_master=mock_sm)

        assert normalizer.security_master == mock_sm

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_simple_match(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with simple symbol match"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Test uppercase conversion
        assert normalizer.to_nasdaq_format('aapl') == 'AAPL'
        assert normalizer.to_nasdaq_format('MSFT') == 'MSFT'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_separator(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with symbols containing separators"""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B', 'ABC-D']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # BRKB should normalize to BRK.B
        assert normalizer.to_nasdaq_format('BRKB') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BRK.B') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BRK-B') == 'BRK.B'

        # ABCD should normalize to ABC-D
        assert normalizer.to_nasdaq_format('ABCD') == 'ABC-D'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_not_in_current_list(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with delisted symbol not in current list"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'MSFT']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Symbol not in current list should return as-is (uppercased)
        assert normalizer.to_nasdaq_format('DELISTD') == 'DELISTD'
        assert normalizer.to_nasdaq_format('oldstock') == 'OLDSTOCK'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_validation_same_security(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with SecurityMaster validation (same security)"""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        # Mock same security_id for both dates
        mock_sm.get_security_id.return_value = 'security_123'

        normalizer = SymbolNormalizer(security_master=mock_sm)

        result = normalizer.to_nasdaq_format('BRKB', day='2024-01-01')

        # Should return Nasdaq format since same security
        assert result == 'BRK.B'

        # Verify get_security_id was called twice
        assert mock_sm.get_security_id.call_count == 2

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_validation_different_security(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with SecurityMaster validation (different security)"""
        mock_stocks = pl.DataFrame({'Ticker': ['ABC.D']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        # Mock different security_ids
        mock_sm.get_security_id.side_effect = ['security_old', 'security_new']

        normalizer = SymbolNormalizer(security_master=mock_sm)

        result = normalizer.to_nasdaq_format('ABCD', day='2022-01-01')

        # Should return original format since different security
        assert result == 'ABCD'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_validation_error(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format when validation raises error"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        mock_sm.get_security_id.side_effect = ValueError("Symbol not found")

        normalizer = SymbolNormalizer(security_master=mock_sm)

        # Should return original format when validation fails
        result = normalizer.to_nasdaq_format('AAPL', day='2024-01-01')
        assert result == 'AAPL'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_empty_symbol(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format with empty symbol"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('') == ''
        assert normalizer.to_nasdaq_format(None) is None

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_no_date_context(self, mock_logger, mock_fetch):
        """Test to_nasdaq_format without date context"""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        normalizer = SymbolNormalizer(security_master=mock_sm)

        result = normalizer.to_nasdaq_format('BRKB')

        # Should return Nasdaq format without validation
        assert result == 'BRK.B'

        # Verify get_security_id was NOT called
        mock_sm.get_security_id.assert_not_called()

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_batch_normalize(self, mock_logger, mock_fetch):
        """Test batch_normalize"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B', 'GOOGL']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        symbols = ['aapl', 'BRKB', 'googl']
        result = normalizer.batch_normalize(symbols)

        assert result == ['AAPL', 'BRK.B', 'GOOGL']

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_batch_normalize_with_date(self, mock_logger, mock_fetch):
        """Test batch_normalize with date context"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B']})
        mock_fetch.return_value = mock_stocks

        mock_sm = Mock()
        mock_sm.get_security_id.return_value = 'security_123'

        normalizer = SymbolNormalizer(security_master=mock_sm)

        symbols = ['AAPL', 'BRKB']
        result = normalizer.batch_normalize(symbols, day='2024-01-01')

        assert result == ['AAPL', 'BRK.B']

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_to_nasdaq_format_with_day_no_security_master(self, mock_logger, mock_fetch):
        """Date context without SecurityMaster uses Nasdaq format."""
        mock_stocks = pl.DataFrame({'Ticker': ['BRK.B']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer(security_master=None)

        assert normalizer.to_nasdaq_format('BRKB', day='2024-01-01') == 'BRK.B'

    def test_to_crsp_format(self):
        """Test to_crsp_format static method"""
        # Test various input formats
        assert SymbolNormalizer.to_crsp_format('BRK.B') == 'BRKB'
        assert SymbolNormalizer.to_crsp_format('BRK-B') == 'BRKB'
        assert SymbolNormalizer.to_crsp_format('ABC.D.E') == 'ABCDE'
        assert SymbolNormalizer.to_crsp_format('AAPL') == 'AAPL'
        assert SymbolNormalizer.to_crsp_format('aapl') == 'AAPL'

    def test_to_sec_format(self):
        """Test to_sec_format static method"""
        assert SymbolNormalizer.to_sec_format('BRK.B') == 'BRK-B'
        assert SymbolNormalizer.to_sec_format('aapl') == 'AAPL'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_handles_non_string_tickers(self, mock_logger, mock_fetch):
        """Test that initialization handles non-string tickers gracefully"""
        # Include NaN and None values
        mock_stocks = pl.DataFrame({
            'Ticker': ['AAPL', None, 'MSFT', 'GOOGL']
        })
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        # Should skip non-string values
        assert 'AAPL' in normalizer.sym_map
        assert 'MSFT' in normalizer.sym_map
        assert 'GOOGL' in normalizer.sym_map
        # None should not cause issues


class TestSymbolNormalizerEdgeCases:
    """Test edge cases for SymbolNormalizer"""

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_case_insensitivity(self, mock_logger, mock_fetch):
        """Test that symbol matching is case-insensitive"""
        mock_stocks = pl.DataFrame({'Ticker': ['AAPL', 'BRK.B']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('aapl') == 'AAPL'
        assert normalizer.to_nasdaq_format('AaPl') == 'AAPL'
        assert normalizer.to_nasdaq_format('brkb') == 'BRK.B'
        assert normalizer.to_nasdaq_format('BrK.b') == 'BRK.B'

    @patch('quantdl.master.security_master.fetch_all_stocks')
    @patch('quantdl.master.security_master.setup_logger')
    def test_multiple_separators(self, mock_logger, mock_fetch):
        """Test symbols with multiple separators"""
        mock_stocks = pl.DataFrame({'Ticker': ['A.B.C', 'X-Y-Z']})
        mock_fetch.return_value = mock_stocks

        normalizer = SymbolNormalizer()

        assert normalizer.to_nasdaq_format('ABC') == 'A.B.C'
        assert normalizer.to_nasdaq_format('XYZ') == 'X-Y-Z'


class TestSecurityMaster:
    """Test SecurityMaster core behaviors with injected data"""

    def test_get_security_id_exact_match(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.auto_resolve = Mock(return_value=999)

        result = sm.get_security_id('AAA', '2020-06-30', auto_resolve=True)

        assert result == 101
        sm.auto_resolve.assert_not_called()

    def test_get_security_id_no_match_no_auto_resolve(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="Symbol BBB not found"):
            sm.get_security_id('BBB', '2020-06-30', auto_resolve=False)

    def test_get_security_id_no_match_auto_resolve(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.auto_resolve = Mock(return_value=555)

        result = sm.get_security_id('BBB', '2020-06-30', auto_resolve=True)

        assert result == 555
        sm.auto_resolve.assert_called_once_with('BBB', '2020-06-30')

    def test_get_security_id_exact_match_no_auto_resolve_flag(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.auto_resolve = Mock(return_value=999)

        result = sm.get_security_id('AAA', '2020-06-30', auto_resolve=False)

        assert result == 101
        sm.auto_resolve.assert_not_called()

    def test_get_security_id_rejects_none_security_id(self):
        master_tb = pl.DataFrame({
            'security_id': [None],
            'symbol': ['AAA'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="security_id is None"):
            sm.get_security_id('AAA', '2020-06-30', auto_resolve=False)

    def test_auto_resolve_selects_closest_symbol_usage(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2],
            'symbol': ['AAA', 'BBB', 'AAA'],
            'company': ['OldCo', 'OldCo', 'NewCo'],
            'start_date': [dt.date(2010, 1, 1), dt.date(2018, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2010, 12, 31), dt.date(2022, 12, 31), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        result = sm.auto_resolve('AAA', '2020-06-15')

        assert result == 2

    def test_auto_resolve_symbol_never_existed(self):
        master_tb = pl.DataFrame({
            'security_id': [1],
            'symbol': ['AAA'],
            'start_date': [dt.date(2010, 1, 1)],
            'end_date': [dt.date(2010, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="never existed"):
            sm.auto_resolve('ZZZ', '2020-06-15')

    def test_sid_to_permno(self):
        """Test sid_to_permno uses master_tb directly (works when loaded from S3)"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = pl.DataFrame({
            'security_id': [101, 102],
            'permno': [555, 666],
            'symbol': ['AAA', 'BBB'],
            'company': ['AAA Corp', 'BBB Corp'],
            'cik': ['0001', '0002'],
            'cusip': ['11111111', '22222222'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })

        result = sm.sid_to_permno(101)

        assert result == 555
    
    def test_sid_to_permno_none(self):
        sm = SecurityMaster.__new__(SecurityMaster)

        with pytest.raises(ValueError, match="security_id is None"):
            sm.sid_to_permno(None)

    def test_sid_to_info(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'company': ['TestCo'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb

        result = sm.sid_to_info(101, '2020-06-30', info='company')

        assert result == 'TestCo'

    def test_get_symbol_history(self):
        master_tb = pl.DataFrame({
            'security_id': [101, 101],
            'symbol': ['AAA', 'BBB'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_table = Mock(return_value=master_tb)

        result = sm.get_symbol_history(101)

        assert set(result) == {
            ('AAA', '2020-01-01', '2020-12-31'),
            ('BBB', '2021-01-01', '2021-12-31')
        }

    def test_auto_resolve_filters_null_security_ids(self):
        master_tb = pl.DataFrame({
            'security_id': [None, 101],
            'symbol': ['AAA', 'AAA'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        result = sm.auto_resolve('AAA', '2020-06-30')

        assert result == 101
        sm.logger.warning.assert_not_called()

    def test_auto_resolve_logs_error_on_sid_to_info_failure(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'company': ['TestCo'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(side_effect=RuntimeError("boom"))

        result = sm.auto_resolve('AAA', '2020-06-30')

        assert result == 101
        sm.logger.error.assert_called_once()

    def test_auto_resolve_no_active_security(self):
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAA'],
            'start_date': [dt.date(2010, 1, 1)],
            'end_date': [dt.date(2010, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        with pytest.raises(ValueError, match="was not active"):
            sm.auto_resolve('AAA', '2020-06-30')

    def test_auto_resolve_multiple_candidates_selects_closest(self):
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2],
            'symbol': ['AAA', 'BBB', 'AAA'],
            'start_date': [dt.date(2010, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 6, 1)],
            'end_date': [dt.date(2010, 12, 31), dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        result = sm.auto_resolve('AAA', '2020-06-15')

        assert result == 2
        sm.logger.info.assert_called()

    def test_fetch_sec_cik_mapping_cached(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        cached = pl.DataFrame({'ticker': ['AAA'], 'cik': ['0000000001']})
        sm._sec_cik_cache = cached

        with patch('quantdl.master.security_master.requests.get') as mock_get:
            result = sm._fetch_sec_cik_mapping()

        assert result is cached
        mock_get.assert_not_called()

    def test_fetch_sec_cik_mapping_success(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._sec_cik_cache = None

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "BRK.B", "cik_str": 123456},
            "1": {"ticker": "AAPL", "cik_str": 320193}
        }

        with patch('quantdl.master.security_master.requests.get', return_value=mock_response) as mock_get:
            result = sm._fetch_sec_cik_mapping()

        assert not result.is_empty()
        assert sm._sec_cik_cache is result
        mock_get.assert_called_once()

    def test_fetch_sec_cik_mapping_failure(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._sec_cik_cache = None

        with patch('quantdl.master.security_master.requests.get', side_effect=Exception("boom")):
            result = sm._fetch_sec_cik_mapping()

        assert result.is_empty()
        assert set(result.columns) == {"ticker", "cik"}

    def test_fetch_sec_cik_mapping_filters_zero_cik(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm._sec_cik_cache = None

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "0": {"ticker": "ZERO", "cik_str": 0},
            "1": {"ticker": "AAPL", "cik_str": 320193}
        }

        with patch('quantdl.master.security_master.requests.get', return_value=mock_response):
            result = sm._fetch_sec_cik_mapping()

        tickers = result['ticker'].to_list()
        assert "ZERO" not in tickers
        assert "AAPL" in tickers

    def test_cik_cusip_mapping_sec_fallback_unavailable(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()

        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': [None],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        sm._fetch_sec_cik_mapping = Mock(return_value=pl.DataFrame({'ticker': [], 'cik': []}))

        result = sm.cik_cusip_mapping()

        assert result.filter(pl.col('cik').is_null()).height == 1
        sm.logger.warning.assert_called()

    def test_cik_cusip_mapping_sec_fallback_fills_nulls(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()

        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1, 2],
            'ticker': ['AAA', 'BBB'],
            'tsymbol': ['AAA', 'BBB'],
            'comnam': ['AAA Corp', 'BBB Corp'],
            'ncusip': ['12345678', '87654321'],
            'cik': [None, '0000000002'],
            'cikdate1': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31'), pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31'), pd.Timestamp('2020-12-31')]
        })

        sm._fetch_sec_cik_mapping = Mock(return_value=pl.DataFrame({
            'ticker': ['AAA'],
            'cik': ['0000000001']
        }))

        result = sm.cik_cusip_mapping()

        filled = result.filter(pl.col('symbol') == 'AAA').select('cik').item()
        assert filled == '0000000001'

    def test_cik_cusip_mapping_no_fallback_needed(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()
        sm._fetch_sec_cik_mapping = Mock()

        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        result = sm.cik_cusip_mapping()

        assert result.filter(pl.col('cik').is_null()).height == 0
        sm._fetch_sec_cik_mapping.assert_not_called()

    def test_security_map_new_business_on_symbol_change(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.cik_cusip = pl.DataFrame({
            'permno': [1, 1],
            'symbol': ['AAA', 'BBB'],
            'company': ['AAA Corp', 'BBB Corp'],
            'cik': ['0001', '0002'],
            'cusip': ['11111111', '22222222'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })

        result = sm.security_map()

        sec_ids = result.select('security_id').unique().to_series().to_list()
        assert len(sec_ids) == 2

    def test_security_map_same_business_with_cik_overlap(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.cik_cusip = pl.DataFrame({
            'permno': [1, 1],
            'symbol': ['AAA', 'BBB'],
            'company': ['AAA Corp', 'BBB Corp'],
            'cik': ['0001', '0001'],
            'cusip': ['11111111', '22222222'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })

        result = sm.security_map()

        sec_ids = result.select('security_id').unique().to_series().to_list()
        assert len(sec_ids) == 1

    def test_security_map_new_business_on_permno_change(self):
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.cik_cusip = pl.DataFrame({
            'permno': [1, 2],
            'symbol': ['AAA', 'AAA'],
            'company': ['AAA Corp', 'AAA Corp'],
            'cik': ['0001', '0001'],
            'cusip': ['11111111', '11111111'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2021, 12, 31)]
        })

        result = sm.security_map()

        sec_ids = result.select('security_id').unique().to_series().to_list()
        assert len(sec_ids) == 2

    def test_master_table_includes_security_id_and_permno(self):
        """Test master_table includes both security_id and permno columns"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.cik_cusip = pl.DataFrame({
            'permno': [1],
            'symbol': ['AAA'],
            'company': ['AAA Corp'],
            'cik': ['0001'],
            'cusip': ['11111111'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm.security_map = Mock(return_value=pl.DataFrame({
            'security_id': [101],
            'permno': [1],
            'symbol': ['AAA'],
            'company': ['AAA Corp'],
            'cik': ['0001'],
            'cusip': ['11111111'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        }))

        result = sm.master_table()

        assert 'security_id' in result.columns
        assert 'permno' in result.columns
        assert result.select('security_id').item() == 101
        assert result.select('permno').item() == 1

    def test_master_table_preserves_security_id_and_permno_with_null_cik(self):
        """Test master_table preserves security_id and permno even with null cik"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.cik_cusip = pl.DataFrame({
            'permno': [1],
            'symbol': ['AAC'],
            'company': ['AAC Corp'],
            'cik': [None],
            'cusip': ['12345678'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2009, 12, 31)]
        })
        sm.security_map = Mock(return_value=pl.DataFrame({
            'security_id': [101],
            'permno': [1],
            'symbol': ['AAC'],
            'company': ['AAC Corp'],
            'cik': [None],
            'cusip': ['12345678'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2009, 12, 31)]
        }))

        result = sm.master_table()

        assert result.select('security_id').item() == 101
        assert result.select('permno').item() == 1


class TestSecurityMasterInitialization:
    """Test SecurityMaster initialization with different configurations"""

    @patch('quantdl.master.security_master.wrds.Connection')
    @patch('quantdl.master.security_master.setup_logger')
    @patch.dict(os.environ, {'WRDS_USERNAME': 'test_user', 'WRDS_PASSWORD': 'test_pass'})
    def test_init_without_db_connection(self, mock_logger, mock_wrds):
        """Test initialization without providing db connection (creates new connection)"""
        # Mock the connection
        mock_db_instance = Mock()
        mock_wrds.return_value = mock_db_instance

        # Mock the methods that are called during initialization
        mock_db_instance.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=mock_db_instance.raw_sql.return_value):
            sm = SecurityMaster(db=None)

        # Verify connection was created with env variables
        mock_wrds.assert_called_once_with(
            wrds_username='test_user',
            wrds_password='test_pass'
        )
        assert sm.db == mock_db_instance

    @patch('quantdl.master.security_master.setup_logger')
    def test_init_with_db_connection(self, mock_logger):
        """Test initialization with provided db connection"""
        mock_db = Mock()

        # Mock the methods that are called during initialization
        mock_db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=mock_db.raw_sql.return_value):
            sm = SecurityMaster(db=mock_db)

        # Verify provided db connection was used
        assert sm.db == mock_db

    @patch('quantdl.master.security_master.setup_logger')
    def test_init_creates_logger(self, mock_logger):
        """Test that initialization creates a logger"""
        mock_db = Mock()
        mock_db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=mock_db.raw_sql.return_value):
            sm = SecurityMaster(db=mock_db)

        # Verify logger setup was called
        mock_logger.assert_called_once()
        assert sm.logger is not None

    @patch('quantdl.master.security_master.setup_logger')
    def test_init_sets_sec_cik_cache_to_none(self, mock_logger):
        """Test that _sec_cik_cache is initialized to None"""
        mock_db = Mock()
        mock_db.raw_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAA'],
            'tsymbol': ['AAA'],
            'comnam': ['AAA Corp'],
            'ncusip': ['12345678'],
            'cik': ['0000000001'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2020-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2020-12-31')]
        })

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=mock_db.raw_sql.return_value):
            sm = SecurityMaster(db=mock_db)

        # Verify cache is initialized to None
        assert sm._sec_cik_cache is None


class TestSecurityMasterCikCusipMapping:
    """Test SecurityMaster cik_cusip_mapping edge cases"""

    def test_cik_cusip_mapping_logs_null_symbols_when_more_than_50(self):
        """Test logging when more than 50 null CIK symbols exist"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()

        # Create data with many null CIKs (60 symbols)
        symbols = [f'SYM{i:03d}' for i in range(60)]
        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': list(range(1, 61)),
            'ticker': symbols,
            'tsymbol': symbols,
            'comnam': [f'{sym} Corp' for sym in symbols],
            'ncusip': [f'{i:08d}' for i in range(1, 61)],
            'cik': [None] * 60,
            'cikdate1': [pd.Timestamp('2020-01-01')] * 60,
            'cikdate2': [pd.Timestamp('2020-12-31')] * 60,
            'namedt': [pd.Timestamp('2020-01-01')] * 60,
            'nameenddt': [pd.Timestamp('2020-12-31')] * 60
        })

        # SEC fallback returns non-empty to trigger the logging path
        sm._fetch_sec_cik_mapping = Mock(return_value=pl.DataFrame({'ticker': ['XYZ'], 'cik': ['0000000999']}))

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=sm.db.raw_sql.return_value):
            result = sm.cik_cusip_mapping()

        # Verify logging was called - should log "... and X more (see detailed log below)"
        # since we have 60 unique symbols with NULL CIK
        log_calls = [str(call) for call in sm.logger.info.call_args_list]
        assert any('more (see detailed log below)' in str(call) for call in log_calls)

    def test_cik_cusip_mapping_logs_detailed_examples_when_more_than_20(self):
        """Test detailed logging when more than 20 null CIK records exist"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = Mock()

        # Create data with many null CIKs (30 records)
        symbols = [f'SYM{i:03d}' for i in range(30)]
        sm.db.raw_sql.return_value = pd.DataFrame({
            'kypermno': list(range(1, 31)),
            'ticker': symbols,
            'tsymbol': symbols,
            'comnam': [f'{sym} Corp' for sym in symbols],
            'ncusip': [f'{i:08d}' for i in range(1, 31)],
            'cik': [None] * 30,
            'cikdate1': [pd.Timestamp('2020-01-01')] * 30,
            'cikdate2': [pd.Timestamp('2020-12-31')] * 30,
            'namedt': [pd.Timestamp('2020-01-01')] * 30,
            'nameenddt': [pd.Timestamp('2020-12-31')] * 30
        })

        # SEC fallback returns non-empty to trigger the logging path
        sm._fetch_sec_cik_mapping = Mock(return_value=pl.DataFrame({'ticker': ['XYZ'], 'cik': ['0000000999']}))

        with patch('quantdl.master.security_master.raw_sql_with_retry', return_value=sm.db.raw_sql.return_value):
            result = sm.cik_cusip_mapping()

        # Verify detailed logging was done - should log "... and X more records"
        log_calls = [str(call) for call in sm.logger.info.call_args_list]
        assert any('more records' in str(call) for call in log_calls)


class TestSecurityMasterAutoResolve:
    """Test SecurityMaster auto_resolve edge cases"""

    def test_auto_resolve_with_null_candidates_defensive_code(self):
        """Test defensive null-checking code in auto_resolve (lines 544-546, 565).

        Note: These lines are defensive code that would handle null candidates.
        In normal execution, line 533 filters nulls, making this code unreachable.
        This test verifies the overall behavior when master_tb contains nulls.
        """
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        # Create a master_tb with null security_ids
        master_tb = pl.DataFrame({
            'security_id': [None, None, 101],
            'symbol': ['AAA', 'AAA', 'AAA'],
            'company': ['AAA Corp 1', 'AAA Corp 2', 'AAA Corp 3'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 6, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })
        sm.master_tb = master_tb

        # The auto_resolve method filters nulls at line 533 before the loop
        # So it will only see security_id=101
        result = sm.auto_resolve('AAA', '2020-06-30')

        # Should successfully resolve to the non-null candidate
        assert result == 101
        # Warning should NOT be called because nulls were filtered before the loop
        sm.logger.warning.assert_not_called()

    def test_auto_resolve_verifies_null_filter_works(self):
        """Verify that lines 544-546 and 565 are defensive code (currently unreachable).

        Lines 544-546 check `if candidate_sid is None` and increment null_candidates.
        Line 565 logs a warning if null_candidates > 0.

        These lines are unreachable because line 533 filters out nulls before the loop:
        .filter(pl.col('security_id').is_not_null())

        This test documents that the defensive code exists but cannot be reached
        in normal execution due to the earlier filter.
        """
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        # Create master_tb with mixed null and valid security_ids
        master_tb = pl.DataFrame({
            'security_id': [None, None, 101],
            'symbol': ['AAA', 'AAA', 'AAA'],
            'company': ['AAA Corp', 'AAA Corp', 'AAA Corp'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })
        sm.master_tb = master_tb

        # The filter at line 533 will remove nulls before the loop at line 543
        result = sm.auto_resolve('AAA', '2020-06-30')

        # Should resolve to the only non-null security
        assert result == 101

        # Verify the warning was NOT logged (because nulls were filtered before the loop)
        # This confirms lines 544-546 and 565 were not executed
        sm.logger.warning.assert_not_called()

        # Verify that candidates were properly filtered by checking the intermediate result
        # The candidates query (lines 529-534) should only return non-null security_ids
        candidates = (
            master_tb.filter(pl.col('symbol').eq('AAA'))
            .select('security_id')
            .unique()
            .filter(pl.col('security_id').is_not_null())
        )
        # Should only have one candidate (101), nulls filtered out
        assert candidates.height == 1
        assert candidates['security_id'][0] == 101

    def test_auto_resolve_date_before_symbol_start(self):
        """Test auto_resolve when query date is before symbol start date"""
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2, 2],
            'symbol': ['BBB', 'AAA', 'CCC', 'AAA'],
            'company': ['OldCo', 'OldCo', 'NewCo', 'NewCo'],
            'start_date': [dt.date(2018, 1, 1), dt.date(2020, 1, 1), dt.date(2018, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2019, 12, 31), dt.date(2020, 12, 31), dt.date(2022, 12, 31), dt.date(2022, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        # Query date is before AAA was used by security_id=1 (2020-01-01)
        # Security 1 was active 2018-2020, used AAA in 2020
        # Security 2 was active 2018-2022, used AAA in 2021
        result = sm.auto_resolve('AAA', '2019-06-15')

        # Should pick security_id=1 (distance = 200 days to future use)
        # vs security_id=2 (distance = 565 days to future use)
        assert result == 1

    def test_auto_resolve_date_after_symbol_end(self):
        """Test auto_resolve when query date is after symbol end date"""
        master_tb = pl.DataFrame({
            'security_id': [1, 1, 2, 2],
            'symbol': ['AAA', 'BBB', 'AAA', 'CCC'],
            'company': ['OldCo', 'OldCo', 'NewCo', 'NewCo'],
            'start_date': [dt.date(2018, 1, 1), dt.date(2020, 1, 1), dt.date(2019, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2019, 12, 31), dt.date(2020, 12, 31), dt.date(2019, 6, 30), dt.date(2020, 12, 31)]
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.sid_to_info = Mock(return_value="info")

        # Query date is after AAA was used by both securities
        # Security 1: used AAA until 2019-12-31, active until 2020-12-31
        # Security 2: used AAA until 2019-06-30, active until 2020-12-31
        result = sm.auto_resolve('AAA', '2020-06-15')

        # Should pick security_id=1 (distance = 167 days from 2019-12-31)
        # vs security_id=2 (distance = 351 days from 2019-06-30)
        assert result == 1


class TestSecurityMasterClose:
    """Test SecurityMaster close method"""

    def test_close_calls_db_close(self):
        """Test that close method calls db.close()"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.db = Mock()

        sm.close()

        sm.db.close.assert_called_once()


class TestSecurityMasterS3Operations:
    """Test SecurityMaster S3 loading and export functionality"""

    @patch('quantdl.master.security_master.setup_logger')
    def test_load_from_s3_success(self, mock_logger):
        """Test successful S3 load with valid metadata"""
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq

        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        # Create mock S3 client and response
        mock_s3 = Mock()

        # Create test data with metadata (including permno for sid_to_permno support)
        test_df = pl.DataFrame({
            'security_id': [101],
            'permno': [555],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })

        # Convert to Arrow and add metadata
        table = test_df.to_arrow()
        metadata = {
            b'crsp_end_date': b'2024-12-31',
            b'export_timestamp': b'2024-01-01T00:00:00',
            b'version': b'1.0',
            b'row_count': b'1'
        }
        table = table.replace_schema_metadata(metadata)

        # Write to buffer
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        # Mock S3 response
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=buffer.read()))
        }

        # Test load
        result_df, result_metadata = sm._load_from_s3(mock_s3, 'test-bucket', 'test-key')

        # Verify
        assert len(result_df) == 1
        assert result_metadata['crsp_end_date'] == '2024-12-31'
        assert result_metadata['version'] == '1.0'
        mock_s3.get_object.assert_called_once_with(Bucket='test-bucket', Key='test-key')

    @patch('quantdl.master.security_master.setup_logger')
    def test_load_from_s3_no_metadata(self, mock_logger):
        """Test S3 load when metadata is missing"""
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq

        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_s3 = Mock()

        # Create test data without metadata (including permno)
        test_df = pl.DataFrame({
            'security_id': [101],
            'permno': [555],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })

        table = test_df.to_arrow()
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=buffer.read()))
        }

        result_df, result_metadata = sm._load_from_s3(mock_s3, 'test-bucket', 'test-key')

        assert len(result_df) == 1
        assert result_metadata == {}

    @patch('quantdl.master.security_master.setup_logger')
    def test_load_from_s3_failure(self, mock_logger):
        """Test S3 load failure raises exception"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_s3 = Mock()
        mock_s3.get_object.side_effect = Exception("S3 error")

        with pytest.raises(Exception, match="S3 error"):
            sm._load_from_s3(mock_s3, 'test-bucket', 'test-key')

    @patch('quantdl.master.security_master.setup_logger')
    def test_export_to_s3_success(self, mock_logger):
        """Test successful export to S3 with metadata"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.CRSP_LATEST_DATE = '2024-12-31'

        # Create test master_tb (including permno)
        sm.master_tb = pl.DataFrame({
            'security_id': [101, 102],
            'permno': [555, 666],
            'symbol': ['AAPL', 'MSFT'],
            'company': ['Apple Inc', 'Microsoft'],
            'cik': ['0000320193', '0000789019'],
            'cusip': ['037833100', '594918104'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })

        mock_s3 = Mock()

        result = sm.export_to_s3(mock_s3, 'test-bucket', 'test-key')

        # Verify upload was called
        mock_s3.upload_fileobj.assert_called_once()
        call_args = mock_s3.upload_fileobj.call_args

        # Verify bucket and key
        assert call_args[0][1] == 'test-bucket'
        assert call_args[0][2] == 'test-key'

        # Verify result
        assert result['status'] == 'success'
        assert 'export_timestamp' in result

    @patch('quantdl.master.security_master.setup_logger')
    def test_export_to_s3_includes_metadata(self, mock_logger):
        """Test export includes correct metadata in Parquet file"""
        import io
        import pyarrow.parquet as pq

        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.CRSP_LATEST_DATE = '2024-12-31'

        sm.master_tb = pl.DataFrame({
            'security_id': [101],
            'permno': [555],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })

        # Capture uploaded data
        uploaded_buffer = None
        def capture_upload(buffer, bucket, key):
            nonlocal uploaded_buffer
            uploaded_buffer = io.BytesIO(buffer.read())

        mock_s3 = Mock()
        mock_s3.upload_fileobj.side_effect = capture_upload

        sm.export_to_s3(mock_s3, 'test-bucket', 'test-key')

        # Read back and verify metadata
        uploaded_buffer.seek(0)
        table = pq.read_table(uploaded_buffer)

        assert table.schema.metadata is not None
        assert b'crsp_end_date' in table.schema.metadata
        assert table.schema.metadata[b'crsp_end_date'] == b'2024-12-31'
        assert b'version' in table.schema.metadata
        assert table.schema.metadata[b'version'] == b'1.0'
        assert b'row_count' in table.schema.metadata
        assert table.schema.metadata[b'row_count'] == b'1'

    @patch('quantdl.master.security_master.setup_logger')
    @patch('quantdl.master.security_master.raw_sql_with_retry')
    def test_init_s3_fast_path_success(self, mock_sql, mock_logger):
        """Test initialization with S3 fast path (lines 212-230)"""
        import io
        import pyarrow.parquet as pq

        # Create test data with valid metadata (including permno)
        test_df = pl.DataFrame({
            'security_id': [101],
            'permno': [555],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })

        table = test_df.to_arrow()
        metadata = {
            b'crsp_end_date': b'2024-12-31',
            b'export_timestamp': b'2024-01-01T00:00:00',
            b'version': b'1.0',
            b'row_count': b'1'
        }
        table = table.replace_schema_metadata(metadata)

        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        mock_s3 = Mock()
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=buffer.read()))
        }

        # Initialize with S3 client
        sm = SecurityMaster(
            db=None,
            s3_client=mock_s3,
            bucket_name='test-bucket',
            s3_key='test-key',
            force_rebuild=False
        )

        # Verify S3 was used (fast path)
        assert sm._from_s3 is True
        assert len(sm.master_tb) == 1
        assert sm.cik_cusip is None  # Not needed when loaded from S3
        mock_s3.get_object.assert_called_once()
        mock_sql.assert_not_called()  # Should not query WRDS

    @patch('quantdl.master.security_master.setup_logger')
    @patch('quantdl.master.security_master.raw_sql_with_retry')
    def test_init_s3_missing_permno_fallback(self, mock_sql, mock_logger):
        """Test fallback to WRDS when S3 data missing permno column (schema mismatch)"""
        import io
        import pyarrow.parquet as pq

        # Create test data WITHOUT permno column (old schema)
        test_df = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })

        table = test_df.to_arrow()
        metadata = {
            b'crsp_end_date': b'2024-12-31',
            b'version': b'1.0'
        }
        table = table.replace_schema_metadata(metadata)

        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        mock_s3 = Mock()
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=buffer.read()))
        }

        # Mock WRDS data
        mock_db = Mock()
        mock_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAPL'],
            'tsymbol': ['AAPL'],
            'comnam': ['Apple Inc'],
            'ncusip': ['037833100'],
            'cik': ['0000320193'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2024-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2024-12-31')]
        })

        # Initialize - should detect missing permno and fallback to WRDS
        sm = SecurityMaster(
            db=mock_db,
            s3_client=mock_s3,
            bucket_name='test-bucket',
            s3_key='test-key',
            force_rebuild=False
        )

        # Verify fallback occurred
        assert sm._from_s3 is False
        assert 'permno' in sm.master_tb.columns  # Built from WRDS has permno
        mock_sql.assert_called()

    @patch('quantdl.master.security_master.setup_logger')
    @patch('quantdl.master.security_master.raw_sql_with_retry')
    def test_init_s3_stale_metadata_fallback(self, mock_sql, mock_logger):
        """Test initialization falls back to WRDS when S3 metadata is stale (lines 216-221)"""
        import io
        import pyarrow.parquet as pq

        # Create test data with STALE metadata (including permno)
        test_df = pl.DataFrame({
            'security_id': [101],
            'permno': [555],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })

        table = test_df.to_arrow()
        metadata = {
            b'crsp_end_date': b'2023-12-31',  # Stale date!
            b'version': b'1.0'
        }
        table = table.replace_schema_metadata(metadata)

        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        mock_s3 = Mock()
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=buffer.read()))
        }

        # Mock WRDS data
        mock_db = Mock()
        mock_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAPL'],
            'tsymbol': ['AAPL'],
            'comnam': ['Apple Inc'],
            'ncusip': ['037833100'],
            'cik': ['0000320193'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2024-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2024-12-31')]
        })

        # Initialize with S3 client (should fallback to WRDS)
        sm = SecurityMaster(
            db=mock_db,
            s3_client=mock_s3,
            bucket_name='test-bucket',
            s3_key='test-key',
            force_rebuild=False
        )

        # Verify fallback to WRDS occurred
        assert sm._from_s3 is False
        mock_sql.assert_called()  # Should query WRDS
        assert sm.cik_cusip is not None  # Built from WRDS

    @patch('quantdl.master.security_master.setup_logger')
    @patch('quantdl.master.security_master.raw_sql_with_retry')
    def test_init_s3_load_failure_fallback(self, mock_sql, mock_logger):
        """Test initialization falls back to WRDS when S3 load fails (lines 232-233)"""
        mock_s3 = Mock()
        mock_s3.get_object.side_effect = Exception("S3 connection error")

        # Mock WRDS data
        mock_db = Mock()
        mock_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAPL'],
            'tsymbol': ['AAPL'],
            'comnam': ['Apple Inc'],
            'ncusip': ['037833100'],
            'cik': ['0000320193'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2024-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2024-12-31')]
        })

        # Initialize with S3 client (should fallback to WRDS)
        sm = SecurityMaster(
            db=mock_db,
            s3_client=mock_s3,
            bucket_name='test-bucket',
            s3_key='test-key',
            force_rebuild=False
        )

        # Verify fallback occurred
        assert sm._from_s3 is False
        mock_sql.assert_called()

    @patch('quantdl.master.security_master.wrds.Connection')
    @patch('quantdl.master.security_master.setup_logger')
    @patch.dict(os.environ, {}, clear=True)  # Clear env vars
    def test_init_missing_wrds_credentials_no_s3(self, mock_logger, mock_wrds):
        """Test initialization raises ValueError when WRDS credentials missing (line 240)"""
        with pytest.raises(ValueError, match="WRDS credentials not found"):
            SecurityMaster(db=None, s3_client=None, force_rebuild=False)

    @patch('quantdl.master.security_master.setup_logger')
    @patch('quantdl.master.security_master.raw_sql_with_retry')
    def test_init_auto_export_to_s3_after_wrds_build(self, mock_sql, mock_logger):
        """Test auto-export to S3 after building from WRDS (lines 258-262)"""
        mock_db = Mock()
        mock_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAPL'],
            'tsymbol': ['AAPL'],
            'comnam': ['Apple Inc'],
            'ncusip': ['037833100'],
            'cik': ['0000320193'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2024-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2024-12-31')]
        })

        mock_s3 = Mock()

        # Initialize WITHOUT S3 data (build from WRDS)
        sm = SecurityMaster(
            db=mock_db,
            s3_client=mock_s3,
            bucket_name='test-bucket',
            s3_key='test-key',
            force_rebuild=True  # Force rebuild from WRDS
        )

        # Verify auto-export was called
        mock_s3.upload_fileobj.assert_called_once()
        call_args = mock_s3.upload_fileobj.call_args
        assert call_args[0][1] == 'test-bucket'
        assert call_args[0][2] == 'test-key'

    @patch('quantdl.master.security_master.setup_logger')
    @patch('quantdl.master.security_master.raw_sql_with_retry')
    def test_init_auto_export_failure_logged(self, mock_sql, mock_logger):
        """Test auto-export failure is logged as warning (line 262)"""
        mock_db = Mock()
        mock_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAPL'],
            'tsymbol': ['AAPL'],
            'comnam': ['Apple Inc'],
            'ncusip': ['037833100'],
            'cik': ['0000320193'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2024-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2024-12-31')]
        })

        mock_s3 = Mock()
        mock_s3.upload_fileobj.side_effect = Exception("Upload failed")

        # Initialize - should not raise, only log warning
        sm = SecurityMaster(
            db=mock_db,
            s3_client=mock_s3,
            bucket_name='test-bucket',
            s3_key='test-key',
            force_rebuild=True
        )

        # Verify warning was logged (check via logger mock)
        # Note: We're verifying it doesn't crash, logging check requires logger fixture
        assert sm._from_s3 is False
        assert sm.master_tb is not None

    @patch('quantdl.master.security_master.setup_logger')
    @patch('quantdl.master.security_master.raw_sql_with_retry')
    def test_init_force_rebuild_skips_s3(self, mock_sql, mock_logger):
        """Test force_rebuild=True skips S3 loading"""
        mock_db = Mock()
        mock_sql.return_value = pd.DataFrame({
            'kypermno': [1],
            'ticker': ['AAPL'],
            'tsymbol': ['AAPL'],
            'comnam': ['Apple Inc'],
            'ncusip': ['037833100'],
            'cik': ['0000320193'],
            'cikdate1': [pd.Timestamp('2020-01-01')],
            'cikdate2': [pd.Timestamp('2024-12-31')],
            'namedt': [pd.Timestamp('2020-01-01')],
            'nameenddt': [pd.Timestamp('2024-12-31')]
        })

        mock_s3 = Mock()

        # Initialize with force_rebuild
        sm = SecurityMaster(
            db=mock_db,
            s3_client=mock_s3,
            force_rebuild=True  # Should skip S3
        )

        # Verify S3 get_object was NOT called
        mock_s3.get_object.assert_not_called()
        assert sm._from_s3 is False
        mock_sql.assert_called()  # Built from WRDS

    @patch('quantdl.master.security_master.setup_logger')
    def test_export_to_s3_preserves_row_count_in_metadata(self, mock_logger):
        """Test export includes correct row count in metadata"""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.CRSP_LATEST_DATE = '2024-12-31'

        # Create larger dataset (including permno)
        sm.master_tb = pl.DataFrame({
            'security_id': list(range(1, 101)),
            'permno': list(range(1000, 1100)),
            'symbol': [f'SYM{i:03d}' for i in range(1, 101)],
            'company': [f'Company {i}' for i in range(1, 101)],
            'cik': [f'{i:010d}' for i in range(1, 101)],
            'cusip': [f'{i:08d}' for i in range(1, 101)],
            'start_date': [dt.date(2020, 1, 1)] * 100,
            'end_date': [dt.date(2020, 12, 31)] * 100
        })

        # Capture uploaded data
        uploaded_buffer = None
        def capture_upload(buffer, bucket, key):
            nonlocal uploaded_buffer
            uploaded_buffer = io.BytesIO(buffer.read())

        mock_s3 = Mock()
        mock_s3.upload_fileobj.side_effect = capture_upload

        result = sm.export_to_s3(mock_s3, 'test-bucket', 'test-key')

        # Verify row count in metadata
        import pyarrow.parquet as pq
        uploaded_buffer.seek(0)
        table = pq.read_table(uploaded_buffer)
        assert table.schema.metadata[b'row_count'] == b'100'

    @patch('quantdl.master.security_master.setup_logger')
    @patch('quantdl.master.security_master.raw_sql_with_retry')
    def test_sid_to_permno_works_when_loaded_from_s3(self, mock_sql, mock_logger):
        """Test sid_to_permno works when cik_cusip is None (S3 fast path).

        This tests the bug fix where sid_to_permno called security_map() which
        required cik_cusip, but cik_cusip was None when loaded from S3.
        """
        import io
        import pyarrow.parquet as pq

        # Create test data WITH permno column (required for sid_to_permno)
        test_df = pl.DataFrame({
            'security_id': [101, 102],
            'permno': [555, 666],
            'symbol': ['AAPL', 'MSFT'],
            'company': ['Apple Inc', 'Microsoft'],
            'cik': ['0000320193', '0000789019'],
            'cusip': ['037833100', '594918104'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31), dt.date(2020, 12, 31)]
        })

        table = test_df.to_arrow()
        metadata = {
            b'crsp_end_date': b'2024-12-31',
            b'version': b'1.0',
            b'row_count': b'2'
        }
        table = table.replace_schema_metadata(metadata)

        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        mock_s3 = Mock()
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=buffer.read()))
        }

        # Initialize from S3 (fast path)
        sm = SecurityMaster(
            db=None,
            s3_client=mock_s3,
            bucket_name='test-bucket',
            s3_key='test-key',
            force_rebuild=False
        )

        # Verify S3 fast path was used
        assert sm._from_s3 is True
        assert sm.cik_cusip is None  # cik_cusip is None when loaded from S3

        # This should work without calling security_map()
        result = sm.sid_to_permno(101)
        assert result == 555

        # Verify second lookup works too
        result2 = sm.sid_to_permno(102)
        assert result2 == 666


class TestSecurityMasterSecOperations:
    """Test SecurityMaster SEC operations"""

    def test_fetch_sec_mapping_full(self):
        """Test _fetch_sec_mapping_full fetches and parses SEC data."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_response = Mock()
        mock_response.json.return_value = {
            '0': {'cik_str': '320193', 'ticker': 'AAPL', 'title': 'Apple Inc'},
            '1': {'cik_str': '1018724', 'ticker': 'AMZN', 'title': 'Amazon.com Inc'}
        }

        with patch('quantdl.master.security_master.requests.get', return_value=mock_response):
            result = sm._fetch_sec_mapping_full()

        assert len(result) == 2
        assert 'ticker' in result.columns
        assert 'cik' in result.columns
        assert 'title' in result.columns

    def test_auto_resolve_null_candidates_warning(self):
        """Test auto_resolve raises error when security not active on query date (lines 651-652, 671)."""
        # Create master_tb with valid security_ids and the symbol, but with
        # date ranges that don't include the query date
        master_tb = pl.DataFrame({
            'security_id': [101, 102],
            'symbol': ['AAA', 'AAA'],
            'permno': [None, None],
            'cik': [None, None],
            'start_date': [dt.date(2019, 1, 1), dt.date(2021, 1, 1)],
            'end_date': [dt.date(2019, 12, 31), dt.date(2021, 12, 31)]  # Neither covers 2020-06-30
        })
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        # auto_resolve should raise ValueError since no candidate is active on the query date
        with pytest.raises(ValueError, match="was not active on"):
            sm.auto_resolve('AAA', '2020-06-30')

    def test_update_from_sec_extends_end_dates(self):
        """Test update_from_sec extends end_dates for active securities."""
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'permno': [555],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2024, 12, 31)]  # Stale end_date
        })

        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        sec_data = pl.DataFrame({
            'ticker': ['AAPL'],
            'cik': ['0000320193'],
            'title': ['Apple Inc']
        })

        with patch.object(sm, '_fetch_sec_mapping_full', return_value=sec_data):
            stats = sm.update_from_sec()

        assert stats['extended'] == 1
        assert stats['added'] == 0
        assert stats['unchanged'] == 0

        # Verify end_date was extended
        updated_row = sm.master_tb.filter(pl.col('security_id') == 101)
        assert updated_row['end_date'][0] > dt.date(2024, 12, 31)

    def test_update_from_sec_adds_new_securities(self):
        """Test update_from_sec adds new securities not in master_tb."""
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'permno': [555],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2024, 12, 31)]
        })

        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()

        sec_data = pl.DataFrame({
            'ticker': ['AAPL', 'NEWIPO'],
            'cik': ['0000320193', '9999999999'],
            'title': ['Apple Inc', 'New IPO Corp']
        })

        with patch.object(sm, '_fetch_sec_mapping_full', return_value=sec_data):
            stats = sm.update_from_sec()

        assert stats['added'] == 1
        assert stats['extended'] == 1  # AAPL extended

        # Verify new security was added
        new_rows = sm.master_tb.filter(pl.col('cik') == '9999999999')
        assert len(new_rows) == 1
        assert new_rows['symbol'][0] == 'NEWIPO'

    def test_update_from_sec_with_s3_export(self):
        """Test update_from_sec exports to S3 when s3_client provided."""
        master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'permno': [555],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2009, 1, 1)],
            'end_date': [dt.date(2024, 12, 31)]
        })

        sm = SecurityMaster.__new__(SecurityMaster)
        sm.master_tb = master_tb
        sm.logger = Mock()
        sm.CRSP_LATEST_DATE = '2024-12-31'

        sec_data = pl.DataFrame({
            'ticker': ['AAPL'],
            'cik': ['0000320193'],
            'title': ['Apple Inc']
        })

        mock_s3 = Mock()

        with patch.object(sm, '_fetch_sec_mapping_full', return_value=sec_data):
            stats = sm.update_from_sec(
                s3_client=mock_s3,
                bucket_name='test-bucket'
            )

        # export_to_s3 uses upload_fileobj, not put_object
        mock_s3.upload_fileobj.assert_called_once()
        call_args = mock_s3.upload_fileobj.call_args
        assert call_args[0][1] == 'test-bucket'  # bucket_name
        assert 'security_master.parquet' in call_args[0][2]  # s3_key


class TestOpenFIGIIntegration:
    """Test OpenFIGI integration methods"""

    def test_fetch_openfigi_mapping_success(self):
        """Test _fetch_openfigi_mapping with successful API response."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = [
            {"data": [{"shareClassFIGI": "BBG001S5N8V8"}]},
            {"data": [{"shareClassFIGI": "BBG000B9XRY4"}]},
            {"error": "No identifier found."}
        ]

        with patch('quantdl.master.security_master.requests.post', return_value=mock_response):
            with patch('quantdl.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                result = sm._fetch_openfigi_mapping(['AAPL', 'MSFT', 'UNKNOWN'])

        assert result['AAPL'] == 'BBG001S5N8V8'
        assert result['MSFT'] == 'BBG000B9XRY4'
        assert result['UNKNOWN'] is None

    def test_fetch_openfigi_mapping_api_error(self):
        """Test _fetch_openfigi_mapping handles API errors gracefully."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        with patch('quantdl.master.security_master.requests.post') as mock_post:
            mock_post.side_effect = Exception("API error")
            with patch('quantdl.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                result = sm._fetch_openfigi_mapping(['AAPL'])

        assert result['AAPL'] is None
        sm.logger.error.assert_called()

    def test_fetch_openfigi_mapping_batching(self):
        """Test _fetch_openfigi_mapping handles batching correctly with API key."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        # Create 250 tickers to test batching (with API key: 100 per batch = 3 batches)
        tickers = [f'SYM{i:03d}' for i in range(250)]

        call_count = 0
        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            batch_size = len(kwargs.get('json', []))
            mock_resp.json.return_value = [
                {"data": [{"shareClassFIGI": f"FIGI{i}"}]}
                for i in range(batch_size)
            ]
            return mock_resp

        with patch.dict(os.environ, {'OPENFIGI_API_KEY': 'test-key'}):
            with patch('quantdl.master.security_master.requests.post', side_effect=mock_post):
                with patch('quantdl.master.security_master.RateLimiter') as mock_rl:
                    mock_rl.return_value.acquire = Mock()
                    result = sm._fetch_openfigi_mapping(tickers)

        # With API key: 100 per batch, 250 tickers = 3 batches
        assert call_count == 3
        assert len(result) == 250

    def test_fetch_openfigi_uses_api_key_when_available(self):
        """Test _fetch_openfigi_mapping uses API key from environment."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_response = Mock()
        mock_response.json.return_value = [{"data": [{"shareClassFIGI": "FIGI123"}]}]

        with patch.dict(os.environ, {'OPENFIGI_API_KEY': 'test-key'}):
            with patch('quantdl.master.security_master.requests.post', return_value=mock_response) as mock_post:
                with patch('quantdl.master.security_master.RateLimiter') as mock_rl:
                    mock_rl.return_value.acquire = Mock()
                    sm._fetch_openfigi_mapping(['AAPL'])

        # Check that API key header was included
        call_args = mock_post.call_args
        headers = call_args.kwargs.get('headers', {})
        assert headers.get('X-OPENFIGI-APIKEY') == 'test-key'


class TestNasdaqUniverse:
    """Test Nasdaq universe fetching"""

    @patch('quantdl.master.security_master.fetch_all_stocks')
    def test_fetch_nasdaq_universe_success(self, mock_fetch):
        """Test _fetch_nasdaq_universe returns set of tickers."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_fetch.return_value = pd.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL']})

        result = sm._fetch_nasdaq_universe()

        assert result == {'AAPL', 'MSFT', 'GOOGL'}
        mock_fetch.assert_called_once_with(with_filter=True, refresh=True, logger=sm.logger)

    @patch('quantdl.master.security_master.fetch_all_stocks')
    def test_fetch_nasdaq_universe_error(self, mock_fetch):
        """Test _fetch_nasdaq_universe returns empty set on error."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_fetch.side_effect = Exception("FTP error")

        result = sm._fetch_nasdaq_universe()

        assert result == set()
        sm.logger.error.assert_called()


class TestRebrandDetection:
    """Test rebrand detection logic"""

    def test_detect_rebrands_finds_match(self):
        """Test _detect_rebrands finds rebrands by FIGI match."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        disappeared = {'FB'}
        appeared = {'META'}
        figi_mapping = {
            'FB': 'BBG000MM2P62',
            'META': 'BBG000MM2P62'  # Same FIGI = rebrand
        }

        result = sm._detect_rebrands(disappeared, appeared, figi_mapping)

        assert len(result) == 1
        assert result[0] == ('FB', 'META', 'BBG000MM2P62')

    def test_detect_rebrands_no_match(self):
        """Test _detect_rebrands with no FIGI matches."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        disappeared = {'OLDCO'}
        appeared = {'NEWCO'}
        figi_mapping = {
            'OLDCO': 'FIGI_OLD',
            'NEWCO': 'FIGI_NEW'  # Different FIGIs
        }

        result = sm._detect_rebrands(disappeared, appeared, figi_mapping)

        assert len(result) == 0

    def test_detect_rebrands_missing_figi(self):
        """Test _detect_rebrands handles missing FIGI mappings."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        disappeared = {'OLD'}
        appeared = {'NEW'}
        figi_mapping = {
            'OLD': None,  # No FIGI
            'NEW': 'FIGI_NEW'
        }

        result = sm._detect_rebrands(disappeared, appeared, figi_mapping)

        assert len(result) == 0


class TestPrevUniversePersistence:
    """Test prev_universe save/load operations"""

    def test_load_prev_universe_success(self):
        """Test _load_prev_universe loads from S3."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_s3 = Mock()
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=b'{"tickers": ["AAPL", "MSFT"], "date": "2025-01-15"}'))
        }

        result, date = sm._load_prev_universe(mock_s3, 'test-bucket')

        assert result == {'AAPL', 'MSFT'}
        assert date == '2025-01-15'

    def test_load_prev_universe_not_found(self):
        """Test _load_prev_universe returns empty set when file not found."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_s3 = Mock()
        mock_s3.exceptions.NoSuchKey = Exception
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey("Not found")

        result, date = sm._load_prev_universe(mock_s3, 'test-bucket')

        assert result == set()
        assert date is None

    def test_save_prev_universe(self):
        """Test _save_prev_universe saves to S3."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        mock_s3 = Mock()
        universe = {'AAPL', 'MSFT', 'GOOGL'}

        sm._save_prev_universe(mock_s3, 'test-bucket', universe, '2025-01-15')

        mock_s3.put_object.assert_called_once()
        call_args = mock_s3.put_object.call_args
        assert call_args.kwargs['Bucket'] == 'test-bucket'
        assert call_args.kwargs['Key'] == 'data/master/prev_universe.json'

        # Verify JSON content
        import json
        body = json.loads(call_args.kwargs['Body'].decode('utf-8'))
        assert set(body['tickers']) == universe
        assert body['date'] == '2025-01-15'


class TestUpdateNoWRDS:
    """Test update_no_wrds method"""

    def test_update_no_wrds_bootstrap(self):
        """Test update_no_wrds bootstraps when no prev_universe exists."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAPL'],
            'company': ['Apple Inc'],
            'permno': [555],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2024, 12, 31)]
        })

        mock_s3 = Mock()
        mock_s3.exceptions.NoSuchKey = Exception
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey("Not found")

        with patch.object(sm, '_fetch_nasdaq_universe', return_value={'AAPL'}):
            with patch.object(sm, 'export_to_s3'):
                stats = sm.update_no_wrds(mock_s3, 'test-bucket')

        assert stats['extended'] == 1
        # Verify end_date was extended
        assert sm.master_tb['end_date'][0] == dt.date.today()

    def test_update_no_wrds_extends_active(self):
        """Test update_no_wrds extends end_date for active securities."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.master_tb = pl.DataFrame({
            'security_id': [101, 102],
            'symbol': ['AAPL', 'MSFT'],
            'company': ['Apple', 'Microsoft'],
            'permno': [555, 666],
            'cik': ['0000320193', '0000789019'],
            'cusip': ['037833100', '594918104'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2024, 12, 31), dt.date(2024, 12, 31)]
        })

        mock_s3 = Mock()
        # Return prev_universe
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=b'{"tickers": ["AAPL", "MSFT"], "date": "2025-01-14"}'))
        }
        mock_s3.exceptions.NoSuchKey = Exception

        with patch.object(sm, '_fetch_nasdaq_universe', return_value={'AAPL', 'MSFT'}):
            with patch.object(sm, 'export_to_s3'):
                stats = sm.update_no_wrds(mock_s3, 'test-bucket')

        assert stats['extended'] == 2

    def test_update_no_wrds_detects_rebrand(self):
        """Test update_no_wrds detects and handles rebrands."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['FB'],
            'company': ['Facebook'],
            'permno': [555],
            'cik': ['0001326801'],
            'cusip': ['30303M102'],
            'start_date': [dt.date(2012, 5, 18)],
            'end_date': [dt.date(2024, 12, 31)]
        })

        mock_s3 = Mock()
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=b'{"tickers": ["FB"], "date": "2025-01-14"}'))
        }
        mock_s3.exceptions.NoSuchKey = Exception

        # FB disappeared, META appeared with same FIGI
        figi_mapping = {'FB': 'BBG000MM2P62', 'META': 'BBG000MM2P62'}

        with patch.object(sm, '_fetch_nasdaq_universe', return_value={'META'}):
            with patch.object(sm, '_fetch_openfigi_mapping', return_value=figi_mapping):
                with patch.object(sm, 'export_to_s3'):
                    stats = sm.update_no_wrds(mock_s3, 'test-bucket')

        assert stats['rebranded'] == 1

        # Verify new row added with same security_id
        meta_rows = sm.master_tb.filter(pl.col('symbol') == 'META')
        assert len(meta_rows) == 1
        assert meta_rows['security_id'][0] == 101  # Same security_id

    def test_update_no_wrds_adds_new_ipo(self):
        """Test update_no_wrds adds new IPOs."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['AAPL'],
            'company': ['Apple'],
            'permno': [555],
            'cik': ['0000320193'],
            'cusip': ['037833100'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2024, 12, 31)]
        })

        mock_s3 = Mock()
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=b'{"tickers": ["AAPL"], "date": "2025-01-14"}'))
        }
        mock_s3.exceptions.NoSuchKey = Exception

        figi_mapping = {'NEWIPO': 'FIGI_NEW'}

        with patch.object(sm, '_fetch_nasdaq_universe', return_value={'AAPL', 'NEWIPO'}):
            with patch.object(sm, '_fetch_openfigi_mapping', return_value=figi_mapping):
                with patch.object(sm, 'export_to_s3'):
                    stats = sm.update_no_wrds(mock_s3, 'test-bucket')

        assert stats['added'] == 1
        assert stats['extended'] == 1  # AAPL extended

        # Verify new IPO row
        new_rows = sm.master_tb.filter(pl.col('symbol') == 'NEWIPO')
        assert len(new_rows) == 1
        assert new_rows['security_id'][0] == 102  # New security_id

    def test_update_no_wrds_grace_period(self):
        """Test update_no_wrds respects grace period for delisting."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        # OLDCO + AAPL existed yesterday, only AAPL in current
        sm.master_tb = pl.DataFrame({
            'security_id': [101, 102],
            'symbol': ['OLDCO', 'AAPL'],
            'company': ['Old Corp', 'Apple'],
            'permno': [555, 666],
            'cik': ['0001234567', '0000320193'],
            'cusip': ['12345678', '037833100'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2024, 12, 31), dt.date(2024, 12, 31)]
        })

        mock_s3 = Mock()
        yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()

        # Prev universe had both OLDCO and AAPL
        mock_s3.get_object.return_value = {
            'Body': Mock(read=Mock(return_value=f'{{"tickers": ["OLDCO", "AAPL"], "date": "{yesterday}"}}'.encode()))
        }
        mock_s3.exceptions.NoSuchKey = type('NoSuchKey', (Exception,), {})

        # Current Nasdaq only has AAPL (OLDCO disappeared but within grace period)
        with patch.object(sm, '_fetch_nasdaq_universe', return_value={'AAPL'}):
            with patch.object(sm, '_fetch_openfigi_mapping', return_value={'OLDCO': 'FIGI_OLD'}):
                with patch.object(sm, 'export_to_s3'):
                    stats = sm.update_no_wrds(mock_s3, 'test-bucket', grace_period_days=14)

        # AAPL extended (active), OLDCO extended (in grace period, only 1 day missing)
        assert stats['extended'] == 2
        assert stats['delisted'] == 0


class TestOverwriteFromCRSP:
    """Test overwrite_from_crsp method"""

    def test_overwrite_from_crsp(self):
        """Test overwrite_from_crsp rebuilds from CRSP with FIGI."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()
        sm.db = None
        sm.master_tb = pl.DataFrame({
            'security_id': [101],
            'symbol': ['OLD'],
            'company': ['Old'],
            'permno': [1],
            'cik': ['0001'],
            'cusip': ['11111111'],
            'start_date': [dt.date(2020, 1, 1)],
            'end_date': [dt.date(2020, 12, 31)]
        })
        sm.CRSP_LATEST_DATE = '2024-12-31'

        mock_db = Mock()
        mock_s3 = Mock()

        # Mock cik_cusip_mapping and master_table
        new_master = pl.DataFrame({
            'security_id': [101, 102],
            'symbol': ['AAPL', 'MSFT'],
            'company': ['Apple', 'Microsoft'],
            'permno': [555, 666],
            'cik': ['0000320193', '0000789019'],
            'cusip': ['037833100', '594918104'],
            'start_date': [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            'end_date': [dt.date(2024, 12, 31), dt.date(2024, 12, 31)]
        })

        figi_mapping = {'AAPL': 'FIGI_AAPL', 'MSFT': 'FIGI_MSFT'}

        with patch.object(sm, 'cik_cusip_mapping'):
            with patch.object(sm, 'master_table', return_value=new_master):
                with patch.object(sm, '_fetch_openfigi_mapping', return_value=figi_mapping):
                    with patch.object(sm, 'export_to_s3'):
                        # Need to manually set master_tb since master_table is mocked
                        sm.master_tb = new_master
                        stats = sm.overwrite_from_crsp(mock_db, mock_s3)

        assert stats['rows'] == 2
        assert stats['figi_mapped'] == 2


class TestOpenFIGIRetryBehavior:
    """Test OpenFIGI retry and backoff behavior"""

    def test_fetch_openfigi_429_retries_with_backoff(self):
        """Test _fetch_openfigi_mapping retries on 429 with exponential backoff."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = Mock()
            # First 2 calls: 429
            if call_count <= 2:
                mock_resp.status_code = 429
                return mock_resp
            # Third call: success
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            mock_resp.json.return_value = [{"data": [{"shareClassFIGI": "FIGI1"}]}]
            return mock_resp

        with patch('quantdl.master.security_master.requests.post', side_effect=mock_post):
            with patch('quantdl.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                with patch('quantdl.master.security_master.time.sleep') as mock_sleep:
                    result = sm._fetch_openfigi_mapping(['AAPL'])

        # Should have retried twice
        assert call_count == 3
        # Should have slept with exponential backoff (1s, 2s)
        assert mock_sleep.call_count == 2
        assert result['AAPL'] == 'FIGI1'

    def test_fetch_openfigi_5xx_retries(self):
        """Test _fetch_openfigi_mapping retries on 5xx server errors."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = Mock()
            if call_count == 1:
                mock_resp.status_code = 503
                return mock_resp
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            mock_resp.json.return_value = [{"data": [{"shareClassFIGI": "FIGI1"}]}]
            return mock_resp

        with patch('quantdl.master.security_master.requests.post', side_effect=mock_post):
            with patch('quantdl.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                with patch('quantdl.master.security_master.time.sleep'):
                    result = sm._fetch_openfigi_mapping(['AAPL'])

        assert call_count == 2
        assert result['AAPL'] == 'FIGI1'

    def test_fetch_openfigi_exhausts_retries(self):
        """Test _fetch_openfigi_mapping marks None after exhausting retries."""
        from quantdl.master.security_master import OPENFIGI_MAX_RETRIES
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        def mock_post(*args, **kwargs):
            mock_resp = Mock()
            mock_resp.status_code = 500
            return mock_resp

        with patch('quantdl.master.security_master.requests.post', side_effect=mock_post):
            with patch('quantdl.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                with patch('quantdl.master.security_master.time.sleep'):
                    result = sm._fetch_openfigi_mapping(['AAPL'])

        assert result['AAPL'] is None
        sm.logger.warning.assert_called()

    def test_fetch_openfigi_progress_logging(self):
        """Test _fetch_openfigi_mapping logs progress every 10 batches."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        # With API key: 100 per batch. 1500 tickers = 15 batches
        # Should log at batch 10 and 15 (final)
        tickers = [f'SYM{i:04d}' for i in range(1500)]

        def mock_post(*args, **kwargs):
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            batch_size = len(kwargs.get('json', []))
            mock_resp.json.return_value = [
                {"data": [{"shareClassFIGI": f"FIGI{i}"}]}
                for i in range(batch_size)
            ]
            return mock_resp

        with patch.dict(os.environ, {'OPENFIGI_API_KEY': 'test-key'}):
            with patch('quantdl.master.security_master.requests.post', side_effect=mock_post):
                with patch('quantdl.master.security_master.RateLimiter') as mock_rl:
                    mock_rl.return_value.acquire = Mock()
                    result = sm._fetch_openfigi_mapping(tickers)

        # Check progress logging (should log at batch 10, 15)
        info_calls = [str(call) for call in sm.logger.info.call_args_list]
        progress_logs = [c for c in info_calls if 'progress' in c.lower()]
        assert len(progress_logs) >= 2  # At least 10th batch and final

    def test_fetch_openfigi_request_exception_retries(self):
        """Test _fetch_openfigi_mapping retries on RequestException."""
        sm = SecurityMaster.__new__(SecurityMaster)
        sm.logger = Mock()

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.RequestException("Connection error")
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = Mock()
            mock_resp.json.return_value = [{"data": [{"shareClassFIGI": "FIGI1"}]}]
            return mock_resp

        with patch('quantdl.master.security_master.requests.post', side_effect=mock_post):
            with patch('quantdl.master.security_master.RateLimiter') as mock_rl:
                mock_rl.return_value.acquire = Mock()
                with patch('quantdl.master.security_master.time.sleep'):
                    result = sm._fetch_openfigi_mapping(['AAPL'])

        assert call_count == 2
        assert result['AAPL'] == 'FIGI1'
