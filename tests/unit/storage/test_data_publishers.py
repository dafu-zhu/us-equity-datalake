"""
Unit tests for storage.data_publishers module
Tests DataPublishers functionality with dependency injection
"""
import os
import pytest
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import queue
import threading
import polars as pl
import requests


def _make_publisher():
    from quantdl.storage.data_publishers import DataPublishers

    s3_client = Mock()
    upload_config = SimpleNamespace(transfer={})
    logger = Mock()
    data_collectors = Mock()
    security_master = Mock()
    security_master.get_security_id = Mock(return_value=12345)

    publisher = DataPublishers(
        s3_client=s3_client,
        upload_config=upload_config,
        logger=logger,
        data_collectors=data_collectors,
        security_master=security_master,
        bucket_name="test-bucket"
    )

    return publisher, s3_client, data_collectors


@patch.dict(os.environ, {'STORAGE_BACKEND': 's3'})
class TestDataPublishers:
    def test_initialization(self):
        publisher, s3_client, data_collectors = _make_publisher()

        assert publisher.s3_client == s3_client
        assert publisher.data_collectors == data_collectors
        assert publisher.bucket_name == "test-bucket"

    def test_publish_daily_ticks_success_yearly(self):
        """Test publishing daily ticks with yearly partition (legacy)"""
        from botocore.exceptions import ClientError

        publisher, s3_client, _ = _make_publisher()

        # Mock get_object to raise NoSuchKey (no existing history file)
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, month=None, by_year=False)

        assert result == {"symbol": "AAPL", "status": "success", "error": None}
        s3_client.upload_fileobj.assert_called_once()

        # Verify security_id-based path (new design)
        call_args = s3_client.upload_fileobj.call_args
        assert "data/raw/ticks/daily/12345/" in call_args.kwargs["Key"]

    def test_publish_daily_ticks_requires_df(self):
        """df is required when by_year is False."""
        publisher, _, _ = _make_publisher()

        with pytest.raises(ValueError, match="df is required"):
            publisher.publish_daily_ticks("AAPL", 2024, 12345, df=None, by_year=False)

    def test_publish_daily_ticks_success_monthly(self):
        """Test publishing daily ticks with monthly partition (recommended)"""
        publisher, s3_client, _ = _make_publisher()

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, month=6, by_year=False)

        assert result == {"symbol": "AAPL", "status": "success", "error": None}
        s3_client.upload_fileobj.assert_called_once()

        # Verify security_id-based monthly path (new design)
        call_args = s3_client.upload_fileobj.call_args
        assert "data/raw/ticks/daily/12345/2024/06/ticks.parquet" in call_args.kwargs["Key"]

    def test_publish_daily_ticks_empty(self):
        """Test that empty dataframes are skipped for both partition types"""
        publisher, s3_client, _ = _make_publisher()

        df = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        })

        # Test yearly partition
        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, month=None, by_year=False)
        assert result["status"] == "skipped"

        # Test monthly partition
        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, month=6, by_year=False)
        assert result["status"] == "skipped"

        s3_client.upload_fileobj.assert_not_called()

    def test_publish_daily_ticks_metadata_monthly(self):
        """Test that monthly partition includes correct metadata"""
        publisher, s3_client, _ = _make_publisher()

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, month=6, by_year=False)

        assert result["status"] == "success"

        # Verify metadata includes month and partition_type
        call_args = s3_client.upload_fileobj.call_args
        metadata = call_args.kwargs["ExtraArgs"]["Metadata"]
        assert metadata.get('month') == '06'
        assert metadata.get('partition_type') == 'monthly'

    def test_publish_daily_ticks_metadata_yearly(self):
        """Test that yearly partition includes correct metadata"""
        from botocore.exceptions import ClientError

        publisher, s3_client, _ = _make_publisher()

        # Mock get_object to raise NoSuchKey (no existing history file)
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, month=None, by_year=False)

        assert result["status"] == "success"

        # Verify metadata includes partition_type (history) but not month
        call_args = s3_client.upload_fileobj.call_args
        metadata = call_args.kwargs["ExtraArgs"]["Metadata"]
        assert 'month' not in metadata or metadata.get('month') is None
        assert metadata.get('partition_type') == 'history'  # New design uses 'history' instead of 'yearly'

    def test_publish_daily_ticks_by_year_skips_empty_year(self):
        """by_year returns skipped when year_df is empty."""
        publisher, _, data_collectors = _make_publisher()

        data_collectors.collect_daily_ticks_year.return_value = pl.DataFrame()

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df=None, by_year=True)

        assert result["status"] == "skipped"
        data_collectors.collect_daily_ticks_year.assert_called_once_with("AAPL", 2024)

    def test_publish_daily_ticks_by_year(self):
        """Test by_year publishing uses year data and uploads monthly in parallel."""
        publisher, s3_client, data_collectors = _make_publisher()

        year_df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        month_df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        data_collectors.collect_daily_ticks_year.return_value = year_df
        data_collectors.collect_daily_ticks_month.return_value = month_df

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df=None, by_year=True, max_workers=2)

        data_collectors.collect_daily_ticks_year.assert_called_once_with("AAPL", 2024)
        assert data_collectors.collect_daily_ticks_month.call_count == 12
        for call in data_collectors.collect_daily_ticks_month.call_args_list:
            assert call.kwargs.get("year_df") is year_df

        assert s3_client.upload_fileobj.call_count == 12
        assert result["status"] == "success"

    def test_publish_daily_ticks_by_year_failed_months(self):
        """by_year returns failed when any month publish fails."""
        publisher, _, data_collectors = _make_publisher()

        year_df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        data_collectors.collect_daily_ticks_year.return_value = year_df
        data_collectors.collect_daily_ticks_month.return_value = year_df

        with patch.object(publisher, "_publish_daily_ticks_df", return_value={"status": "failed"}):
            result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df=None, by_year=True)

        assert result["status"] == "failed"

    def test_publish_daily_ticks_by_year_all_skipped(self):
        """by_year returns skipped when all months are skipped."""
        publisher, _, data_collectors = _make_publisher()

        year_df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })
        data_collectors.collect_daily_ticks_year.return_value = year_df
        data_collectors.collect_daily_ticks_month.return_value = year_df

        with patch.object(publisher, "_publish_daily_ticks_df", return_value={"status": "skipped"}):
            result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df=None, by_year=True)

        assert result["status"] == "skipped"

    def test_publish_fundamental_skips_without_cik(self):
        publisher, _, data_collectors = _make_publisher()

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik=None,
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "skipped"
        data_collectors.collect_fundamental_long.assert_not_called()

    def test_publish_daily_ticks_value_error_skips(self):
        from botocore.exceptions import ClientError

        publisher, s3_client, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=ValueError("not active on"))

        # Mock get_object to raise NoSuchKey (no existing history file)
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, by_year=False)

        assert result["status"] == "skipped"

    def test_publish_daily_ticks_request_exception(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=requests.RequestException("boom"))

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, by_year=False)

        assert result["status"] == "failed"

    def test_publish_daily_ticks_unexpected_error(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=RuntimeError("boom"))

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, by_year=False)

        assert result["status"] == "failed"

    def test_publish_fundamental_empty(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors.collect_fundamental_long.return_value = pl.DataFrame()

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "skipped"

    def test_publish_fundamental_success(self):
        publisher, _, data_collectors = _make_publisher()
        publisher.upload_fileobj = Mock()
        data_collectors._load_concepts.return_value = ["rev", "net_inc"]
        data_collectors.collect_fundamental_long.return_value = pl.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "as_of_date": ["2024-06-30", "2024-06-30"],
            "accn": ["1", "1"],
            "form": ["10-Q", "10-Q"],
            "concept": ["rev", "net_inc"],
            "value": [100.0, 10.0],
            "start": ["2024-04-01", "2024-04-01"],
            "end": ["2024-06-30", "2024-06-30"],
            "frame": ["CY2024Q2", "CY2024Q2"],
            "is_instant": [False, False]
        })

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "success"
        publisher.upload_fileobj.assert_called_once()

    def test_publish_fundamental_uses_cik_path(self):
        """Test that publish_fundamental uses CIK-based path, not symbol"""
        publisher, s3_client, data_collectors = _make_publisher()
        data_collectors._load_concepts.return_value = ["rev"]
        data_collectors.collect_fundamental_long.return_value = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "accn": ["1"],
            "form": ["10-Q"],
            "concept": ["rev"],
            "value": [100.0],
            "start": ["2024-04-01"],
            "end": ["2024-06-30"],
            "frame": ["CY2024Q2"],
            "is_instant": [False]
        })

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "success"

        # Verify CIK-based path is used
        call_args = s3_client.upload_fileobj.call_args
        s3_key = call_args.kwargs["Key"]
        assert s3_key == "data/raw/fundamental/0000320193/fundamental.parquet"
        assert "AAPL" not in s3_key  # Symbol should NOT be in path

        # Verify metadata includes both symbol and CIK
        metadata = call_args.kwargs["ExtraArgs"]["Metadata"]
        assert metadata["symbol"] == "AAPL"
        assert metadata["cik"] == "0000320193"

    def test_publish_fundamental_request_exception(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors.collect_fundamental_long.side_effect = requests.RequestException("boom")

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "failed"

    def test_publish_fundamental_value_error(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors.collect_fundamental_long.side_effect = ValueError("bad data")

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "failed"

    def test_publish_fundamental_unexpected_exception(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors.collect_fundamental_long.side_effect = RuntimeError("boom")

        sec_rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "failed"

    def test_publish_ttm_fundamental_success(self):
        publisher, _, data_collectors = _make_publisher()
        publisher.upload_fileobj = Mock()
        data_collectors._load_concepts.return_value = ["rev"]
        data_collectors.collect_ttm_long_range.return_value = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "concept": ["rev"],
            "value": [100.0]
        })

        sec_rate_limiter = Mock()

        result = publisher.publish_ttm_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "success"
        publisher.upload_fileobj.assert_called_once()

    def test_publish_ttm_fundamental_uses_cik_path(self):
        """Test that publish_ttm_fundamental uses CIK-based path, not symbol"""
        publisher, s3_client, data_collectors = _make_publisher()
        data_collectors._load_concepts.return_value = ["rev"]
        data_collectors.collect_ttm_long_range.return_value = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "concept": ["rev"],
            "value": [100.0]
        })

        sec_rate_limiter = Mock()

        result = publisher.publish_ttm_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "success"

        # Verify CIK-based path is used
        call_args = s3_client.upload_fileobj.call_args
        s3_key = call_args.kwargs["Key"]
        assert s3_key == "data/derived/features/fundamental/0000320193/ttm.parquet"
        assert "AAPL" not in s3_key  # Symbol should NOT be in path

        # Verify metadata includes both symbol and CIK
        metadata = call_args.kwargs["ExtraArgs"]["Metadata"]
        assert metadata["symbol"] == "AAPL"
        assert metadata["cik"] == "0000320193"

    def test_publish_ttm_fundamental_skips_without_cik(self):
        publisher, _, data_collectors = _make_publisher()

        sec_rate_limiter = Mock()

        result = publisher.publish_ttm_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik=None,
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "skipped"
        data_collectors.collect_ttm_long_range.assert_not_called()

    def test_publish_ttm_fundamental_empty(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors._load_concepts.return_value = ["rev"]
        data_collectors.collect_ttm_long_range.return_value = pl.DataFrame()

        sec_rate_limiter = Mock()

        result = publisher.publish_ttm_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "skipped"

    def test_publish_ttm_fundamental_request_exception(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors._load_concepts.return_value = ["rev"]
        data_collectors.collect_ttm_long_range.side_effect = requests.RequestException("boom")

        sec_rate_limiter = Mock()

        result = publisher.publish_ttm_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "failed"

    def test_publish_ttm_fundamental_value_error(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors._load_concepts.return_value = ["rev"]
        data_collectors.collect_ttm_long_range.side_effect = ValueError("bad data")

        sec_rate_limiter = Mock()

        result = publisher.publish_ttm_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "failed"

    def test_publish_ttm_fundamental_unexpected_exception(self):
        publisher, _, data_collectors = _make_publisher()
        data_collectors._load_concepts.return_value = ["rev"]
        data_collectors.collect_ttm_long_range.side_effect = RuntimeError("boom")

        sec_rate_limiter = Mock()

        result = publisher.publish_ttm_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            cik="0000320193",
            sec_rate_limiter=sec_rate_limiter
        )

        assert result["status"] == "failed"

    def test_publish_derived_fundamental_success(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock()
        derived_df = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "metric": ["net_mgn"],
            "value": [0.1]
        })

        result = publisher.publish_derived_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            derived_df=derived_df,
            cik="0000320193"
        )

        assert result["status"] == "success"
        publisher.upload_fileobj.assert_called_once()

    def test_publish_derived_fundamental_uses_cik_path(self):
        """Test that publish_derived_fundamental uses CIK-based path, not symbol"""
        publisher, s3_client, _ = _make_publisher()
        derived_df = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "metric": ["net_mgn"],
            "value": [0.1]
        })

        result = publisher.publish_derived_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            derived_df=derived_df,
            cik="0000320193"
        )

        assert result["status"] == "success"

        # Verify CIK-based path is used
        call_args = s3_client.upload_fileobj.call_args
        s3_key = call_args.kwargs["Key"]
        assert s3_key == "data/derived/features/fundamental/0000320193/metrics.parquet"
        assert "AAPL" not in s3_key  # Symbol should NOT be in path

        # Verify metadata includes both symbol and CIK
        metadata = call_args.kwargs["ExtraArgs"]["Metadata"]
        assert metadata["symbol"] == "AAPL"
        assert metadata["cik"] == "0000320193"

    def test_publish_derived_fundamental_skips_without_cik(self):
        """Test that publish_derived_fundamental skips when CIK is None"""
        publisher, s3_client, _ = _make_publisher()
        derived_df = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "metric": ["net_mgn"],
            "value": [0.1]
        })

        result = publisher.publish_derived_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            derived_df=derived_df,
            cik=None
        )

        assert result["status"] == "skipped"
        assert "No CIK" in result["error"]
        s3_client.upload_fileobj.assert_not_called()

    def test_publish_derived_fundamental_upload_error(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=RuntimeError("boom"))
        derived_df = pl.DataFrame({
            "symbol": ["AAPL"],
            "as_of_date": ["2024-06-30"],
            "metric": ["net_mgn"],
            "value": [0.1]
        })

        result = publisher.publish_derived_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            derived_df=derived_df,
            cik="0000320193"
        )

        assert result["status"] == "failed"

    def test_publish_derived_fundamental_empty(self):
        publisher, _, _ = _make_publisher()
        derived_df = pl.DataFrame()

        result = publisher.publish_derived_fundamental(
            sym="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            derived_df=derived_df,
            cik="0000320193"
        )

        assert result["status"] == "skipped"

    def test_publish_top_3000_success(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock()

        result = publisher.publish_top_3000(
            year=2024,
            month=6,
            as_of="2024-06-30",
            symbols=["AAPL", "MSFT"],
            source="test"
        )

        assert result["status"] == "success"
        publisher.upload_fileobj.assert_called_once()

    def test_publish_top_3000_upload_error(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=RuntimeError("boom"))

        result = publisher.publish_top_3000(
            year=2024,
            month=6,
            as_of="2024-06-30",
            symbols=["AAPL", "MSFT"],
            source="test"
        )

        assert result["status"] == "failed"

    def test_publish_top_3000_empty(self):
        publisher, _, _ = _make_publisher()

        result = publisher.publish_top_3000(
            year=2024,
            month=6,
            as_of="2024-06-30",
            symbols=[],
            source="test"
        )

        assert result["status"] == "skipped"

    def test_minute_ticks_worker_success(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock()
        stats = {"success": 0, "failed": 0, "skipped": 0, "completed": 0}
        lock = threading.Lock()
        q = queue.Queue()
        df = pl.DataFrame({
            "timestamp": ["2024-06-30T09:30:00"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100]
        })
        q.put(("AAPL", "2024-06-30", df))
        q.put(None)

        t = threading.Thread(target=publisher.minute_ticks_worker, args=(q, stats, lock))
        t.start()
        t.join()

        assert stats["success"] == 1
        publisher.upload_fileobj.assert_called_once()

    def test_minute_ticks_worker_upload_error(self):
        publisher, _, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=RuntimeError("boom"))
        stats = {"success": 0, "failed": 0, "skipped": 0, "completed": 0}
        lock = threading.Lock()
        q = queue.Queue()
        df = pl.DataFrame({
            "timestamp": ["2024-06-30T09:30:00"],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [100]
        })
        q.put(("AAPL", "2024-06-30", df))
        q.put(None)

        t = threading.Thread(target=publisher.minute_ticks_worker, args=(q, stats, lock))
        t.start()
        t.join()

        assert stats["failed"] == 1

    def test_minute_ticks_worker_skips_empty(self):
        publisher, _, _ = _make_publisher()
        stats = {"success": 0, "failed": 0, "skipped": 0, "completed": 0}
        lock = threading.Lock()
        q = queue.Queue()
        q.put(("AAPL", "2024-06-30", pl.DataFrame()))
        q.put(None)

        t = threading.Thread(target=publisher.minute_ticks_worker, args=(q, stats, lock))
        t.start()
        t.join()

        assert stats["skipped"] == 1

    def test_publish_daily_ticks_value_error_other(self):
        """Test ValueError that doesn't contain 'not active on' returns failed status (lines 251-252)"""
        from botocore.exceptions import ClientError

        publisher, s3_client, _ = _make_publisher()
        publisher.upload_fileobj = Mock(side_effect=ValueError("some other error"))

        # Mock get_object to raise NoSuchKey (no existing history file)
        error_response = {'Error': {'Code': 'NoSuchKey'}}
        s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, by_year=False)

        assert result["status"] == "failed"
        assert result["error"] == "some other error"
        # Verify error logging was called
        publisher.logger.error.assert_called()

    def test_minute_ticks_worker_queue_timeout(self):
        """Test that worker handles queue.Empty exception and continues (lines 331-332)"""
        publisher, _, _ = _make_publisher()
        stats = {"success": 0, "failed": 0, "skipped": 0, "completed": 0}
        lock = threading.Lock()
        q = queue.Queue()

        # Start worker with empty queue - it will timeout waiting for items
        t = threading.Thread(target=publisher.minute_ticks_worker, args=(q, stats, lock))
        t.start()

        # Wait longer than the queue timeout (1 second) to ensure queue.Empty is raised
        # The worker uses queue.get(timeout=1), so we need to wait > 1 second
        time.sleep(1.5)

        # Add poison pill to stop worker
        q.put(None)
        t.join(timeout=2.0)

        # Worker should have stopped cleanly
        assert not t.is_alive()
        # No items were processed
        assert stats["success"] == 0
        assert stats["failed"] == 0
        assert stats["skipped"] == 0

    def test_get_fundamental_metadata_exists(self):
        """Test getting metadata for existing fundamental data"""
        publisher, s3_client, _ = _make_publisher()

        # Mock S3 head_object response with metadata
        s3_client.head_object.return_value = {
            'Metadata': {
                'symbol': 'AAPL',
                'cik': '0000320193',
                'latest_filing_date': '2024-01-31',
                'latest_accn': '0000320193-24-000010'
            }
        }

        metadata = publisher.get_fundamental_metadata('0000320193')

        assert metadata is not None
        assert metadata['symbol'] == 'AAPL'
        assert metadata['latest_filing_date'] == '2024-01-31'
        assert metadata['latest_accn'] == '0000320193-24-000010'
        s3_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='data/raw/fundamental/0000320193/fundamental.parquet'
        )

    def test_get_fundamental_metadata_not_exists(self):
        """Test getting metadata when file doesn't exist"""
        publisher, s3_client, _ = _make_publisher()

        # Mock S3 head_object to raise exception (file not found)
        s3_client.head_object.side_effect = Exception("404")

        metadata = publisher.get_fundamental_metadata('0000320193')

        assert metadata is None

    def test_publish_fundamental_includes_metadata_tracking(self):
        """Test that publish_fundamental includes latest_filing_date and latest_accn in metadata"""
        publisher, s3_client, data_collectors = _make_publisher()

        # Mock data
        mock_df = pl.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'as_of_date': ['2024-01-31', '2024-10-31'],
            'accn': ['0000320193-24-000010', '0000320193-24-000078'],
            'form': ['10-Q', '10-K'],
            'concept': ['Assets', 'Assets'],
            'value': [1000000, 1100000],
            'start': [None, None],
            'end': ['2024-01-31', '2024-10-31'],
            'frame': ['CY2024Q1', 'CY2024Q4'],
            'is_instant': [True, True]
        })

        data_collectors.collect_fundamental_long.return_value = mock_df
        data_collectors._load_concepts.return_value = ['Assets']

        # Mock rate limiter
        rate_limiter = Mock()

        result = publisher.publish_fundamental(
            sym='AAPL',
            start_date='2024-01-01',
            end_date='2024-12-31',
            cik='0000320193',
            sec_rate_limiter=rate_limiter
        )

        # Verify upload was called
        assert s3_client.upload_fileobj.called

        # Get the metadata argument from the upload call
        call_args = s3_client.upload_fileobj.call_args
        metadata = call_args[1]['ExtraArgs']['Metadata']

        # Verify latest_filing_date and latest_accn are in metadata
        assert 'latest_filing_date' in metadata
        assert 'latest_accn' in metadata
        assert metadata['latest_filing_date'] == '2024-10-31'  # Latest date
        assert metadata['latest_accn'] == '0000320193-24-000078'  # Accn for latest date
        assert result['status'] == 'success'

    def test_publish_daily_ticks_history_reraises_non_nosuchkey_error(self):
        """Test history file path re-raises non-NoSuchKey ClientError (lines 244,247,253)."""
        from botocore.exceptions import ClientError

        publisher, s3_client, _ = _make_publisher()

        # Mock get_object to raise a different error (e.g., AccessDenied)
        error_response = {'Error': {'Code': 'AccessDenied'}}
        s3_client.get_object.side_effect = ClientError(error_response, 'GetObject')

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, month=None, by_year=False)

        # Should fail with error
        assert result["status"] == "failed"

    def test_publish_daily_ticks_request_exception(self):
        """Test RequestException handling (lines 289,290)."""
        publisher, s3_client, _ = _make_publisher()

        s3_client.upload_fileobj.side_effect = requests.RequestException("Connection failed")

        df = pl.DataFrame({
            "timestamp": ["2024-06-30"],
            "open": [190.0],
            "high": [195.0],
            "low": [189.0],
            "close": [193.0],
            "volume": [50000000]
        })

        result = publisher.publish_daily_ticks("AAPL", 2024, 12345, df, month=6, by_year=False)

        assert result["status"] == "failed"
        assert "Connection failed" in result["error"]

    def test_publish_daily_ticks_to_history_success(self):
        """Test publish_daily_ticks_to_history with valid data."""
        publisher, s3_client, _ = _make_publisher()

        df = pl.DataFrame({
            "timestamp": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000, 2000, 3000]
        })

        result = publisher.publish_daily_ticks_to_history(
            security_id=1001,
            df=df,
            symbol="AAPL"
        )

        assert result["status"] == "success"
        assert result["security_id"] == 1001
        s3_client.upload_fileobj.assert_called_once()

        # Verify path
        call_args = s3_client.upload_fileobj.call_args
        assert "data/raw/ticks/daily/1001/history.parquet" in call_args.kwargs["Key"]

    def test_publish_daily_ticks_to_history_empty_df(self):
        """Test publish_daily_ticks_to_history with empty DataFrame."""
        publisher, s3_client, _ = _make_publisher()

        df = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        }).cast({"timestamp": pl.Utf8, "open": pl.Float64, "high": pl.Float64,
                 "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64})

        result = publisher.publish_daily_ticks_to_history(
            security_id=1001,
            df=df,
            symbol="AAPL"
        )

        assert result["status"] == "skipped"
        assert "No data available" in result["error"]
        s3_client.upload_fileobj.assert_not_called()

    def test_publish_daily_ticks_to_history_error(self):
        """Test publish_daily_ticks_to_history with upload error."""
        publisher, s3_client, _ = _make_publisher()

        s3_client.upload_fileobj.side_effect = Exception("Upload failed")

        df = pl.DataFrame({
            "timestamp": ["2020-01-02"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "volume": [1000]
        })

        result = publisher.publish_daily_ticks_to_history(
            security_id=1001,
            df=df,
            symbol="AAPL"
        )

        assert result["status"] == "failed"
        assert "Upload failed" in result["error"]
