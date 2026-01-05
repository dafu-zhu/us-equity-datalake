"""
Data publishing functionality for uploading market data to S3.

This module handles publishing collected data to S3 storage:
- Daily ticks (yearly Parquet files)
- Minute ticks (daily Parquet files)
- Fundamental data (yearly Parquet files)
"""

import io
import json
import datetime as dt
import logging
import queue
import threading
from typing import Dict, List, Optional, Callable
import requests
import polars as pl

from collection.fundamental import Fundamental


class DataPublishers:
    """
    Handles publishing market data to S3 storage.
    """

    def __init__(
        self,
        upload_fileobj_func: Callable,
        logger: logging.Logger
    ):
        """
        Initialize data publishers.

        :param upload_fileobj_func: Function to upload file objects to S3
        :param logger: Logger instance
        """
        self.upload_fileobj = upload_fileobj_func
        self.logger = logger

    def publish_daily_ticks(
        self,
        sym: str,
        year: int,
        df: pl.DataFrame
    ) -> Dict[str, Optional[str]]:
        """
        Publish daily ticks for a single symbol for entire year to S3.
        Returns dict with status for progress tracking.

        Storage: data/raw/ticks/daily/{symbol}/{YYYY}/ticks.parquet
        Contains all trading days for the year (~252 days).

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :param df: Polars DataFrame with daily ticks data
        :return: Dict with status info
        """
        try:
            # Check if DataFrame is empty (no rows fetched)
            if len(df) == 0:
                return {'symbol': sym, 'status': 'skipped', 'error': 'No data available'}

            # Setup S3 message (Parquet format)
            buffer = io.BytesIO()
            df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            # Year-based storage: data/raw/ticks/daily/{symbol}/{YYYY}/ticks.parquet
            s3_key = f"data/raw/ticks/daily/{sym}/{year}/ticks.parquet"
            s3_metadata = {
                'symbol': sym,
                'year': str(year),
                'data_type': 'ticks',
                'source': 'crsp' if year < 2025 else 'alpaca',
                'trading_days': str(len(df))
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None}

        except ValueError as e:
            # Handle expected conditions like security not active on date
            if "not active on" in str(e):
                self.logger.info(f'Skipping {sym}: {e}')
                return {'symbol': sym, 'status': 'skipped', 'error': str(e)}
            else:
                self.logger.error(f'ValueError for {sym}: {e}')
                return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except requests.RequestException as e:
            self.logger.warning(f'Failed to publish daily ticks for {sym}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error publishing {sym}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def minute_ticks_worker(
        self,
        data_queue: queue.Queue,
        stats: dict,
        stats_lock: threading.Lock
    ):
        """
        Worker thread that consumes fetched minute tick data and uploads to S3.

        :param data_queue: Queue containing (sym, trade_day, minute_df) tuples
        :param stats: Shared statistics dictionary
        :param stats_lock: Lock for updating statistics
        """
        while True:
            try:
                item = data_queue.get(timeout=1)
                if item is None:  # Poison pill to stop worker
                    break

                sym, trade_day, minute_df = item

                try:
                    # Skip if DataFrame is empty
                    if len(minute_df) == 0:
                        with stats_lock:
                            stats['skipped'] += 1
                        continue

                    # Round numeric fields to 4 decimal places
                    for field in ['open', 'high', 'low', 'close']:
                        if field in minute_df.columns:
                            minute_df = minute_df.with_columns(
                                minute_df[field].round(4)
                            )

                    # Setup S3 message
                    buffer = io.BytesIO()
                    minute_df.write_parquet(buffer)
                    buffer.seek(0)

                    # Parse date for S3 key
                    date_obj = dt.datetime.strptime(trade_day, '%Y-%m-%d').date()
                    year = date_obj.strftime('%Y')
                    month = date_obj.strftime('%m')
                    day = date_obj.strftime('%d')

                    s3_key = f"data/raw/ticks/minute/{sym}/{year}/{month}/{day}/ticks.parquet"
                    s3_metadata = {
                        'symbol': sym,
                        'trade_day': trade_day,
                        'data_type': 'ticks'
                    }
                    s3_metadata_prepared = {
                        k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                        for k, v in s3_metadata.items()
                    }

                    # Upload to S3
                    self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

                    with stats_lock:
                        stats['success'] += 1

                except Exception as e:
                    self.logger.error(f'Upload error for {sym} on {trade_day}: {e}')
                    with stats_lock:
                        stats['failed'] += 1

                finally:
                    data_queue.task_done()

            except queue.Empty:
                continue

    def publish_fundamental(
        self,
        sym: str,
        year: int,
        cik: Optional[str],
        dei_fields: List[str],
        gaap_fields: List[str],
        sec_rate_limiter
    ) -> Dict[str, Optional[str]]:
        """
        Publish fundamental data for a single symbol for an entire year to S3.
        Returns dict with status for progress tracking.

        Storage: data/raw/fundamental/{symbol}/{YYYY}/fundamental.parquet
        Contains all quarterly/annual filings for the year (no forward fill).

        :param sym: Symbol in SEC format
        :param year: Year to fetch data for
        :param cik: CIK string (or None to skip)
        :param dei_fields: DEI fields to collect
        :param gaap_fields: US-GAAP fields to collect
        :param sec_rate_limiter: Rate limiter for SEC API
        :return: Dict with status info
        """
        try:
            if cik is None:
                self.logger.warning(f'Skipping {sym}: No CIK found (should have been filtered earlier)')
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'No CIK found for {sym} in {year}',
                    'cik': None
                }

            # Rate limit before making SEC API request
            sec_rate_limiter.acquire()

            # Fetch from SEC EDGAR API
            fnd = Fundamental(cik, sym)

            # Build fields_dict using new API
            fields_dict = {}

            # Collect DEI fields
            for field in dei_fields:
                try:
                    dps = fnd.get_dps(field, 'dei')
                    fields_dict[field] = fnd.get_value_tuple(dps)
                except KeyError:
                    # Field not available for this company
                    fields_dict[field] = []

            # Collect US-GAAP fields
            for field in gaap_fields:
                try:
                    dps = fnd.get_dps(field, 'us-gaap')
                    fields_dict[field] = fnd.get_value_tuple(dps)
                except KeyError:
                    # Field not available for this company
                    fields_dict[field] = []

            # Define date range for the ENTIRE YEAR
            start_day = f"{year}-01-01"
            end_day = f"{year}-12-31"

            # Collect fields into DataFrame (only actual quarterly filings, no forward-fill)
            combined_df = fnd.collect_fields_raw(start_day, end_day, fields_dict)

            # Check if DataFrame is empty
            if len(combined_df) == 0:
                cik_str = f" (CIK {cik})" if cik else ""
                self.logger.info(f'No fundamental data for {sym}{cik_str} in {year} - likely no filings this year')
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'No fundamental data available for {sym} in {year}',
                    'cik': cik
                }

            # Setup S3 message
            buffer = io.BytesIO()
            combined_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            # Year-based storage: data/raw/fundamental/{symbol}/{YYYY}/fundamental.parquet
            s3_key = f"data/raw/fundamental/{sym}/{year}/fundamental.parquet"
            s3_metadata = {
                'symbol': sym,
                'year': str(year),
                'data_type': 'fundamental',
                'quarters': str(len(combined_df))  # Track number of quarters
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None}

        except requests.RequestException as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.warning(f'Failed to fetch data for {sym}{cik_str}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except ValueError as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.error(f'Invalid data for {sym}{cik_str}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.error(f'Unexpected error for {sym}{cik_str}: {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
