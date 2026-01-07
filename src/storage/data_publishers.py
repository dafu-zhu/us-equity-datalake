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
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, cast
import requests
import polars as pl
import yaml
from boto3.s3.transfer import TransferConfig

from collection.fundamental import Fundamental


class DataPublishers:
    """
    Handles publishing market data to S3 storage.
    """

    def __init__(
        self,
        s3_client,
        upload_config,
        logger: logging.Logger
    ):
        """
        Initialize data publishers.

        :param s3_client: Boto3 S3 client instance
        :param upload_config: UploadConfig instance with transfer settings
        :param logger: Logger instance
        """
        self.s3_client = s3_client
        self.upload_config = upload_config
        self.logger = logger

    def upload_fileobj(
        self,
        data: io.BytesIO,
        key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Upload file object to S3 with proper configuration"""

        # Define transfer config
        cfg = cast(Dict[str, Any], self.upload_config.transfer or {})
        transfer_config = TransferConfig(
            multipart_threshold=int(cfg.get('multipart_threshold', 10485760)),
            max_concurrency=int(cfg.get('max_concurrency', 5)),
            multipart_chunksize=int(cfg.get('multipart_chunksize', 10485760)),
            num_download_attempts=int(cfg.get('num_download_attempts', 5)),
            max_io_queue=int(cfg.get('max_io_queue', 100)),
            io_chunksize=int(cfg.get('io_chunksize', 262144)),
            use_threads=(str(cfg.get('use_threads', True)).lower() == "true")
        )

        # Determine content type
        content_type_map = {
            '.parquet': 'application/x-parquet',
            '.json': 'application/json',
            '.csv': 'text/csv'
        }
        file_ext = Path(key).suffix
        content_type = content_type_map.get(file_ext, 'application/octet-stream')

        # Build ExtraArgs
        extra_args = {
            'ContentType': content_type,
            'ServerSideEncryption': 'AES256',
            'StorageClass': 'INTELLIGENT_TIERING',
            'Metadata': metadata or {}
        }

        # Upload
        self.s3_client.upload_fileobj(
            Fileobj=data,
            Bucket='us-equity-datalake',
            Key=key,
            Config=transfer_config,
            ExtraArgs=extra_args
        )

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
        sec_rate_limiter,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> Dict[str, Optional[str]]:
        """
        Publish fundamental data for a single symbol for an entire year to S3.
        Uses concept-based extraction with approved_mapping.yaml.
        Returns dict with status for progress tracking.

        Storage: data/raw/fundamental/{symbol}/{YYYY}/fundamental.parquet
        Contains all quarterly/annual filings for the year (no forward fill).

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B') - used for storage path
        :param year: Year to fetch data for
        :param cik: CIK string (or None to skip)
        :param sec_rate_limiter: Rate limiter for SEC API
        :param concepts: Optional list of concepts to fetch. If None, fetches all from config.
        :param config_path: Optional path to approved_mapping.yaml
        :return: Dict with status info
        """
        try:
            if cik is None:
                self.logger.warning(f'Skipping {sym}: No CIK found')
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'No CIK found for {sym} in {year}',
                    'cik': None
                }

            # Rate limit before making SEC API request
            sec_rate_limiter.acquire()

            # Load concept mappings if not provided
            if concepts is None:
                if config_path is None:
                    config_path = Path("configs/approved_mapping.yaml")

                with open(config_path) as f:
                    mappings = yaml.safe_load(f)
                    concepts = list(mappings.keys())
                    self.logger.debug(f"Loaded {len(concepts)} concepts from {config_path}")

            # Fetch from SEC EDGAR API using concept-based extraction
            fnd = Fundamental(cik=cik, symbol=sym)

            # Collect data for each concept
            fields_dict = {}
            concepts_found = []
            concepts_missing = []

            for concept in concepts:
                try:
                    dps = fnd.get_concept_data(concept)
                    if dps:
                        fields_dict[concept] = fnd.get_value_tuple(dps)
                        concepts_found.append(concept)
                    else:
                        # Add missing concept with empty list to ensure column exists with null values
                        fields_dict[concept] = []
                        concepts_missing.append(concept)
                except Exception as e:
                    self.logger.debug(f"Failed to extract concept '{concept}' for {sym} (CIK {cik}): {e}")
                    # Add failed concept with empty list to ensure column exists with null values
                    fields_dict[concept] = []
                    concepts_missing.append(concept)

            # If no data found for any concept, return skipped
            if not fields_dict:
                self.logger.warning(f'No fundamental data found for {sym} (CIK {cik}) in {year}')
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'No fundamental data available for {sym} in {year}',
                    'cik': cik
                }

            # Log coverage statistics
            self.logger.debug(
                f"{sym} (CIK {cik}): {len(concepts_found)}/{len(concepts)} concepts available "
                f"({len(concepts_missing)} missing)"
            )

            # Define date range for the ENTIRE YEAR
            start_day = f"{year}-01-01"
            end_day = f"{year}-12-31"

            # Collect fields into DataFrame (only actual quarterly filings, no forward-fill)
            combined_df = fnd.collect_fields_raw(start_day, end_day, fields_dict)

            # Convert timestamp to string for consistency
            if len(combined_df) > 0 and 'timestamp' in combined_df.columns:
                combined_df = combined_df.with_columns(
                    pl.col('timestamp').dt.strftime('%Y-%m-%d')
                )

            # Check if DataFrame is empty
            if len(combined_df) == 0:
                self.logger.info(f'No fundamental data for {sym} (CIK {cik}) in {year} - likely no filings this year')
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
            # Use Alpaca format symbol (e.g., 'BRK.B') for consistency with ticks storage
            s3_key = f"data/raw/fundamental/{sym}/{year}/fundamental.parquet"
            s3_metadata = {
                'symbol': sym,
                'year': str(year),
                'data_type': 'fundamental',
                'quarters': str(len(combined_df)),
                'concepts_found': str(len(concepts_found)),
                'concepts_total': str(len(concepts))
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None, 'cik': cik}

        except requests.RequestException as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.warning(f'Failed to fetch data for {sym}{cik_str}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e), 'cik': cik}
        except ValueError as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.error(f'Invalid data for {sym}{cik_str}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e), 'cik': cik}
        except Exception as e:
            cik_str = f" (CIK {cik})" if cik else ""
            self.logger.error(f'Unexpected error for {sym}{cik_str}: {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e), 'cik': cik}
