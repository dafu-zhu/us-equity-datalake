"""
Data publishing functionality for uploading market data to S3.

This module handles publishing collected data to S3 storage:
- Daily ticks (yearly Parquet files)
- Minute ticks (daily Parquet files)
- Fundamental data (yearly Parquet files)
- Fundamental TTM features (yearly Parquet files)
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
from boto3.s3.transfer import TransferConfig



class DataPublishers:
    """
    Handles publishing market data to S3 storage.
    """

    def __init__(
        self,
        s3_client,
        upload_config,
        logger: logging.Logger,
        data_collectors
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
        self.data_collectors = data_collectors

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
            '.csv': 'text/csv',
            '.txt': 'text/plain'
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
        start_date: str,
        end_date: str,
        cik: Optional[str],
        sec_rate_limiter,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> Dict[str, Optional[str]]:
        """
        Publish fundamental data for a single symbol for a date range to S3.
        Uses concept-based extraction with approved_mapping.yaml.
        Returns dict with status for progress tracking.

        Storage: data/raw/fundamental/{symbol}/fundamental.parquet
        Stored in long format with [symbol, as_of_date, accn, form, concept, value, start, end, frame, is_instant].

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B') - used for storage path
        :param start_date: Start date (YYYY-MM-DD) for filing date filter
        :param end_date: End date (YYYY-MM-DD) for filing date filter
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
                    'error': f'No CIK found for {sym} from {start_date} to {end_date}',
                    'cik': None
                }

            # Rate limit before making SEC API request
            sec_rate_limiter.acquire()

            combined_df = self.data_collectors.collect_fundamental_long(
                cik=cik,
                start_date=start_date,
                end_date=end_date,
                symbol=sym,
                concepts=concepts,
                config_path=config_path
            )

            if len(combined_df) == 0:
                self.logger.warning(
                    f'No fundamental data found for {sym} (CIK {cik}) '
                    f"from {start_date} to {end_date}"
                )
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'No fundamental data available for {sym} from {start_date} to {end_date}',
                    'cik': cik
                }

            concepts_total = len(self.data_collectors._load_concepts(concepts, config_path))
            # Setup S3 message
            buffer = io.BytesIO()
            combined_df.write_parquet(buffer)
            buffer.seek(0)

            s3_key = f"data/raw/fundamental/{sym}/fundamental.parquet"
            concepts_found = combined_df.select(pl.col("concept").n_unique()).item()
            s3_metadata = {
                'symbol': sym,
                'data_type': 'fundamental',
                'rows': str(len(combined_df)),
                'concepts_found': str(concepts_found),
                'concepts_total': str(concepts_total),
                'start_date': start_date,
                'end_date': end_date
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

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

    def publish_ttm_fundamental(
        self,
        sym: str,
        start_date: str,
        end_date: str,
        cik: Optional[str],
        sec_rate_limiter,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> Dict[str, Optional[str]]:
        """
        Publish TTM fundamental data for a single symbol for a date range to S3.
        Computed in-memory from long-format raw data.

        Storage:
        - data/derived/features/fundamental/{symbol}/ttm.parquet (long)
        """
        try:
            if cik is None:
                self.logger.warning(f'Skipping {sym}: No CIK found')
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'No CIK found for {sym} from {start_date} to {end_date}',
                    'cik': None
                }

            sec_rate_limiter.acquire()

            concepts_total = len(self.data_collectors._load_concepts(concepts, config_path))

            ttm_df = self.data_collectors.collect_ttm_long_range(
                cik=cik,
                start_date=start_date,
                end_date=end_date,
                symbol=sym,
                concepts=concepts,
                config_path=config_path
            )
            if len(ttm_df) == 0:
                self.logger.info(
                    f'No TTM data for {sym} (CIK {cik}) from {start_date} to {end_date}'
                )
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'No TTM data available for {sym} from {start_date} to {end_date}',
                    'cik': cik
                }

            buffer = io.BytesIO()
            ttm_df.write_parquet(buffer)
            buffer.seek(0)

            s3_key = f"data/derived/features/fundamental/{sym}/ttm.parquet"
            concepts_found = ttm_df.select(pl.col("concept").n_unique()).item()
            s3_metadata = {
                'symbol': sym,
                'data_type': 'ttm',
                'rows': str(len(ttm_df)),
                'concepts_found': str(concepts_found),
                'concepts_total': str(concepts_total),
                'start_date': start_date,
                'end_date': end_date
            }
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

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

    def publish_derived_fundamental(
        self,
        sym: str,
        start_date: str,
        end_date: str,
        derived_df: pl.DataFrame
    ) -> Dict[str, Optional[str]]:
        """
        Publish derived fundamental data for a single symbol for a date range to S3.

        Storage: data/derived/features/fundamental/{symbol}/metrics.parquet
        Contains ONLY derived metrics (long format with symbol/as_of_date/metric/value).

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param start_date: Start date (YYYY-MM-DD)
        :param end_date: End date (YYYY-MM-DD)
        :param derived_df: Derived DataFrame (from compute_derived)
        :return: Dict with status info
        """
        try:
            # Check if DataFrame is empty
            if len(derived_df) == 0:
                self.logger.info(f'No derived fundamental data for {sym}')
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'Empty derived DataFrame for {sym}'
                }

            # Setup S3 message
            buffer = io.BytesIO()
            derived_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            s3_key = f"data/derived/features/fundamental/{sym}/metrics.parquet"
            s3_metadata = {
                'symbol': sym,
                'data_type': 'derived_metrics',
                'rows': str(len(derived_df)),
                'columns': str(derived_df.shape[1]),
                'start_date': start_date,
                'end_date': end_date
            }

            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'status': 'success', 'error': None}

        except Exception as e:
            self.logger.error(f'Unexpected error publishing derived for {sym}: {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def publish_top_3000(
        self,
        year: int,
        month: int,
        as_of: str,
        symbols: List[str],
        source: str
    ) -> Dict[str, Optional[str]]:
        """
        Publish monthly top 3000 symbols to S3 as a newline-delimited text file.

        Storage: data/symbols/{YYYY}/{MM}/top3000.txt
        """
        try:
            if not symbols:
                return {
                    'status': 'skipped',
                    'error': 'No symbols provided',
                    'year': str(year),
                    'month': f"{month:02d}"
                }

            content = "\n".join(symbols) + "\n"
            buffer = io.BytesIO(content.encode("utf-8"))
            buffer.seek(0)

            s3_key = f"data/symbols/{year}/{month:02d}/top3000.txt"
            s3_metadata = {
                'year': str(year),
                'month': f"{month:02d}",
                'as_of': as_of,
                'data_type': 'top3000',
                'count': str(len(symbols)),
                'source': source
            }
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

            return {
                'status': 'success',
                'error': None,
                'year': str(year),
                'month': f"{month:02d}"
            }
        except Exception as e:
            self.logger.error(
                f"Unexpected error publishing top3000 for {year}-{month:02d}: {e}",
                exc_info=True
            )
            return {
                'status': 'failed',
                'error': str(e),
                'year': str(year),
                'month': f"{month:02d}"
            }
