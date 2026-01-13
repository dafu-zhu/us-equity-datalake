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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, cast
import requests
import polars as pl
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


class DataPublishers:
    """
    Handles publishing market data to S3 storage.
    """

    def __init__(
        self,
        s3_client,
        upload_config,
        logger: logging.Logger,
        data_collectors,
        security_master,
        bucket_name: Optional[str] = None,
        alpaca_start_year: int = 2025
    ):
        """
        Initialize data publishers.

        :param s3_client: Boto3 S3 client instance
        :param upload_config: UploadConfig instance with transfer settings
        :param logger: Logger instance
        :param data_collectors: Data collectors instance
        :param security_master: SecurityMaster instance for symbol→security_id resolution
        :param bucket_name: S3 bucket name (defaults to environment variable or 'us-equity-datalake')
        """
        self.s3_client = s3_client
        self.upload_config = upload_config
        self.logger = logger
        self.data_collectors = data_collectors
        self.security_master = security_master
        self.alpaca_start_year = alpaca_start_year
        # Allow bucket_name to be passed as parameter or from environment variable
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME', 'us-equity-datalake')

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
            Bucket=self.bucket_name,
            Key=key,
            Config=transfer_config,
            ExtraArgs=extra_args
        )

    def publish_daily_ticks(
        self,
        sym: str,
        year: int,
        security_id: int,
        df: Optional[pl.DataFrame],
        month: Optional[int] = None,
        by_year: bool = False,
        max_workers: int = 6,
        year_df: Optional[pl.DataFrame] = None
    ) -> Dict[str, Optional[str]]:
        """
        Publish daily ticks for a single symbol to S3.

        Supports both monthly and yearly partitioning:
        - Monthly: data/raw/ticks/daily/{security_id}/{YYYY}/{MM}/ticks.parquet
        - Yearly: data/raw/ticks/daily/{security_id}/{YYYY}/ticks.parquet

        Current year uses monthly partitions (optimizes updates).
        Completed years use yearly partitions (optimizes queries).

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :param security_id: Security ID from SecurityMaster
        :param df: Polars DataFrame with daily ticks data
        :param month: Optional month (1-12) for monthly partitioning
        :param by_year: If True, collect year data once and publish monthly partitions in parallel
        :param max_workers: Max workers for parallel monthly uploads (by_year only)
        :return: Dict with status info
        """
        if by_year:
            return self._publish_daily_ticks_by_year(sym, year, security_id, max_workers=max_workers, year_df=year_df)

        if df is None:
            raise ValueError("df is required when by_year=False")

        return self._publish_daily_ticks_df(sym, year, df, security_id, month=month)

    def _publish_daily_ticks_by_year(
        self,
        sym: str,
        year: int,
        security_id: int,
        max_workers: int = 6,
        year_df: Optional[pl.DataFrame] = None # type: ignore
    ) -> Dict[str, Optional[str]]:
        if year_df is None:
            year_df: pl.DataFrame = self.data_collectors.collect_daily_ticks_year(sym, year)
        if len(year_df) == 0:
            return {'symbol': sym, 'status': 'skipped', 'error': 'No data available'}

        futures = {}
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for month in range(1, 13):
                month_df = self.data_collectors.collect_daily_ticks_month(
                    sym,
                    year,
                    month,
                    year_df=year_df
                )
                futures[executor.submit(
                    self._publish_daily_ticks_df,
                    sym,
                    year,
                    month_df,
                    security_id,
                    month
                )] = month

            for future in as_completed(futures):
                results.append(future.result())

        failed = [r for r in results if r.get('status') == 'failed']
        skipped = [r for r in results if r.get('status') == 'skipped']

        if failed:
            return {
                'symbol': sym,
                'status': 'failed',
                'error': f'{len(failed)} month(s) failed'
            }
        if len(skipped) == 12:
            return {'symbol': sym, 'status': 'skipped', 'error': 'No data available'}

        return {'symbol': sym, 'status': 'success', 'error': None}

    def _publish_daily_ticks_df(
        self,
        sym: str,
        year: int,
        df: pl.DataFrame,
        security_id: int,
        month: Optional[int] = None
    ) -> Dict[str, Optional[str]]:
        try:
            # Check if DataFrame is empty (no rows fetched)
            if len(df) == 0:
                return {'symbol': sym, 'status': 'skipped', 'error': 'No data available'}

            # Choose storage path based on partitioning
            if month is not None:
                # Monthly partition (for current year daily updates)
                s3_key = f"data/raw/ticks/daily/{security_id}/{year}/{month:02d}/ticks.parquet"

                # Setup S3 message (Parquet format)
                buffer = io.BytesIO()
                df.write_parquet(buffer)
                buffer.seek(0)

                s3_metadata = {
                    'security_id': str(security_id),
                    'symbols': [sym],
                    'year': str(year),
                    'month': f"{month:02d}",
                    'data_type': 'daily_ticks',
                    'source': 'crsp' if year < self.alpaca_start_year else 'alpaca',
                    'trading_days': str(len(df)),
                    'partition_type': 'monthly'
                }
            else:
                # History file (all completed years consolidated)
                s3_key = f"data/raw/ticks/daily/{security_id}/history.parquet"

                # Read existing history and append new year data
                try:
                    response = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=s3_key
                    )
                    history_df = pl.read_parquet(response['Body'])

                    # Remove existing year data (if any) and append new
                    history_df = history_df.filter(
                        ~pl.col('timestamp').str.starts_with(str(year))
                    )
                    combined_df = pl.concat([history_df, df]).sort('timestamp')
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        # No existing history, use new data only
                        combined_df = df
                    else:
                        raise

                # Setup S3 message
                buffer = io.BytesIO()
                combined_df.write_parquet(buffer)
                buffer.seek(0)

                s3_metadata = {
                    'security_id': str(security_id),
                    'symbols': [sym],
                    'data_type': 'daily_ticks',
                    'source': 'mixed',
                    'trading_days': str(len(combined_df)),
                    'partition_type': 'history'
                }

            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

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

    def publish_daily_ticks_to_history(
        self,
        security_id: int,
        df: pl.DataFrame,
        symbol: str = ""
    ) -> Dict[str, Optional[str]]:
        """
        Publish complete daily ticks history for a security_id.

        Unlike _publish_daily_ticks_df, this writes complete history data
        without year-by-year append logic. Used for bulk backfill operations.

        Storage: data/raw/ticks/daily/{security_id}/history.parquet

        :param security_id: Security ID from SecurityMaster
        :param df: Complete Polars DataFrame with all daily ticks
        :param symbol: Symbol name for logging (optional)
        :return: Dict with status info
        """
        try:
            if len(df) == 0:
                return {
                    'symbol': symbol,
                    'security_id': security_id,
                    'status': 'skipped',
                    'error': 'No data available'
                }

            s3_key = f"data/raw/ticks/daily/{security_id}/history.parquet"

            # Sort by timestamp
            df = df.sort('timestamp')

            # Write to buffer
            buffer = io.BytesIO()
            df.write_parquet(buffer)
            buffer.seek(0)

            s3_metadata = {
                'security_id': str(security_id),
                'symbols': [symbol] if symbol else [],
                'data_type': 'daily_ticks',
                'source': 'crsp',
                'trading_days': str(len(df)),
                'partition_type': 'history'
            }

            # Prepare metadata
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Upload to S3
            self.upload_fileobj(buffer, s3_key, s3_metadata_prepared)

            return {
                'symbol': symbol,
                'security_id': security_id,
                'status': 'success',
                'error': None
            }

        except Exception as e:
            self.logger.error(f'Error publishing history for sid={security_id}: {e}')
            return {
                'symbol': symbol,
                'security_id': security_id,
                'status': 'failed',
                'error': str(e)
            }

    def minute_ticks_worker(
        self,
        data_queue: queue.Queue,
        stats: dict,
        stats_lock: threading.Lock
    ):
        """
        Worker thread that consumes fetched minute tick data and uploads to S3.

        Storage: data/raw/ticks/minute/{security_id}/{YYYY}/{MM}/{DD}/ticks.parquet

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

                    # Resolve symbol to security_id at trade day
                    security_id = self.security_master.get_security_id(sym, trade_day)

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

                    s3_key = f"data/raw/ticks/minute/{security_id}/{year}/{month}/{day}/ticks.parquet"
                    s3_metadata = {
                        'security_id': str(security_id),
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

    def get_fundamental_metadata(self, cik: str) -> Optional[Dict[str, str]]:
        """
        Get metadata for existing fundamental data from S3.

        :param cik: CIK string
        :return: Metadata dict or None if file doesn't exist
        """
        s3_key = f"data/raw/fundamental/{cik}/fundamental.parquet"
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return response.get('Metadata', {})
        except Exception:
            return None

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

        Storage: data/raw/fundamental/{cik}/fundamental.parquet
        Stored in long format with [symbol, as_of_date, accn, form, concept, value, start, end, frame, is_instant].

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B') - stored in metadata and data
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

            # Calculate latest filing info for metadata tracking
            latest_date = combined_df.select(pl.col("as_of_date").max()).item()
            latest_accn = combined_df.filter(pl.col("as_of_date") == latest_date).select("accn").unique().item()

            # Setup S3 message
            buffer = io.BytesIO()
            combined_df.write_parquet(buffer)
            buffer.seek(0)

            s3_key = f"data/raw/fundamental/{cik}/fundamental.parquet"
            concepts_found = combined_df.select(pl.col("concept").n_unique()).item()
            s3_metadata = {
                'symbol': sym,
                'cik': cik,
                'data_type': 'fundamental',
                'rows': str(len(combined_df)),
                'concepts_found': str(concepts_found),
                'concepts_total': str(concepts_total),
                'start_date': start_date,
                'end_date': end_date,
                'latest_filing_date': str(latest_date),
                'latest_accn': str(latest_accn)
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
        - data/derived/features/fundamental/{cik}/ttm.parquet (long)
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

            s3_key = f"data/derived/features/fundamental/{cik}/ttm.parquet"
            concepts_found = ttm_df.select(pl.col("concept").n_unique()).item()
            s3_metadata = {
                'symbol': sym,
                'cik': cik,
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
        derived_df: pl.DataFrame,
        cik: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """
        Publish derived fundamental data for a single symbol for a date range to S3.

        Storage: data/derived/features/fundamental/{cik}/metrics.parquet
        Contains ONLY derived metrics (long format with symbol/as_of_date/metric/value).

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param start_date: Start date (YYYY-MM-DD)
        :param end_date: End date (YYYY-MM-DD)
        :param derived_df: Derived DataFrame (from compute_derived)
        :return: Dict with status info
        """
        try:
            # Check if DataFrame is empty or CIK is None
            if len(derived_df) == 0:
                self.logger.info(f'No derived fundamental data for {sym}')
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'Empty derived DataFrame for {sym}'
                }

            if cik is None:
                self.logger.warning(f'Skipping {sym}: No CIK found')
                return {
                    'symbol': sym,
                    'status': 'skipped',
                    'error': f'No CIK found for {sym}'
                }

            # Setup S3 message
            buffer = io.BytesIO()
            derived_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            s3_key = f"data/derived/features/fundamental/{cik}/metrics.parquet"
            s3_metadata = {
                'symbol': sym,
                'cik': cik,
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
