import io
import json
import datetime as dt
from typing import List
from pathlib import Path
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from boto3.s3.transfer import TransferConfig
import polars as pl

from utils.logger import setup_logger
from utils.mapping import symbol_cik_mapping
from storage.config_loader import UploadConfig
from storage.s3_client import S3Client
from collection.fundamental import Fundamental
from collection.ticks import Ticks
from storage.validation import Validator


class UploadApp:
    def __init__(self, symbol_file: str="universe_top3000.txt"):
        # Load symbols
        self.symbol_path = Path(f"data/symbols/{symbol_file}")
        self.sec_symbols = self._load_symbols('sec')
        self.alpaca_symbols = self._load_symbols('alpaca')
        self.cik_map = symbol_cik_mapping()

        # Setup config and client
        self.config = UploadConfig()
        self.client = S3Client().client

        # Setup logger
        self.logger = setup_logger(
            name=f"uploadapp",
            log_dir=Path("data/logs/upload"),
            level=logging.INFO,
            console_output = True
        )

        self.validator = Validator(self.client)

        # Load trading calendar
        self.calendar_path = Path("data/calendar/master.parquet")
        self.trading_days = self._load_trading_days()

    def _load_symbols(self, sym_type: str) -> List[str]:
        """
        Load symbol as list from file
        SEC uses '-' as separator ('BRK-B'), Alpaca uses '.' ('BRK.B')

        :param sym_type: "sec" or "alpaca"
        """
        symbols = []
        with open(self.symbol_path, 'r') as file:
            if sym_type == "alpaca":
                for line in file:
                    symbol = line.strip()
                    symbols.append(symbol)
            elif sym_type == "sec":
                for line in file:
                    symbol = line.strip().replace('.', '-')
                    symbols.append(symbol)
            else:
                msg = f"Expected sym_type: 'sec' or 'alpaca', get {sym_type}"
                raise ValueError(msg)

        return symbols

    def _load_trading_days(self, year: int = 2024) -> List[str]:
        """
        Load trading days from master calendar for a specific year.

        :param year: Year to load trading days for (default: 2024)
        :return: List of trading days in 'YYYY-MM-DD' format
        """
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        df = (
            pl.scan_parquet(self.calendar_path)
            .filter(pl.col('Date').is_between(start_date, end_date))
            .select('Date')
            .collect()
        )

        return [d.strftime('%Y-%m-%d') for d in df['Date'].to_list()]

    # ===========================
    # Upload ticks
    # ===========================
    def _process_symbol_daily_ticks(self, sym: str, year: int, overwrite: bool = False) -> dict:
        """
        Process daily ticks for a single symbol.
        Returns dict with status for progress tracking.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param overwrite: If True, skip existence check and overwrite existing data
        """
        # Convert to SEC format for S3 key (BRK.B -> BRK-B)
        sec_symbol = sym.replace('.', '-')

        if not overwrite and self.validator.data_exists(sec_symbol, 'ticks', year):
            return {'symbol': sym, 'status': 'canceled', 'error': f'Symbol {sym} for year {year} already exists'}
        try:
            # Fetch from Alpaca API using Alpaca format (with '.')
            ticks = Ticks(sym)
            daily_df = ticks.collect_daily_ticks(year=year)

            # Check if all data columns are null (no actual data available)
            if daily_df['open'].is_null().all():
                return {'symbol': sym, 'status': 'skipped', 'error': 'No data available'}

            # Setup S3 message
            buffer = io.BytesIO()
            daily_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            # Use SEC format (with '-') in S3 key
            s3_key = f"data/raw/ticks/daily/{sec_symbol}/{year}/ticks.parquet"
            s3_metadata = {
                'symbol': sec_symbol,
                'year': str(year),
                'data_type': 'ticks'
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
            self.logger.warning(f'Failed to fetch daily ticks for {sym}: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error for {sym}: {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def daily_ticks(self, overwrite: bool = False):
        """
        Upload daily ticks for all symbols sequentially (no concurrency to avoid rate limits).

        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        year = 2024

        total = len(self.alpaca_symbols)
        completed = 0
        success = 0
        failed = 0
        canceled = 0
        skipped = 0

        self.logger.info(f"Starting daily ticks upload for {total} symbols (sequential processing, overwrite={overwrite})")

        for sym in self.alpaca_symbols:
            result = self._process_symbol_daily_ticks(sym, year, overwrite=overwrite)
            completed += 1

            if result['status'] == 'success':
                success += 1
            elif result['status'] == 'canceled':
                canceled += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1

            # Progress logging every 10 symbols
            if completed % 10 == 0:
                self.logger.info(f"Progress: {completed}/{total} ({success} success, {failed} failed, {canceled} canceled, {skipped} skipped)")

        self.logger.info(f"Daily ticks upload completed: {success} success, {failed} failed, {canceled} canceled, {skipped} skipped out of {total} total")

    def _process_symbol_minute_ticks(self, sym: str, trade_day: str, overwrite: bool = False) -> dict:
        """
        Process minute ticks for a single symbol and trading day.
        Returns dict with status for progress tracking.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param overwrite: If True, skip existence check and overwrite existing data
        """
        # Convert to SEC format for S3 key (BRK.B -> BRK-B)
        sec_symbol = sym.replace('.', '-')

        if not overwrite and self.validator.data_exists(sec_symbol, 'ticks', day=trade_day):
            return {'symbol': sym, 'day': trade_day, 'status': 'canceled', 'error': f'Symbol {sym} for day {trade_day} already exists'}
        try:
            # Fetch from Alpaca API using Alpaca format (with '.')
            ticks = Ticks(sym)
            minute_df = ticks.collect_minute_ticks(trade_day=trade_day)

            # Skip if DataFrame is empty (no data available)
            if len(minute_df) == 0:
                return {'symbol': sym, 'day': trade_day, 'status': 'skipped', 'error': 'No data available'}

            # Setup S3 message
            buffer = io.BytesIO()
            minute_df.write_parquet(buffer)
            buffer.seek(0)

            # Parse date for S3 key
            date_obj = dt.datetime.strptime(trade_day, '%Y-%m-%d').date()
            year = date_obj.strftime('%Y')
            month = date_obj.strftime('%m')
            day = date_obj.strftime('%d')

            s3_data = buffer
            # Use SEC format (with '-') in S3 key
            s3_key = f"data/raw/ticks/minute/{sec_symbol}/{year}/{month}/{day}/ticks.parquet"
            s3_metadata = {
                'symbol': sec_symbol,
                'trade_day': trade_day,
                'data_type': 'ticks'
            }
            # Allow list or dict as metadata value
            s3_metadata_prepared = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in s3_metadata.items()
            }

            # Publish onto AWS S3
            self.upload_fileobj(s3_data, s3_key, s3_metadata_prepared)

            return {'symbol': sym, 'day': trade_day, 'status': 'success', 'error': None}

        except requests.RequestException as e:
            self.logger.warning(f'Failed to fetch minute ticks for {sym} on {trade_day}: {e}')
            return {'symbol': sym, 'day': trade_day, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error for {sym} on {trade_day}: {e}', exc_info=True)
            return {'symbol': sym, 'day': trade_day, 'status': 'failed', 'error': str(e)}

    def minute_ticks(self, overwrite: bool = False):
        """
        Upload minute ticks for all symbols and trading days sequentially (no concurrency to avoid rate limits).

        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        total_symbols = len(self.alpaca_symbols)
        total_days = len(self.trading_days)
        total_tasks = total_symbols * total_days

        completed = 0
        success = 0
        failed = 0
        canceled = 0
        skipped = 0

        self.logger.info(f"Starting minute ticks upload for {total_symbols} symbols × {total_days} days = {total_tasks} tasks (sequential processing, overwrite={overwrite})")

        for sym in self.alpaca_symbols:
            for day in self.trading_days:
                result = self._process_symbol_minute_ticks(sym, day, overwrite=overwrite)
                completed += 1

                if result['status'] == 'success':
                    success += 1
                elif result['status'] == 'canceled':
                    canceled += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                else:
                    failed += 1

                # Progress logging every 100 tasks
                if completed % 100 == 0:
                    self.logger.info(f"Progress: {completed}/{total_tasks} ({success} success, {failed} failed, {canceled} canceled, {skipped} skipped)")

        self.logger.info(f"Minute ticks upload completed: {success} success, {failed} failed, {canceled} canceled, {skipped} skipped out of {total_tasks} total")

    # ===========================
    # Upload fundamental
    # ===========================
    def _process_symbol_fundamental(self, sym: str, year: int, dei_fields: List[str], gaap_fields: List[str], overwrite: bool = False) -> dict:
        """
        Process fundamental data for a single symbol.
        Returns dict with status for progress tracking.

        :param overwrite: If True, skip existence check and overwrite existing data
        """
        if not overwrite and self.validator.data_exists(sym, 'fundamental', year):
            return {'symbol': sym, 'status': 'canceled', 'error': f'Symbol {sym} for year {year} already exists'}
        try:
            cik = self.cik_map[sym]

            # Fetch from SEC EDGAR API
            fnd = Fundamental(cik, sym)

            # Load data on RAM
            dei_df = fnd.collect_fields(year=year, fields=dei_fields, location='dei')
            gaap_df = fnd.collect_fields(year=year, fields=gaap_fields, location='us-gaap')

            # Merge on Date column
            combined_df = dei_df.join(
                gaap_df,
                on='Date',
                how='inner'
            )

            # Setup S3 message
            buffer = io.BytesIO()
            combined_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            s3_key = f"data/raw/fundamental/{sym}/{year}/fundamental.parquet"
            s3_metadata = {
                'symbol': sym,
                'year': str(year),
                'data_type': 'fundamental'
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
            self.logger.warning(f'Failed to fetch data for {sym} (CIK {cik}): {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except KeyError as e:
            self.logger.warning(f'Invalid symbol ({sym}) for CIK mapping: {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except ValueError as e:
            self.logger.error(f'Invalid data for {sym} (CIK {cik}): {e}')
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}
        except Exception as e:
            self.logger.error(f'Unexpected error for {sym} (CIK {cik}): {e}', exc_info=True)
            return {'symbol': sym, 'status': 'failed', 'error': str(e)}

    def fundamental(self, max_workers: int = 20, overwrite: bool = False):
        """
        Upload fundamental data for all symbols using threading.

        :param max_workers: Number of concurrent threads (default: 20)
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        # Fields
        dei_fields = self.config.dei_fields
        gaap_fields = self.config.us_gaap_fields
        year = 2024

        total = len(self.sec_symbols)
        completed = 0
        success = 0
        failed = 0
        canceled = 0

        self.logger.info(f"Starting fundamental upload for {total} symbols with {max_workers} workers (overwrite={overwrite})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._process_symbol_fundamental, sym, year, dei_fields, gaap_fields, overwrite): sym
                for sym in self.sec_symbols
            }

            # Process completed tasks
            for future in as_completed(future_to_symbol):
                result = future.result()
                completed += 1

                if result['status'] == 'success':
                    success += 1
                elif result['status'] == 'canceled':
                    canceled += 1
                else:
                    failed += 1

                # Progress logging every 10 symbols
                if completed % 10 == 0:
                    self.logger.info(f"Progress: {completed}/{total} ({success} success, {failed} failed, {canceled} canceled)")

        self.logger.info(f"Fundamental upload completed: {success} success, {failed} failed, {canceled} canceled out of {total} total")

    def upload_fileobj(
            self, 
            data: io.BytesIO, 
            key: str, 
            metadata: dict=None
        ) -> None:
        """Upload file object to S3 with proper configuration"""

        # Define transfer config
        cfg = self.config.transfer
        transfer_config = TransferConfig(
            multipart_threshold=int(eval(cfg.get('multipart_threshold', 10*1024*1024))),
            max_concurrency=int(cfg.get('max_concurrency', 5)),
            multipart_chunksize=int(eval(cfg.get('multipart_chunksize', 10*1024*1024))),
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
        self.client.upload_fileobj(
            Fileobj=data,
            Bucket='us-equity-datalake',
            Key=key,
            Config=transfer_config,
            ExtraArgs=extra_args
        )

if __name__ == "__main__":
    import time
    app = UploadApp()
    start = time.time()
    app.daily_ticks(overwrite=True)
    print(f"Execution time: {time.time() - start:.2f} seconds")