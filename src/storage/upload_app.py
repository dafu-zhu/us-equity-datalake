import io
import json
from typing import List
from pathlib import Path
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from boto3.s3.transfer import TransferConfig

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

    # ===========================
    # Upload ticks
    # ===========================
    def _process_symbol_daily_ticks(self, sym: str, year: int) -> dict:
        """
        Process daily ticks for a single symbol.
        Returns dict with status for progress tracking.
        """
        try:
            # Fetch from Alpaca API
            ticks = Ticks(sym)
            daily_df = ticks.collect_daily_ticks(year=year)

            # Setup S3 message
            buffer = io.BytesIO()
            daily_df.write_parquet(buffer)
            buffer.seek(0)

            s3_data = buffer
            s3_key = f"data/raw/ticks/daily/{sym}/{year}/ticks.parquet"
            s3_metadata = {
                'symbol': sym,
                'year': str(year),
                'data_type': 'daily_ticks'
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

    def daily_ticks(self, year: int = 2024, max_workers: int = 10):
        """
        Upload daily ticks for all symbols using threading.

        :param year: Year to fetch data for (default: 2024)
        :param max_workers: Number of concurrent threads (default: 10)
        """
        total = len(self.alpaca_symbols)
        completed = 0
        success = 0
        failed = 0

        self.logger.info(f"Starting daily ticks upload for {total} symbols with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._process_symbol_daily_ticks, sym, year): sym
                for sym in self.alpaca_symbols
            }

            # Process completed tasks
            for future in as_completed(future_to_symbol):
                result = future.result()
                completed += 1

                if result['status'] == 'success':
                    success += 1
                else:
                    failed += 1

                # Progress logging every 10 symbols
                if completed % 10 == 0:
                    self.logger.info(f"Progress: {completed}/{total} ({success} success, {failed} failed)")

        self.logger.info(f"Daily ticks upload completed: {success} success, {failed} failed out of {total} total")

    def minute_ticks(self):
        pass

    # ===========================
    # Upload fundamental
    # ===========================
    def _process_symbol_fundamental(self, sym: str, year: int, dei_fields: List[str], gaap_fields: List[str]) -> dict:
        """
        Process fundamental data for a single symbol.
        Returns dict with status for progress tracking.
        """
        if self.validator.data_exists(sym, 'fundamental', year):
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

    def fundamental(self, max_workers: int = 20):
        """
        Upload fundamental data for all symbols using threading.

        :param max_workers: Number of concurrent threads (default: 10)
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

        self.logger.info(f"Starting fundamental upload for {total} symbols with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._process_symbol_fundamental, sym, year, dei_fields, gaap_fields): sym
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
    app.fundamental()
    print(f"Execution time: {time.time() - start:.2f} seconds")