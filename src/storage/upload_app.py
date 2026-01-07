import os
from typing import List, Optional, Dict
from pathlib import Path
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import polars as pl

from utils.logger import setup_logger
from utils.calendar import TradingCalendar
from storage.config_loader import UploadConfig
from storage.s3_client import S3Client
from storage.validation import Validator
from storage.rate_limiter import RateLimiter
from storage.cik_resolver import CIKResolver
from storage.data_collectors import DataCollectors
from storage.data_publishers import DataPublishers
from collection.alpaca_ticks import Ticks
from collection.crsp_ticks import CRSPDailyTicks
from stock_pool.universe_manager import UniverseManager

load_dotenv()


class UploadApp:
    def __init__(self):
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

        # Initialize data fetchers once to reuse connections
        self.alpaca_ticks = Ticks()
        self.crsp_ticks = CRSPDailyTicks()

        # Initialize universe manager
        self.universe_manager = UniverseManager()

        # Initialize trading calendar
        self.calendar = TradingCalendar()

        # Load Alpaca Key
        ALPACA_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET
        }

        # Rate limiter for SEC EDGAR API (10 requests per second limit)
        self.sec_rate_limiter = RateLimiter(max_rate=9.5)  # Slightly under 10 to be safe

        # Initialize helper modules
        self.cik_resolver = CIKResolver(
            security_master=self.crsp_ticks.security_master,
            logger=self.logger
        )

        self.data_collectors = DataCollectors(
            crsp_ticks=self.crsp_ticks,
            alpaca_ticks=self.alpaca_ticks,
            alpaca_headers=self.headers,
            logger=self.logger
        )

        self.data_publishers = DataPublishers(
            s3_client=self.client,
            upload_config=self.config,
            logger=self.logger
        )

    # ===========================
    # Upload ticks
    # ===========================
    def _publish_single_daily_ticks(
            self,
            sym: str,
            year: int,
            overwrite: bool = False
        ) -> Dict[str, Optional[str]]:
        """
        Process daily ticks for a single symbol for entire year.
        Returns dict with status for progress tracking.

        Storage: data/raw/ticks/daily/{symbol}/{YYYY}/ticks.parquet
        Contains all trading days for the year (~252 days).

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :param overwrite: If True, skip existence check and overwrite existing data
        """
        # Validate: Check if data already exists (before collection)
        if not overwrite and self.validator.data_exists(sym, 'ticks', year):
            return {
                'symbol': sym,
                'status': 'canceled',
                'error': f'Symbol {sym} for {year} already exists'
            }

        # Fetch data for entire year
        df = self.data_collectors.collect_daily_ticks_year(sym, year)

        # Publish to S3
        return self.data_publishers.publish_daily_ticks(sym, year, df)

    def upload_daily_ticks(self, year: int, overwrite: bool = False):
        """
        Upload daily ticks for all symbols for an entire year.

        Storage strategy: data/raw/ticks/daily/{symbol}/{YYYY}/ticks.parquet
        - Stores all trading days for the year (~252 days)
        - Parquet format for efficient storage and querying
        - One file per symbol per year

        Uses CRSP for years < 2025, Alpaca for years >= 2025.
        Sequential processing to avoid rate limits.

        :param year: Year to fetch data for
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        # Load symbols for this year
        alpaca_symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')

        total = len(alpaca_symbols)
        completed = 0
        success = 0
        failed = 0
        canceled = 0
        skipped = 0

        data_source = 'crsp' if year < 2025 else 'alpaca'
        self.logger.info(f"Starting {year} daily ticks upload for {total} symbols (source={data_source}, year-based, sequential processing, overwrite={overwrite})")
        self.logger.info(f"Storage: data/raw/ticks/daily/{{symbol}}/{year}/ticks.parquet")

        for sym in alpaca_symbols:
            result = self._publish_single_daily_ticks(sym, year, overwrite=overwrite)
            completed += 1

            if result['status'] == 'success':
                success += 1
            elif result['status'] == 'canceled':
                canceled += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1

            # Progress logging every 50 symbols
            if completed % 50 == 0:
                self.logger.info(f"{year} Progress: {completed}/{total} ({success} success, {failed} failed, {canceled} canceled, {skipped} skipped)")

        self.logger.info(f"{year} Daily ticks upload completed ({data_source}): {success} success, {failed} failed, {canceled} canceled, {skipped} skipped out of {total} total")

    def upload_minute_ticks(self, year: int, month: int, overwrite: bool = False, num_workers: int = 50, chunk_size: int = 30, sleep_time: float = 0.2):
        """
        Upload minute ticks using bulk fetch + concurrent processing.
        Fetches 30 symbols at a time for the specified month, then processes concurrently.

        :param year: Year to fetch data for
        :param month: Month to fetch data for (1-12)
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        :param num_workers: Number of concurrent processing workers (default: 50)
        :param chunk_size: Number of symbols to fetch at once (default: 30)
        :param sleep_time: Sleep time between API requests in seconds (default: 0.2)
        """
        # Load symbols for this month
        alpaca_symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')

        # Update trading days for the specified month
        trading_days = self.calendar.load_trading_days(year, month)

        total_symbols = len(alpaca_symbols)
        total_days = len(trading_days)
        total_tasks = total_symbols * total_days

        # Shared statistics
        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()

        # Queue for passing data from parser to upload workers
        data_queue = queue.Queue(maxsize=200)

        self.logger.info(f"Starting {year}-{month:02d} minute ticks upload for {total_symbols} symbols × {total_days} days = {total_tasks} tasks")
        self.logger.info(f"Bulk fetching {chunk_size} symbols at a time | {num_workers} concurrent processors | sleep_time={sleep_time}s")

        # Start consumer threads
        workers = []
        for _ in range(num_workers):
            worker = threading.Thread(
                target=self.data_publishers.minute_ticks_worker,
                args=(data_queue, stats, stats_lock),
                daemon=True
            )
            worker.start()
            workers.append(worker)

        # Producer: Bulk fetch and parse
        try:
            for i in range(0, total_symbols, chunk_size):
                chunk = alpaca_symbols[i:i + chunk_size]
                chunk_num = i // chunk_size + 1

                # Pre-filter symbols that need data (skip if all days exist and overwrite=False)
                if not overwrite:
                    symbols_to_fetch = []
                    for sym in chunk:
                        # Check if any day is missing for this symbol
                        needs_fetch = False
                        for day in trading_days:
                            if not self.validator.data_exists(sym, 'ticks', day=day):
                                needs_fetch = True
                                break
                        if needs_fetch:
                            symbols_to_fetch.append(sym)
                        else:
                            # All days exist, mark them all as canceled
                            for day in trading_days:
                                with stats_lock:
                                    stats['canceled'] += 1
                                    stats['completed'] += 1

                    if not symbols_to_fetch:
                        self.logger.info(f"Skipping chunk {chunk_num}: all data already exists")
                        continue

                    self.logger.info(f"Fetching chunk {chunk_num}/{(total_symbols + chunk_size - 1) // chunk_size}: {len(symbols_to_fetch)}/{len(chunk)} symbols (skipped {len(chunk) - len(symbols_to_fetch)})")
                    chunk = symbols_to_fetch
                else:
                    self.logger.info(f"Fetching chunk {chunk_num}/{(total_symbols + chunk_size - 1) // chunk_size}: {len(chunk)} symbols")

                # Bulk fetch for the month
                start = time.perf_counter()
                symbol_bars = self.data_collectors.fetch_minute_month(chunk, year, month, sleep_time=sleep_time)
                self.logger.debug(f"fetch_minute_month: {time.perf_counter()-start:.2f}s")

                # Parse and organize by (symbol, day) using DataCollectors
                start = time.perf_counter()
                parsed_data = self.data_collectors.parse_minute_bars_to_daily(symbol_bars, trading_days)
                self.logger.debug(f"parse_minute_bars_to_daily: {time.perf_counter()-start:.2f}s")

                # Put parsed data into queue for upload
                for (sym, day), minute_df in parsed_data.items():
                    # Check if data is empty and update stats accordingly
                    if len(minute_df) == 0:
                        with stats_lock:
                            stats['skipped'] += 1
                            stats['completed'] += 1
                    else:
                        # Put in queue for upload
                        data_queue.put((sym, day, minute_df))

                        with stats_lock:
                            stats['completed'] += 1
                            completed = stats['completed']

                        # Progress logging
                        if completed % 100 == 0:
                            with stats_lock:
                                self.logger.info(
                                    f"Progress: {completed}/{total_tasks} "
                                    f"({stats['success']} success, {stats['failed']} failed, "
                                    f"{stats['canceled']} canceled, {stats['skipped']} skipped)"
                                )
        except Exception as e:
            self.logger.error(f"Error in bulk fetch/parse: {e}", exc_info=True)

        finally:
            # Signal workers to stop
            for _ in range(num_workers):
                data_queue.put(None)

            # Wait for all workers to finish
            for worker in workers:
                worker.join()

        self.logger.info(
            f"Minute ticks upload completed: {stats['success']} success, {stats['failed']} failed, "
            f"{stats['canceled']} canceled, {stats['skipped']} skipped out of {total_tasks} total"
        )

    # ===========================
    # Upload fundamental
    # ===========================
    def _process_symbol_fundamental(
            self,
            sym: str,
            year: int,
            overwrite: bool = False,
            cik: Optional[str] = None
        ) -> dict:
        """
        Process fundamental data for a single symbol for an entire year.
        Uses concept-based extraction with approved_mapping.yaml.
        Returns dict with status for progress tracking.

        Storage: data/raw/fundamental/{symbol}/{YYYY}/fundamental.parquet
        Contains all quarterly/annual filings for the year (no forward fill).

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :param overwrite: If True, skip existence check and overwrite existing data
        :param cik: Pre-fetched CIK (if None, will look up)
        """
        # Validate: Check if data already exists (before collection)
        if not overwrite and self.validator.data_exists(sym, 'fundamental', year):
            return {
                'symbol': sym,
                'status': 'canceled',
                'error': f'Symbol {sym} for {year} already exists'
            }

        # Use CIKResolver if CIK not provided
        if cik is None:
            reference_date = f"{year}-06-30"  # Mid-year reference
            cik = self.cik_resolver.get_cik(sym, reference_date, year=year)

        # Publish fundamental data using concept-based extraction
        return self.data_publishers.publish_fundamental(
            sym=sym,
            year=year,
            cik=cik,
            sec_rate_limiter=self.sec_rate_limiter
        )

    def upload_fundamental(self, year: int, max_workers: int = 50, overwrite: bool = False):
        """
        Upload fundamental data for all symbols for an entire year.
        Uses concept-based extraction with approved_mapping.yaml.

        Storage strategy: data/raw/fundamental/{symbol}/{YYYY}/fundamental.parquet
        - Stores all quarterly/annual filings for the year
        - No forward filling - only actual filed data
        - One file per symbol per year
        - Uses all concepts from approved_mapping.yaml

        Performance optimizations:
        1. Batch pre-fetch CIKs to avoid per-symbol database queries
        2. Rate limiting to maximize SEC API throughput (9.5 req/sec)
        3. Increased worker pool (50 workers) - rate limiter controls actual request rate
        4. CIK caching for repeated use

        :param year: Year to fetch data for
        :param max_workers: Number of concurrent threads (default: 50, rate limited to 9.5 req/sec)
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        start_time = time.time()

        # Load symbols for this year in Alpaca format
        alpaca_symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')

        total = len(alpaca_symbols)
        self.logger.info(f"Starting {year} fundamental upload for {total} symbols with {max_workers} workers (rate limited to 9.5 req/sec)")
        self.logger.info(f"Using concept-based extraction with approved_mapping.yaml")
        self.logger.info(f"Storage: data/raw/fundamental/{{symbol}}/{year}/fundamental.parquet")

        # OPTIMIZATION: Batch pre-fetch all CIKs before starting (avoids per-symbol DB queries)
        self.logger.info(f"Step 1/3: Pre-fetching CIKs for {total} symbols...")
        prefetch_start = time.time()
        cik_map = self.cik_resolver.batch_prefetch_ciks(alpaca_symbols, year, batch_size=100)
        prefetch_time = time.time() - prefetch_start
        self.logger.info(f"CIK pre-fetch completed in {prefetch_time:.1f}s ({total/prefetch_time:.1f} symbols/sec)")

        # Step 2: Filter to only symbols with valid CIKs
        self.logger.info(f"Step 2/3: Filtering symbols with valid CIKs...")
        symbols_with_cik = [sym for sym in alpaca_symbols if cik_map.get(sym) is not None]
        symbols_without_cik = [sym for sym in alpaca_symbols if cik_map.get(sym) is None]

        self.logger.info(
            f"Symbol filtering complete: {len(symbols_with_cik)}/{total} have CIKs, "
            f"{len(symbols_without_cik)} are non-SEC filers (will be skipped)"
        )

        # Log examples of skipped symbols with company names
        if len(symbols_without_cik) > 0:
            if len(symbols_without_cik) <= 30:
                # For small lists, show all
                self.logger.info(f"Non-SEC filers (skipped): {sorted(symbols_without_cik)}")
            else:
                # For large lists, show first 30
                self.logger.info(
                    f"Non-SEC filers (skipped, showing first 30/{len(symbols_without_cik)}): "
                    f"{sorted(symbols_without_cik)[:30]}"
                )

        # Update total to reflect only symbols we'll process
        total = len(symbols_with_cik)
        if total == 0:
            self.logger.warning(f"No symbols with CIKs found for {year}, skipping fundamental upload")
            return

        # Statistics
        completed = 0
        success = 0
        failed = 0
        canceled = 0
        skipped = 0

        # Track skipped symbols with details for logging
        skipped_symbols = []  # List of (symbol, cik, error) tuples

        self.logger.info(f"Step 3/3: Fetching fundamental data from SEC EDGAR API for {total} symbols...")
        fetch_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with pre-fetched CIKs (only for symbols with CIKs)
            future_to_symbol = {
                executor.submit(
                    self._process_symbol_fundamental,
                    sym,
                    year,
                    overwrite,
                    cik_map.get(sym)  # Pass pre-fetched CIK (guaranteed non-NULL)
                ): sym
                for sym in symbols_with_cik  # Only process symbols with CIKs
            }

            # Process completed tasks
            for future in as_completed(future_to_symbol):
                result = future.result()
                completed += 1

                if result['status'] == 'success':
                    success += 1
                elif result['status'] == 'canceled':
                    canceled += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                    # Track skipped symbol with details
                    skipped_symbols.append({
                        'symbol': result.get('symbol'),
                        'cik': result.get('cik'),
                        'error': result.get('error', 'Unknown reason')
                    })
                else:
                    failed += 1

                # Progress logging every 50 symbols
                if completed % 50 == 0:
                    elapsed = time.time() - fetch_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    self.logger.info(
                        f"Progress: {completed}/{total} ({success} success, {failed} failed, "
                        f"{canceled} canceled, {skipped} skipped) | Rate: {rate:.1f} sym/sec | ETA: {eta:.0f}s"
                    )

        # Final statistics
        total_time = time.time() - start_time
        fetch_time = time.time() - fetch_start
        avg_rate = completed / fetch_time if fetch_time > 0 else 0

        self.logger.info(
            f"Fundamental upload for {year} completed in {total_time:.1f}s: "
            f"{success} success, {failed} failed, {canceled} canceled, {skipped} skipped out of {total} total"
        )
        self.logger.info(
            f"Performance: CIK fetch={prefetch_time:.1f}s, Data fetch={fetch_time:.1f}s, "
            f"Avg rate={avg_rate:.2f} sym/sec"
        )

        # Log detailed information about skipped symbols
        if skipped > 0:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"SKIPPED COMPANIES DETAILS ({skipped} total)")
            self.logger.info(f"{'='*80}")

            # Get company names from SecurityMaster for skipped symbols
            skipped_symbol_list = [s['symbol'] for s in skipped_symbols]

            try:
                master_tb = self.crsp_ticks.security_master.master_tb

                # Fetch company names for skipped symbols
                skipped_details = master_tb.filter(
                    pl.col('symbol').is_in(skipped_symbol_list)
                ).select(['symbol', 'company', 'cik']).unique()

                # Merge with skip reasons
                for skip_info in skipped_symbols:
                    sym = skip_info['symbol']
                    cik = skip_info['cik']
                    reason = skip_info['error']

                    # Try to find company name
                    company_match = skipped_details.filter(pl.col('symbol') == sym)
                    if not company_match.is_empty():
                        company = company_match['company'].head(1).item()
                    else:
                        company = "Unknown"

                    cik_str = f"CIK {cik}" if cik else "No CIK"
                    self.logger.info(f"  {sym:10} - {company:50} ({cik_str})")
                    self.logger.info(f"             Reason: {reason}")

            except Exception as e:
                self.logger.error(f"Error fetching company details for skipped symbols: {e}", exc_info=True)
                # Fallback to simple list
                self.logger.info(f"Skipped symbols: {sorted(skipped_symbol_list)}")

            self.logger.info(f"{'='*80}\n")

    def run(
            self,
            start_year: int,
            end_year: int,
            max_workers: int=50,
            overwrite: bool=False,
            chunk_size: int=30,
            sleep_time: float=0.02,
            run_fundamental: bool=False,
            run_daily_ticks: bool=True,
            run_minute_ticks: bool=False
        ) -> None:
        """
        Run the complete workflow, fetch and upload fundamental, daily ticks and minute ticks data within the period

        Storage strategy:
        - Fundamental: Once per year -> data/raw/fundamental/{symbol}/{YYYY}/fundamental.parquet
        - Daily ticks: Once per year -> data/raw/ticks/daily/{symbol}/{YYYY}/ticks.parquet
        - Minute ticks: Monthly -> data/raw/ticks/minute/{symbol}/{YYYY}/{MM}/{DD}/ticks.parquet

        :param start_year: Starting year (inclusive)
        :param end_year: Ending year (inclusive)
        :param max_workers: Number of concurrent workers
        :param overwrite: If True, overwrite existing data
        :param chunk_size: Number of symbols to fetch at once for minute data
        :param sleep_time: Sleep time between API requests
        :param run_fundamental: If True, upload fundamental data
        :param run_daily_ticks: If True, upload daily ticks data
        :param run_minute_ticks: If True, upload minute ticks data (all months for each year)
        """
        for year in range(start_year, end_year + 1):
            self.logger.info(f"Processing year {year}")

            if run_fundamental:
                self.logger.info(f"Uploading fundamental data for {year} (year-based, all quarters)")
                self.upload_fundamental(year, max_workers, overwrite)

            if run_daily_ticks:
                self.logger.info(f"Uploading daily ticks for {year} (year-based, all trading days)")
                self.upload_daily_ticks(year, overwrite)

            if run_minute_ticks:
                # Upload minute ticks for all months in the year (only for years >= 2017)
                if year >= 2017:
                    self.logger.info(f"Uploading minute ticks for {year} (all 12 months)")
                    for month in range(1, 13):
                        self.logger.info(f"Processing minute ticks for {year}-{month:02d}")
                        self.upload_minute_ticks(year, month, overwrite, max_workers, chunk_size, sleep_time)
                else:
                    self.logger.info(f"Skipping minute ticks for {year} (data only available from 2017+)")

    def close(self):
        """Close WRDS database connections"""
        if hasattr(self, 'crsp_ticks') and self.crsp_ticks is not None:
            if hasattr(self.crsp_ticks, 'conn') and self.crsp_ticks.conn is not None:
                self.crsp_ticks.conn.close()
                self.logger.info("WRDS connection closed")


if __name__ == "__main__":
    app = UploadApp()
    try:
        # Example: Run from 2010 to 2025 (yearly processing)
        app.run(start_year=2010, end_year=2024, overwrite=False)
    finally:
        app.close()
