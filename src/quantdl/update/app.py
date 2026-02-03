"""
Daily Update App for US Equity Data Lake
=========================================

Automates daily updates for:
1. Daily ticks (OHLCV) - updates year parquet with yesterday's data
2. Minute ticks - adds new daily parquet files
3. Fundamental data - checks EDGAR for new filings and updates if available
"""

import os
import logging
import datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
import requests
import polars as pl
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from quantdl.storage.exceptions import NoSuchKeyError
import time
from threading import Semaphore

from quantdl.utils.logger import setup_logger
from quantdl.utils.calendar import TradingCalendar
from quantdl.storage.config_loader import UploadConfig
from quantdl.storage.s3_client import S3Client
from quantdl.storage.rate_limiter import RateLimiter
from quantdl.storage.cik_resolver import CIKResolver
from quantdl.storage.data_collectors import DataCollectors
from quantdl.storage.data_publishers import DataPublishers
from quantdl.collection.alpaca_ticks import Ticks
from quantdl.collection.crsp_ticks import CRSPDailyTicks
from quantdl.collection.fundamental import SECClient
from quantdl.universe.manager import UniverseManager
from quantdl.master.security_master import SecurityMaster

load_dotenv()


class DailyUpdateApp:
    """
    Orchestrates daily updates for all data types.

    Daily workflow:
    1. Check if market was open yesterday
    2. If yes: Update ticks data (daily + minute)
    3. Check EDGAR for new filings
    4. Update fundamental data for stocks with new filings
    """

    def __init__(self, config_path: str = "configs/storage.yaml"):
        """Initialize the daily update app."""
        # Setup config and clients
        self.config = UploadConfig(config_path)
        self.s3_client = S3Client(config_path).client

        # Setup logger
        self.logger = setup_logger(
            name="daily_update",
            log_dir=Path("data/logs/update"),
            level=logging.DEBUG,
            console_output=True
        )

        # Initialize calendar
        self.calendar = TradingCalendar()

        # Initialize data fetchers
        self.alpaca_ticks = Ticks()

        # Try WRDS, but allow S3-only mode for 2025+ operations
        try:
            self.crsp_ticks = CRSPDailyTicks(
                s3_client=self.s3_client,
                bucket_name='us-equity-datalake',
                require_wrds=False  # Don't crash if WRDS unavailable
            )
            self.security_master = self.crsp_ticks.security_master
            self._wrds_available = (self.crsp_ticks._conn is not None)
        except Exception as e:
            self.logger.warning(f"CRSPDailyTicks initialization failed: {e}, using S3-only mode")
            self.crsp_ticks = None
            # Initialize SecurityMaster directly from S3
            self.security_master = SecurityMaster(
                s3_client=self.s3_client,
                bucket_name='us-equity-datalake'
            )
            self._wrds_available = False

        # Initialize universe manager
        self.universe_manager = UniverseManager(
            crsp_fetcher=self.crsp_ticks if self._wrds_available else None,
            security_master=self.security_master
        )
        self._symbols_cache: Dict[int, List[str]] = {}

        # Load Alpaca credentials
        ALPACA_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET
        }

        # Rate limiters
        self.sec_rate_limiter = RateLimiter(max_rate=9.5)  # SEC EDGAR: 10 req/sec limit

        # Initialize CIK resolver
        self.cik_resolver = CIKResolver(
            security_master=self.security_master,
            logger=self.logger
        )

        # Initialize data collectors and publishers
        self.data_collectors = DataCollectors(
            crsp_ticks=self.crsp_ticks,
            alpaca_ticks=self.alpaca_ticks,
            alpaca_headers=self.headers,
            logger=self.logger,
            sec_rate_limiter=self.sec_rate_limiter
        )

        self.data_publishers = DataPublishers(
            s3_client=self.s3_client,
            upload_config=self.config,
            logger=self.logger,
            data_collectors=self.data_collectors,
            security_master=self.security_master
        )

        # SEC client for checking recent filings
        self.sec_client = SECClient(rate_limiter=self.sec_rate_limiter)

    def check_market_open(self, date: dt.date) -> bool:
        """
        Check if market was open on a given date.

        :param date: Date to check
        :return: True if market was open, False otherwise
        """
        is_open = self.calendar.is_trading_day(date)
        self.logger.info(f"Market {'was' if is_open else 'was NOT'} open on {date}")
        return is_open

    def _get_symbols_for_year(self, year: int) -> List[str]:
        if year in self._symbols_cache:
            return self._symbols_cache[year]
        symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')
        self._symbols_cache[year] = symbols
        return symbols

    def get_recent_edgar_filings(self, cik: str, lookback_days: int = 7) -> List[Dict]:
        """
        Check EDGAR for recent filings for a given CIK.

        Uses SEC EDGAR submissions endpoint to get recent filings.

        :param cik: Company CIK number
        :param lookback_days: Number of days to look back for filings
        :return: List of filing dictionaries
        """
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

        try:
            self.sec_rate_limiter.acquire()
            response = requests.get(url, headers=self.sec_client.header)
            response.raise_for_status()
            data = response.json()

            # Get recent filings
            recent_filings = data.get('filings', {}).get('recent', {})

            if not recent_filings:
                return []

            # Extract filing dates and forms
            filing_dates = recent_filings.get('filingDate', [])
            forms = recent_filings.get('form', [])
            accession_numbers = recent_filings.get('accessionNumber', [])

            # Filter for relevant forms (10-K, 10-Q, 8-K amendments)
            relevant_forms = {'10-K', '10-Q', '10-K/A', '10-Q/A', '8-K'}

            # Calculate cutoff date
            cutoff_date = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()

            recent = []
            for i, filing_date in enumerate(filing_dates):
                if filing_date >= cutoff_date and forms[i] in relevant_forms:
                    recent.append({
                        'filingDate': filing_date,
                        'form': forms[i],
                        'accessionNumber': accession_numbers[i]
                    })

            return recent

        except requests.RequestException as e:
            self.logger.debug(f"Failed to fetch EDGAR submissions for CIK {cik}: {e}")
            return []
        except Exception as e:
            self.logger.debug(f"Unexpected error checking EDGAR for CIK {cik}: {e}")
            return []

    def _check_filing(self, symbol: str, cik: str, lookback_days: int, semaphore: Semaphore) -> Dict:
        """Check if a symbol has recent filings (helper for parallel execution)."""
        with semaphore:
            time.sleep(0.1)
            recent_filings = self.get_recent_edgar_filings(cik, lookback_days)

            if recent_filings:
                self.logger.debug(
                    f"{symbol} (CIK {cik}): {len(recent_filings)} recent filings - "
                    f"{[f['form'] for f in recent_filings]}"
                )

            return {
                'symbol': symbol,
                'cik': cik,
                'has_recent_filing': len(recent_filings) > 0,
                'filing_types': [f['form'] for f in recent_filings]
            }

    def get_symbols_with_recent_filings(
        self,
        update_date: dt.date,
        symbols: List[str],
        lookback_days: int = 7
    ) -> Tuple[Set[str], Set[str], Dict[str, int]]:
        """Identify symbols that have new EDGAR filings (parallelized).

        Returns:
            Tuple of (symbols_with_fundamental_filings, symbols_with_any_filings, filing_type_counts)
            - symbols_with_fundamental_filings: symbols with 10-K/10-Q (contain financials)
            - symbols_with_any_filings: symbols with any filing type including 8-K
        """
        symbols_with_filings = set()
        symbols_with_fundamental_filings = set()
        filing_stats: Dict[str, int] = {}

        # Filing types that contain fundamental financial data
        fundamental_forms = {'10-K', '10-Q', '10-K/A', '10-Q/A'}

        self.logger.info(f"Checking EDGAR for recent filings ({lookback_days} day lookback)...")

        # Batch resolve symbols to CIKs
        year = update_date.year
        symbol_to_cik = self.cik_resolver.batch_prefetch_ciks(symbols, year)

        # Filter out None CIKs
        symbol_to_cik = {sym: cik for sym, cik in symbol_to_cik.items() if cik is not None}

        self.logger.info(f"Resolved {len(symbol_to_cik)}/{len(symbols)} symbols to CIKs")

        # Parallel EDGAR checks with rate limiting
        max_workers = 10
        semaphore = Semaphore(max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._check_filing, sym, cik, lookback_days, semaphore): sym
                for sym, cik in symbol_to_cik.items()
            }

            checked = 0
            for future in as_completed(futures):
                result = future.result()
                if result['has_recent_filing']:
                    symbols_with_filings.add(result['symbol'])
                    # Check if any filing is fundamental-relevant (10-K/10-Q)
                    if any(form in fundamental_forms for form in result['filing_types']):
                        symbols_with_fundamental_filings.add(result['symbol'])
                    # Count filing types
                    for form in result['filing_types']:
                        filing_stats[form] = filing_stats.get(form, 0) + 1

                checked += 1
                if checked % 100 == 0:
                    self.logger.info(f"Checked {checked}/{len(symbol_to_cik)} symbols for filings")

        # Log filing breakdown
        stats_str = ", ".join(f"{k}: {v}" for k, v in sorted(filing_stats.items()))
        self.logger.info(
            f"Found {len(symbols_with_filings)} symbols with recent filings "
            f"out of {len(symbol_to_cik)} checked ({stats_str})"
        )
        self.logger.info(
            f"Of these, {len(symbols_with_fundamental_filings)} have 10-K/10-Q filings with financial data"
        )

        return symbols_with_fundamental_filings, symbols_with_filings, filing_stats

    def update_daily_ticks(
        self,
        update_date: dt.date,
        symbols: Optional[List[str]] = None,
        max_workers: int = 50
    ) -> Dict[str, int]:
        """
        Update daily ticks for a specific date using total refetch approach.

        Fetches entire month-to-date from Alpaca and overwrites S3 file.
        No merge logic - simpler and avoids S3 GET costs.

        Storage: data/raw/ticks/daily/{security_id}/{YYYY}/{MM}/ticks.parquet

        :param update_date: Date to update (typically yesterday)
        :param symbols: List of symbols to update (if None, uses current universe)
        :param max_workers: Number of concurrent workers
        :return: Dict with statistics
        """
        year = update_date.year
        month = update_date.month

        # Load symbols if not provided
        if symbols is None:
            symbols = self._get_symbols_for_year(year)

        self.logger.info(
            f"Fetching daily ticks for {len(symbols)} symbols "
            f"(month-to-date: {year}-{month:02d}-01 to {update_date})"
        )

        # Fetch entire month-to-date from Alpaca (total refetch approach)
        month_start = dt.date(year, month, 1)
        start_str = dt.datetime.combine(
            month_start, dt.time(0, 0), tzinfo=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")
        end_str = dt.datetime.combine(
            update_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")

        symbol_bars = self.alpaca_ticks.fetch_daily_range_bulk(
            symbols=symbols,
            start_str=start_str,
            end_str=end_str,
            sleep_time=0.2
        )

        self.logger.info(
            f"Uploading daily ticks for {len(symbols)} symbols "
            f"(monthly partition: {year}/{month:02d})"
        )

        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        import io
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_symbol(sym: str) -> Dict[str, Optional[str]]:
            """Process single symbol update - total refetch, no merge"""
            try:
                # Resolve symbol to security_id
                security_id = self.security_master.get_security_id(
                    sym, update_date.isoformat()
                )

                ticks = symbol_bars.get(sym, [])

                if not ticks:
                    self.logger.debug(f"Skipping {sym} daily ticks: no data from Alpaca")
                    return {'symbol': sym, 'status': 'skipped', 'error': 'No data from Alpaca'}

                # Parse ticks to DataFrame
                parsed_ticks = self.alpaca_ticks.parse_ticks(ticks)
                from dataclasses import asdict
                ticks_data = [asdict(dp) for dp in parsed_ticks]

                month_df = pl.DataFrame(ticks_data).with_columns([
                    pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S').dt.date().cast(pl.Utf8),
                    pl.col('open').cast(pl.Float64).round(4),
                    pl.col('high').cast(pl.Float64).round(4),
                    pl.col('low').cast(pl.Float64).round(4),
                    pl.col('close').cast(pl.Float64).round(4),
                    pl.col('volume').cast(pl.Int64)
                ]).select(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                if len(month_df) == 0:
                    self.logger.debug(f"Skipping {sym} daily ticks: empty DataFrame after parsing")
                    return {'symbol': sym, 'status': 'skipped', 'error': 'Empty DataFrame after parsing'}

                # Drop rows that only have timestamp (all other fields null)
                null_row = (
                    pl.col('open').is_null()
                    & pl.col('high').is_null()
                    & pl.col('low').is_null()
                    & pl.col('close').is_null()
                    & pl.col('volume').is_null()
                )
                month_df = month_df.filter(~null_row).sort('timestamp')

                # Upload to S3 (overwrites existing file)
                s3_key = f"data/raw/ticks/daily/{security_id}/{year}/{month:02d}/ticks.parquet"

                buffer = io.BytesIO()
                month_df.write_parquet(buffer)
                buffer.seek(0)

                metadata = {
                    'security_id': str(security_id),
                    'symbols': [sym],
                    'year': str(year),
                    'month': f"{month:02d}",
                    'data_type': 'daily_ticks',
                    'source': 'alpaca',
                    'trading_days': str(len(month_df)),
                    'partition_type': 'monthly'
                }
                metadata_prepared = {
                    k: str(v) if not isinstance(v, list) else str(v) for k, v in metadata.items()
                }

                self.data_publishers.upload_fileobj(buffer, s3_key, metadata_prepared)

                return {'symbol': sym, 'status': 'success', 'error': None}

            except Exception as e:
                self.logger.error(f"Error updating daily ticks for {sym}: {e}")
                return {'symbol': sym, 'status': 'failed', 'error': str(e)}

        # Process symbols concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_symbol, sym): sym for sym in symbols}

            pbar = tqdm(as_completed(futures), total=len(symbols), desc="Daily ticks", unit="sym")
            for future in pbar:
                result = future.result()

                if result['status'] == 'success':
                    stats['success'] += 1
                elif result['status'] == 'skipped':
                    stats['skipped'] += 1
                else:
                    stats['failed'] += 1

                pbar.set_postfix(ok=stats['success'], fail=stats['failed'], skip=stats['skipped'])

        self.logger.info(
            f"Daily ticks: {stats['success']} ok, {stats['failed']} fail, {stats['skipped']} skip"
        )

        return stats

    def consolidate_year(
        self,
        year: int,
        symbols: Optional[List[str]] = None,
        max_workers: int = 50,
        force: bool = False
    ) -> Dict[str, int]:
        """
        Consolidate previous year's monthly files into history.parquet.

        Run on Jan 1 to consolidate previous year data:
        1. Read 12 monthly files for the year
        2. Read existing history.parquet (if exists)
        3. Remove target year data from history (if any)
        4. Append new year data
        5. Write to history.parquet (OVERWRITES file)
        6. Delete monthly files

        :param year: Year to consolidate (e.g., 2025 on Jan 1, 2026)
        :param symbols: List of symbols (if None, uses year's universe)
        :param max_workers: Number of concurrent workers
        :param force: If True, allow overwriting existing year data in history
        :return: Dict with statistics
        """
        self.logger.info(f"Starting year consolidation for {year}")

        # Load symbols if not provided
        if symbols is None:
            symbols = self._get_symbols_for_year(year)

        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import io

        def consolidate_symbol(sym: str) -> Dict[str, Optional[str]]:
            """Consolidate monthly files for a single symbol"""
            try:
                # Resolve symbol to security_id
                security_id = self.crsp_ticks.security_master.get_security_id(
                    sym, f"{year}-12-31"
                )

                # Read all monthly files for the year
                monthly_dfs = []
                for month in range(1, 13):
                    s3_key = f"data/raw/ticks/daily/{security_id}/{year}/{month:02d}/ticks.parquet"
                    try:
                        response = self.s3_client.get_object(
                            Bucket=self.data_publishers.bucket_name,
                            Key=s3_key
                        )
                        monthly_dfs.append(pl.read_parquet(response['Body']))
                    except (ClientError, NoSuchKeyError) as e:
                        if e.response['Error']['Code'] == 'NoSuchKey':
                            # Month file doesn't exist, skip
                            continue
                        else:
                            self.logger.warning(f"Could not read {s3_key}: {e}")
                            continue
                    except Exception as e:
                        self.logger.warning(f"Could not read {s3_key}: {e}")
                        continue

                if not monthly_dfs:
                    return {'symbol': sym, 'status': 'skipped', 'error': 'No monthly files found'}

                # Combine monthly data
                year_df = pl.concat(monthly_dfs).sort('timestamp')

                # Read existing history file if it exists
                history_key = f"data/raw/ticks/daily/{security_id}/history.parquet"
                try:
                    response = self.s3_client.get_object(
                        Bucket=self.data_publishers.bucket_name,
                        Key=history_key
                    )
                    history_df = pl.read_parquet(response['Body'])

                    # Check if year already exists in history (safeguard)
                    if not force:
                        existing_years = history_df['timestamp'].str.slice(0, 4).unique().to_list()
                        if str(year) in existing_years:
                            return {
                                'symbol': sym,
                                'status': 'failed',
                                'error': f'Year {year} already exists in history.parquet. Use --force to overwrite.'
                            }

                    # Remove existing year data from history (if any)
                    history_df = history_df.filter(
                        ~pl.col('timestamp').str.starts_with(str(year))
                    )
                    # Append new year data
                    combined_df = pl.concat([history_df, year_df]).sort('timestamp')
                except (ClientError, NoSuchKeyError) as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        # No history file yet, use year data only
                        combined_df = year_df
                    else:
                        raise

                # Write consolidated history file
                buffer = io.BytesIO()
                combined_df.write_parquet(buffer)
                buffer.seek(0)

                metadata = {
                    'security_id': str(security_id),
                    'symbols': [sym],
                    'data_type': 'daily_ticks',
                    'source': 'mixed',  # Could be CRSP + Alpaca
                    'trading_days': str(len(combined_df)),
                    'partition_type': 'history',
                    'years_included': f"All up to {year}"
                }
                metadata_prepared = {
                    k: str(v) for k, v in metadata.items()
                }

                self.data_publishers.upload_fileobj(buffer, history_key, metadata_prepared)

                # Delete monthly files after successful upload
                for month in range(1, 13):
                    s3_key = f"data/raw/ticks/daily/{security_id}/{year}/{month:02d}/ticks.parquet"
                    try:
                        self.s3_client.delete_object(
                            Bucket=self.data_publishers.bucket_name,
                            Key=s3_key
                        )
                    except Exception as e:
                        self.logger.debug(f"Could not delete {s3_key}: {e}")

                return {'symbol': sym, 'status': 'success', 'error': None}

            except Exception as e:
                self.logger.error(f"Error consolidating {sym}: {e}")
                return {'symbol': sym, 'status': 'failed', 'error': str(e)}

        # Process symbols concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(consolidate_symbol, sym): sym for sym in symbols}

            pbar = tqdm(as_completed(futures), total=len(symbols), desc="Consolidate", unit="sym")
            for future in pbar:
                result = future.result()

                if result['status'] == 'success':
                    stats['success'] += 1
                elif result['status'] == 'skipped':
                    stats['skipped'] += 1
                else:
                    stats['failed'] += 1

                pbar.set_postfix(ok=stats['success'], fail=stats['failed'], skip=stats['skipped'])

        self.logger.info(
            f"Year consolidation: {stats['success']} ok, {stats['failed']} fail, {stats['skipped']} skip"
        )

        return stats

    def update_minute_ticks(
        self,
        update_date: dt.date,
        symbols: Optional[List[str]] = None,
        max_workers: int = 50
    ) -> Dict[str, int]:
        """
        Update minute ticks for a specific trading day.

        Fetches minute data from Alpaca and adds new daily parquet files to S3.
        Storage: data/raw/ticks/minute/{security_id}/{YYYY}/{MM}/{DD}/ticks.parquet

        :param update_date: Trading day to update (typically yesterday)
        :param symbols: List of symbols to update (if None, uses current universe)
        :param max_workers: Number of concurrent workers for uploading
        :return: Dict with statistics
        """
        year = update_date.year
        trade_day_str = update_date.strftime('%Y-%m-%d')

        # Load symbols if not provided
        if symbols is None:
            symbols = self._get_symbols_for_year(year)

        self.logger.info(
            f"Updating minute ticks for {len(symbols)} symbols for {trade_day_str}"
        )

        # Fetch minute data for all symbols for the trading day
        symbol_bars = self.data_collectors.fetch_minute_day(
            symbols=symbols,
            trade_day=trade_day_str,
            sleep_time=0.2
        )

        # Parse bars into daily DataFrames
        parsed_data = self.data_collectors.parse_minute_bars_to_daily(
            symbol_bars=symbol_bars,
            trading_days=[trade_day_str]
        )

        # Upload data
        stats = {'success': 0, 'failed': 0, 'skipped': 0}
        import io

        def upload_minute_tick(item):
            """Upload single symbol-day minute tick data"""
            (sym, day), minute_df = item
            try:
                if len(minute_df) == 0:
                    return {'symbol': sym, 'status': 'skipped', 'error': 'Empty DataFrame'}

                # Resolve symbol to security_id at trade day
                security_id = self.security_master.get_security_id(sym, day)

                # Upload to S3
                date_obj = dt.datetime.strptime(day, '%Y-%m-%d').date()
                year_str = date_obj.strftime('%Y')
                month_str = date_obj.strftime('%m')
                day_str = date_obj.strftime('%d')

                buffer = io.BytesIO()
                minute_df.write_parquet(buffer)
                buffer.seek(0)

                s3_key = f"data/raw/ticks/minute/{security_id}/{year_str}/{month_str}/{day_str}/ticks.parquet"
                self.data_publishers.upload_fileobj(buffer, s3_key, {
                    'security_id': str(security_id),
                    'symbol': sym,
                    'trade_day': day,
                    'data_type': 'ticks'
                })

                return {'symbol': sym, 'status': 'success', 'error': None}

            except Exception as e:
                self.logger.error(f"Error uploading minute ticks for {sym} on {day}: {e}")
                return {'symbol': sym, 'status': 'failed', 'error': str(e)}

        # Process uploads concurrently
        items = list(parsed_data.items())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(upload_minute_tick, item): item for item in items}

            pbar = tqdm(as_completed(futures), total=len(items), desc="Minute ticks", unit="sym-day")
            for future in pbar:
                result = future.result()

                if result['status'] == 'success':
                    stats['success'] += 1
                elif result['status'] == 'skipped':
                    stats['skipped'] += 1
                else:
                    stats['failed'] += 1

                pbar.set_postfix(ok=stats['success'], fail=stats['failed'], skip=stats['skipped'])

        self.logger.info(
            f"Minute ticks: {stats['success']} ok, {stats['failed']} fail, {stats['skipped']} skip"
        )

        return stats

    def update_fundamental(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        update_raw: bool = True,
        update_ttm: bool = True,
        update_derived: bool = True,
        max_workers: int = 10
    ) -> Dict[str, int]:
        """
        Update fundamental data for symbols with recent EDGAR filings.

        Updates three types of fundamental data (selectively):
        1. Raw fundamental data (data/raw/fundamental/{cik}/fundamental.parquet)
        2. TTM features (data/derived/features/fundamental/{cik}/ttm.parquet)
        3. Derived metrics (data/derived/features/fundamental/{cik}/metrics.parquet)

        :param symbols: List of symbols to update
        :param start_date: Start date for data collection (YYYY-MM-DD)
        :param end_date: End date for data collection (YYYY-MM-DD)
        :param update_raw: Whether to update raw fundamental data
        :param update_ttm: Whether to update TTM fundamental data
        :param update_derived: Whether to update derived metrics
        :param max_workers: Number of concurrent workers
        :return: Dict with statistics
        """
        self.logger.info(
            f"Updating fundamental data for {len(symbols)} symbols "
            f"from {start_date} to {end_date}"
        )

        # Resolve symbols to CIKs
        symbol_to_cik = {}
        for sym in symbols:
            cik = self.cik_resolver.get_cik(sym, end_date)
            if cik:
                symbol_to_cik[sym] = cik

        self.logger.info(f"Resolved {len(symbol_to_cik)}/{len(symbols)} symbols to CIKs")

        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        # Process each symbol
        for sym, cik in symbol_to_cik.items():
            try:
                # 1. Update raw fundamental data
                if update_raw:
                    result = self.data_publishers.publish_fundamental(
                        sym=sym,
                        start_date=start_date,
                        end_date=end_date,
                        cik=cik,
                        sec_rate_limiter=self.sec_rate_limiter
                    )

                    if result['status'] == 'skipped':
                        stats['skipped'] += 1
                        continue
                    elif result['status'] == 'failed':
                        stats['failed'] += 1
                        continue

                # 2. Update TTM features
                if update_ttm:
                    ttm_result = self.data_publishers.publish_ttm_fundamental(
                        sym=sym,
                        start_date=start_date,
                        end_date=end_date,
                        cik=cik,
                        sec_rate_limiter=self.sec_rate_limiter
                    )

                # 3. Update derived metrics
                if update_derived:
                    derived_df, error = self.data_collectors.collect_derived_long(
                        cik=cik,
                        start_date=start_date,
                        end_date=end_date,
                        symbol=sym
                    )

                    if error is None and len(derived_df) > 0:
                        self.data_publishers.publish_derived_fundamental(
                            sym=sym,
                            start_date=start_date,
                            end_date=end_date,
                            derived_df=derived_df,
                            cik=cik
                        )

                stats['success'] += 1
                self.logger.info(f"Successfully updated fundamental data for {sym}")

            except Exception as e:
                self.logger.error(f"Error updating fundamental data for {sym}: {e}")
                stats['failed'] += 1

        self.logger.info(
            f"Fundamental update completed: {stats['success']} success, "
            f"{stats['failed']} failed, {stats['skipped']} skipped"
        )

        return stats

    def update_sentiment(
        self,
        symbols: List[str],
        end_date: str,
    ) -> Dict[str, int]:
        """
        Update sentiment data for symbols with new 10-K/10-Q filings.

        Uses read-check-append pattern:
        1. Fetch MD&A text from new filings
        2. Compute sentiment using FinBERT
        3. Read existing sentiment.parquet (if exists)
        4. Check if filing already has sentiment computed
        5. Append new sentiment data and write back

        Storage: data/derived/features/sentiment/{cik}/sentiment.parquet

        :param symbols: List of symbols to update
        :param end_date: End date (YYYY-MM-DD)
        :return: Dict with statistics
        """
        from quantdl.collection.sentiment import SentimentCollector
        from quantdl.derived.sentiment import compute_sentiment_long
        from quantdl.models.finbert import FinBERTModel
        from quantdl.storage.rate_limiter import RateLimiter
        import io

        self.logger.info(f"Updating sentiment for {len(symbols)} symbols")

        # Load FinBERT model (GPU-accelerated)
        self.logger.info("Loading FinBERT model...")
        model = FinBERTModel()
        model.load()
        self.logger.info(f"FinBERT loaded on {model._device}")

        # Initialize collector with rate limiter (10 req/sec)
        rate_limiter = RateLimiter(max_rate=10.0)
        collector = SentimentCollector(
            rate_limiter=rate_limiter,
            logger=self.logger
        )

        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        for sym in tqdm(symbols, desc="Sentiment update", unit="sym"):
            try:
                # Resolve symbol to CIK
                cik = self.cik_resolver.get_cik(sym, end_date)
                if not cik:
                    self.logger.debug(f"No CIK found for {sym}")
                    stats['skipped'] += 1
                    continue

                # Get filing metadata for this CIK
                filings = collector.get_filings_metadata(cik)
                if not filings:
                    self.logger.debug(f"No filings found for {sym} (CIK {cik})")
                    stats['skipped'] += 1
                    continue

                # Read existing sentiment data (if exists)
                s3_key = f"data/derived/features/sentiment/{cik}/sentiment.parquet"
                existing_accessions = set()
                existing_df = None

                try:
                    response = self.s3_client.get_object(
                        Bucket='us-equity-datalake',
                        Key=s3_key
                    )
                    existing_df = pl.read_parquet(response['Body'])
                    # Get unique accession numbers already processed
                    if 'accession_number' in existing_df.columns:
                        existing_accessions = set(
                            existing_df.select('accession_number').unique().to_series().to_list()
                        )
                except (ClientError, NoSuchKeyError):
                    pass  # No existing file, will create new

                # Filter to new filings only (not already processed)
                new_filings = [
                    f for f in filings
                    if f.get('accession') not in existing_accessions
                ]

                if not new_filings:
                    self.logger.debug(f"{sym}: all filings already processed")
                    stats['skipped'] += 1
                    continue

                self.logger.debug(f"{sym}: {len(new_filings)} new filings to process")

                # Extract MD&A text from new filings
                filing_texts = collector.collect_filing_texts(
                    cik=cik,
                    filings=new_filings
                )

                if not filing_texts:
                    self.logger.debug(f"{sym}: no MD&A text extracted")
                    stats['skipped'] += 1
                    continue

                # Compute sentiment for each filing
                new_sentiment_df = compute_sentiment_long(filing_texts, model)

                if len(new_sentiment_df) == 0:
                    self.logger.debug(f"{sym}: no sentiment computed")
                    stats['skipped'] += 1
                    continue

                # Add CIK column and accession_number for deduplication
                new_sentiment_df = new_sentiment_df.with_columns([
                    pl.lit(cik).alias('cik')
                ])

                # Merge with existing data
                if existing_df is not None and len(existing_df) > 0:
                    combined_df = pl.concat([existing_df, new_sentiment_df])
                else:
                    combined_df = new_sentiment_df

                # Sort by filing_date descending
                combined_df = combined_df.sort('as_of_date', descending=True)

                # Upload to S3
                buffer = io.BytesIO()
                combined_df.write_parquet(buffer)
                buffer.seek(0)

                self.s3_client.put_object(
                    Bucket='us-equity-datalake',
                    Key=s3_key,
                    Body=buffer.getvalue(),
                    Metadata={
                        'cik': str(cik),
                        'symbol': sym,
                        'model_name': model.name,
                        'model_version': model.version,
                        'filings_count': str(len(combined_df.select('accession_number').unique())),
                    }
                )

                stats['success'] += 1
                self.logger.debug(
                    f"{sym}: updated sentiment ({len(new_sentiment_df)} new metrics)"
                )

            except Exception as e:
                self.logger.error(f"Error updating sentiment for {sym}: {e}")
                stats['failed'] += 1

        self.logger.info(
            f"Sentiment update: {stats['success']} ok, "
            f"{stats['failed']} fail, {stats['skipped']} skip"
        )

        return stats

    def update_top3000(self, target_date: dt.date) -> Dict[str, int]:
        """
        Refresh top 3000 universe for current month if missing.

        Uses UniverseManager.get_top_3000() for liquidity ranking.
        Storage (S3): data/symbols/{YYYY}/{MM}/top3000.txt
        Storage (local): {LOCAL_STORAGE_PATH}/symbols/{YYYY}/{MM}/top3000.txt
        """
        year = target_date.year
        month = target_date.month

        storage_backend = os.getenv('STORAGE_BACKEND', 's3').lower()
        local_path = os.getenv('LOCAL_STORAGE_PATH', '')

        # Check if already exists
        if storage_backend == 'local':
            local_file = Path(local_path) / 'data' / 'symbols' / str(year) / f'{month:02d}' / 'top3000.txt'
            if local_file.exists():
                self.logger.info(f"Top 3000 for {year}-{month:02d} already exists locally, skipping")
                return {'status': 'skipped'}
        else:
            from quantdl.storage.validation import Validator
            validator = Validator(self.s3_client, self.logger)
            if validator.top_3000_exists(year, month):
                self.logger.info(f"Top 3000 for {year}-{month:02d} already exists, skipping")
                return {'status': 'skipped'}

        self.logger.info(f"Refreshing top 3000 for {year}-{month:02d}...")

        # Get all symbols
        symbols = self._get_symbols_for_year(year)

        # Calculate top 3000 by liquidity (uses Alpaca data)
        as_of = target_date.isoformat()
        source = 'alpaca'  # Always Alpaca for 2025+
        top_3000 = self.universe_manager.get_top_3000(as_of, symbols, source)

        if not top_3000:
            self.logger.warning("No symbols returned for top 3000")
            return {'status': 'failed', 'error': 'No symbols'}

        # Store based on backend
        if storage_backend == 'local':
            import json
            local_dir = Path(local_path) / 'data' / 'symbols' / str(year) / f'{month:02d}'
            local_dir.mkdir(parents=True, exist_ok=True)

            # Write symbols list
            txt_file = local_dir / 'top3000.txt'
            txt_file.write_text('\n'.join(top_3000))

            # Write metadata (hidden file)
            metadata = {
                'year': year,
                'month': month,
                'as_of': as_of,
                'source': source,
                'count': len(top_3000),
                'created_at': dt.datetime.now().isoformat()
            }
            json_file = local_dir / '.top3000.txt.metadata.json'
            json_file.write_text(json.dumps(metadata, indent=2))

            self.logger.info(f"Saved top 3000 locally: {txt_file}")
            result = {'status': 'success', 'path': str(txt_file)}
        else:
            result = self.data_publishers.publish_top_3000(
                year=year,
                month=month,
                as_of=as_of,
                symbols=top_3000,
                source=source
            )
            self.logger.info(f"Uploaded top 3000 for {year}-{month:02d} ({len(top_3000)} symbols)")

        return result

    def run_daily_update(
        self,
        target_date: Optional[dt.date] = None,
        update_daily_ticks: bool = True,
        update_minute_ticks: bool = True,
        update_fundamental: bool = True,
        update_ttm: bool = True,
        update_derived: bool = True,
        update_sentiment: bool = True,
        fundamental_lookback_days: int = 7,
        update_top3000: bool = True,
    ):
        """
        Run complete daily update workflow.

        Workflow:
        1. Check if market was open on target date
        2. If yes: Update daily and/or minute ticks
        3. Check EDGAR for recent filings
        4. Update fundamental data for symbols with new filings

        :param target_date: Date to update (default: yesterday)
        :param update_daily_ticks: Whether to update daily ticks data
        :param update_minute_ticks: Whether to update minute ticks data
        :param update_fundamental: Whether to update raw fundamental data
        :param update_ttm: Whether to update TTM fundamental data
        :param update_derived: Whether to update derived metrics
        :param update_sentiment: Whether to update sentiment data (FinBERT on MD&A)
        :param fundamental_lookback_days: Days to look back for EDGAR filings
        """
        # Default to yesterday if no date specified
        if target_date is None:
            target_date = dt.date.today() - dt.timedelta(days=1)

        self.logger.info(f"=" * 80)
        self.logger.info(f"Starting daily update for {target_date}")
        self.logger.info(f"=" * 80)

        # Update SecurityMaster from SEC (extend end_dates for active securities)
        self.logger.info("Updating SecurityMaster from SEC...")
        sm_stats = self.security_master.update_from_sec(
            s3_client=self.s3_client,
            bucket_name='us-equity-datalake'
        )
        self.logger.info(
            f"SecurityMaster: {sm_stats['extended']} extended, "
            f"{sm_stats['added']} added, {sm_stats['unchanged']} unchanged"
        )

        # Update top 3000 universe if needed
        if update_top3000:
            self.logger.info("[EMAIL] Update top3000 started")
            self.update_top3000(target_date)
            self.logger.info("[EMAIL] Update top3000 successful")

        # Get current universe
        year = target_date.year
        symbols = self._get_symbols_for_year(year)

        # 1. Update ticks data (only if market was open)
        if update_daily_ticks or update_minute_ticks:
            market_open = self.check_market_open(target_date)

            if market_open:
                self.logger.info(f"Updating ticks data for {target_date}...")

                # Update daily ticks
                if update_daily_ticks:
                    self.logger.info("[EMAIL] Update daily ticks started")
                    daily_stats = self.update_daily_ticks(target_date, symbols)
                    self.logger.info("[EMAIL] Update daily ticks successful")

                # Update minute ticks
                if update_minute_ticks:
                    self.logger.info("[EMAIL] Update minute ticks started")
                    minute_stats = self.update_minute_ticks(target_date, symbols)
                    self.logger.info("[EMAIL] Update minute ticks successful")
            else:
                self.logger.info(
                    f"Market was closed on {target_date}, skipping ticks update"
                )

        # 2. Update fundamental/sentiment data (check for recent filings)
        symbols_with_fundamental_filings = set()
        if update_fundamental or update_ttm or update_derived or update_sentiment:
            self.logger.info("Checking EDGAR for recent filings...")

            symbols_with_fundamental_filings, symbols_with_filings, filing_stats = self.get_symbols_with_recent_filings(
                symbols=symbols,
                update_date=target_date,
                lookback_days=fundamental_lookback_days
            )

        if (update_fundamental or update_ttm or update_derived) and symbols_with_fundamental_filings:
            self.logger.info(
                f"Updating fundamental data for {len(symbols_with_fundamental_filings)} symbols "
                f"with 10-K/10-Q filings..."
            )

            # Use a date range covering the lookback period
            end_date = dt.date.today().isoformat()
            start_date = (
                dt.date.today() - dt.timedelta(days=fundamental_lookback_days)
            ).isoformat()

            # Log [EMAIL] markers for each enabled type
            if update_fundamental:
                self.logger.info("[EMAIL] Update raw fundamental started")
            if update_ttm:
                self.logger.info("[EMAIL] Update ttm fundamental started")
            if update_derived:
                self.logger.info("[EMAIL] Update derived fundamental started")

            fundamental_stats = self.update_fundamental(
                symbols=list(symbols_with_fundamental_filings),
                start_date=start_date,
                end_date=end_date,
                update_raw=update_fundamental,
                update_ttm=update_ttm,
                update_derived=update_derived
            )

            # Log [EMAIL] success markers
            if update_fundamental:
                self.logger.info("[EMAIL] Update raw fundamental successful")
            if update_ttm:
                self.logger.info("[EMAIL] Update ttm fundamental successful")
            if update_derived:
                self.logger.info("[EMAIL] Update derived fundamental successful")
        elif (update_fundamental or update_ttm or update_derived):
            self.logger.info("No symbols with 10-K/10-Q filings, skipping fundamental update")

        # 3. Update sentiment data (for symbols with new 10-K/10-Q filings)
        if update_sentiment and symbols_with_fundamental_filings:
            self.logger.info("[EMAIL] Update sentiment started")
            try:
                sentiment_stats = self.update_sentiment(
                    symbols=list(symbols_with_fundamental_filings),
                    end_date=target_date.isoformat()
                )
                self.logger.info("[EMAIL] Update sentiment successful")
            except Exception as e:
                self.logger.error(f"Sentiment update failed: {e}")
                self.logger.info("[EMAIL] Update sentiment failed")

        self.logger.info(f"=" * 80)
        self.logger.info(f"Daily update completed for {target_date}")
        self.logger.info(f"=" * 80)