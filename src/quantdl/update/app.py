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
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import requests
import polars as pl
from dotenv import load_dotenv

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
        self.crsp_ticks = CRSPDailyTicks()

        # Initialize universe manager
        self.universe_manager = UniverseManager(
            crsp_fetcher=self.crsp_ticks,
            security_master=self.crsp_ticks.security_master
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
            security_master=self.crsp_ticks.security_master,
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
            data_collectors=self.data_collectors
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

    def get_symbols_with_recent_filings(
        self,
        update_date: dt.date,
        symbols: List[str],
        lookback_days: int = 7
    ) -> Set[str]:
        """
        Identify symbols that have new EDGAR filings.

        :param update_date: Date to update (typically yesterday)
        :param symbols: List of symbols to check
        :param lookback_days: Number of days to look back
        :return: Set of symbols with recent filings
        """
        symbols_with_filings = set()

        self.logger.info(f"Checking EDGAR for recent filings ({lookback_days} day lookback)...")

        # Resolve symbols to CIKs
        symbol_to_cik = {}
        for sym in symbols:
            cik = self.cik_resolver.get_cik(sym, update_date.strftime('%Y-%m-%d'))
            if cik:
                symbol_to_cik[sym] = cik

        self.logger.info(f"Resolved {len(symbol_to_cik)}/{len(symbols)} symbols to CIKs")

        # Check each CIK for recent filings
        checked = 0
        for sym, cik in symbol_to_cik.items():
            recent_filings = self.get_recent_edgar_filings(cik, lookback_days)

            if recent_filings:
                symbols_with_filings.add(sym)
                self.logger.debug(
                    f"{sym} (CIK {cik}): {len(recent_filings)} recent filings - "
                    f"{[f['form'] for f in recent_filings]}"
                )

            checked += 1
            if checked % 100 == 0:
                self.logger.info(f"Checked {checked}/{len(symbol_to_cik)} symbols for filings")

        self.logger.info(
            f"Found {len(symbols_with_filings)} symbols with recent filings "
            f"out of {len(symbol_to_cik)} checked"
        )

        return symbols_with_filings

    def update_daily_ticks(
        self,
        update_date: dt.date,
        symbols: Optional[List[str]] = None,
        max_workers: int = 50
    ) -> Dict[str, int]:
        """
        Update daily ticks for a specific date.

        Directly replace daily data without downloading existing files.
        Uses monthly partitioning to avoid large file downloads/uploads.

        Storage: data/raw/ticks/daily/{symbol}/{YYYY}/{MM}/ticks.parquet

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

        symbol_bars = self.alpaca_ticks.fetch_daily_day_bulk(
            symbols=symbols,
            trade_day=update_date.isoformat(),
            sleep_time=0.2
        )

        self.logger.info(
            f"Updating daily ticks for {len(symbols)} symbols for {update_date} "
            f"(monthly partition: {year}/{month:02d})"
        )

        # Check if month file exists for each symbol, download and merge if yes
        # Otherwise just upload new data
        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        import io
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_symbol(sym: str) -> Dict[str, Optional[str]]:
            """Process single symbol update"""
            try:
                ticks = symbol_bars.get(sym, [])

                if not ticks:
                    self.logger.debug(f"Skipping {sym} daily ticks: no data from Alpaca")
                    return {'symbol': sym, 'status': 'skipped', 'error': 'No data from Alpaca'}

                # Parse ticks to DataFrame
                parsed_ticks = self.alpaca_ticks.parse_ticks(ticks)
                from dataclasses import asdict
                ticks_data = [asdict(dp) for dp in parsed_ticks]

                new_df = pl.DataFrame(ticks_data).with_columns([
                    pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S').dt.date().cast(pl.Utf8),
                    pl.col('open').cast(pl.Float64).round(4),
                    pl.col('high').cast(pl.Float64).round(4),
                    pl.col('low').cast(pl.Float64).round(4),
                    pl.col('close').cast(pl.Float64).round(4),
                    pl.col('volume').cast(pl.Int64)
                ]).select(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                if len(new_df) == 0:
                    self.logger.debug(f"Skipping {sym} daily ticks: empty DataFrame after parsing")
                    return {'symbol': sym, 'status': 'skipped', 'error': 'Empty DataFrame after parsing'}

                # Check if monthly file exists
                s3_key = f"data/raw/ticks/daily/{sym}/{year}/{month:02d}/ticks.parquet"

                try:
                    response = self.s3_client.get_object(
                        Bucket=self.data_publishers.bucket_name,
                        Key=s3_key
                    )
                    existing_df = pl.read_parquet(response['Body'])

                    # Remove existing data for this date (if any) and append new
                    updated_df = pl.concat([
                        existing_df.filter(pl.col('timestamp') != update_date.isoformat()),
                        new_df
                    ]).sort('timestamp')

                except self.s3_client.exceptions.NoSuchKey:
                    # File doesn't exist yet, use new data only
                    updated_df = new_df
                except Exception as e:
                    # Other S3 errors, log and use new data only
                    self.logger.debug(f"Could not read existing file for {sym}: {e}, creating new file")
                    updated_df = new_df

                # Drop rows that only have timestamp (all other fields null)
                null_row = (
                    pl.col('open').is_null()
                    & pl.col('high').is_null()
                    & pl.col('low').is_null()
                    & pl.col('close').is_null()
                    & pl.col('volume').is_null()
                )
                updated_df = updated_df.filter(~null_row)

                # Upload to S3
                buffer = io.BytesIO()
                updated_df.write_parquet(buffer)
                buffer.seek(0)

                metadata = {
                    'symbol': sym,
                    'year': str(year),
                    'month': f"{month:02d}",
                    'data_type': 'daily_ticks',
                    'source': 'alpaca',
                    'trading_days': str(len(updated_df))
                }
                metadata_prepared = {
                    k: str(v) for k, v in metadata.items()
                }

                self.data_publishers.upload_fileobj(buffer, s3_key, metadata_prepared)

                return {'symbol': sym, 'status': 'success', 'error': None}

            except Exception as e:
                self.logger.error(f"Error updating daily ticks for {sym}: {e}")
                return {'symbol': sym, 'status': 'failed', 'error': str(e)}

        # Process symbols concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_symbol, sym): sym for sym in symbols}

            for future in as_completed(futures):
                result = future.result()

                if result['status'] == 'success':
                    stats['success'] += 1
                elif result['status'] == 'skipped':
                    stats['skipped'] += 1
                else:
                    stats['failed'] += 1

                # Progress logging
                total_processed = stats['success'] + stats['failed'] + stats['skipped']
                if total_processed % 100 == 0:
                    self.logger.info(
                        f"Progress: {total_processed}/{len(symbols)} "
                        f"({stats['success']} success, {stats['failed']} failed, {stats['skipped']} skipped)"
                    )

        self.logger.info(
            f"Daily ticks update completed: {stats['success']} success, "
            f"{stats['failed']} failed, {stats['skipped']} skipped"
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
        Storage: data/raw/ticks/minute/{symbol}/{YYYY}/{MM}/{DD}/ticks.parquet

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

        # Upload data concurrently
        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        self.logger.info(f"Uploading minute ticks for {len(parsed_data)} symbol-days...")

        for (sym, day), minute_df in parsed_data.items():
            try:
                if len(minute_df) == 0:
                    self.logger.debug(f"Skipping {sym} minute ticks for {day}: empty DataFrame")
                    stats['skipped'] += 1
                    continue

                # Upload to S3
                date_obj = dt.datetime.strptime(day, '%Y-%m-%d').date()
                year_str = date_obj.strftime('%Y')
                month_str = date_obj.strftime('%m')
                day_str = date_obj.strftime('%d')

                import io
                buffer = io.BytesIO()
                minute_df.write_parquet(buffer)
                buffer.seek(0)

                s3_key = f"data/raw/ticks/minute/{sym}/{year_str}/{month_str}/{day_str}/ticks.parquet"
                self.data_publishers.upload_fileobj(buffer, s3_key, {
                    'symbol': sym,
                    'trade_day': day,
                    'data_type': 'ticks'
                })

                stats['success'] += 1

            except Exception as e:
                self.logger.error(f"Error uploading minute ticks for {sym} on {day}: {e}")
                stats['failed'] += 1

        self.logger.info(
            f"Minute ticks update completed: {stats['success']} success, "
            f"{stats['failed']} failed, {stats['skipped']} skipped"
        )

        return stats

    def update_fundamental(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        max_workers: int = 10
    ) -> Dict[str, int]:
        """
        Update fundamental data for symbols with recent EDGAR filings.

        Updates three types of fundamental data:
        1. Raw fundamental data (data/raw/fundamental/{symbol}/fundamental.parquet)
        2. TTM features (data/derived/features/fundamental/{symbol}/ttm.parquet)
        3. Derived metrics (data/derived/features/fundamental/{symbol}/metrics.parquet)

        :param symbols: List of symbols to update
        :param start_date: Start date for data collection (YYYY-MM-DD)
        :param end_date: End date for data collection (YYYY-MM-DD)
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
                ttm_result = self.data_publishers.publish_ttm_fundamental(
                    sym=sym,
                    start_date=start_date,
                    end_date=end_date,
                    cik=cik,
                    sec_rate_limiter=self.sec_rate_limiter
                )

                # 3. Update derived metrics
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
                        derived_df=derived_df
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

    def run_daily_update(
        self,
        target_date: Optional[dt.date] = None,
        update_ticks: bool = True,
        update_fundamentals: bool = True,
        fundamental_lookback_days: int = 7
    ):
        """
        Run complete daily update workflow.

        Workflow:
        1. Check if market was open on target date
        2. If yes: Update daily and minute ticks
        3. Check EDGAR for recent filings
        4. Update fundamental data for symbols with new filings

        :param target_date: Date to update (default: yesterday)
        :param update_ticks: Whether to update ticks data
        :param update_fundamentals: Whether to update fundamental data
        :param fundamental_lookback_days: Days to look back for EDGAR filings
        """
        # Default to yesterday if no date specified
        if target_date is None:
            target_date = dt.date.today() - dt.timedelta(days=1)

        self.logger.info(f"=" * 80)
        self.logger.info(f"Starting daily update for {target_date}")
        self.logger.info(f"=" * 80)

        # Get current universe
        year = target_date.year
        symbols = self._get_symbols_for_year(year)

        # 1. Update ticks data (only if market was open)
        if update_ticks:
            market_open = self.check_market_open(target_date)

            if market_open:
                self.logger.info(f"Updating ticks data for {target_date}...")

                # Update daily ticks
                self.logger.info("Step 1/2: Updating daily ticks...")
                daily_stats = self.update_daily_ticks(target_date, symbols)

                # Update minute ticks
                self.logger.info("Step 2/2: Updating minute ticks...")
                minute_stats = self.update_minute_ticks(target_date, symbols)
            else:
                self.logger.info(
                    f"Market was closed on {target_date}, skipping ticks update"
                )

        # 2. Update fundamental data (check for recent filings)
        if update_fundamentals:
            self.logger.info("Checking EDGAR for recent filings...")

            symbols_with_filings = self.get_symbols_with_recent_filings(
                symbols=symbols,
                update_date=target_date,
                lookback_days=fundamental_lookback_days
            )

            if symbols_with_filings:
                self.logger.info(
                    f"Updating fundamental data for {len(symbols_with_filings)} symbols "
                    f"with recent filings..."
                )

                # Use a date range covering the lookback period
                end_date = dt.date.today().isoformat()
                start_date = (
                    dt.date.today() - dt.timedelta(days=fundamental_lookback_days)
                ).isoformat()

                fundamental_stats = self.update_fundamental(
                    symbols=list(symbols_with_filings),
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                self.logger.info("No symbols with recent filings, skipping fundamental update")

        self.logger.info(f"=" * 80)
        self.logger.info(f"Daily update completed for {target_date}")
        self.logger.info(f"=" * 80)


def main() -> None:
    """Main entry point for daily update."""
    import argparse

    parser = argparse.ArgumentParser(description="Run daily data lake update")
    parser.add_argument(
        '--date',
        type=str,
        help='Target date in YYYY-MM-DD format (default: yesterday)'
    )
    parser.add_argument(
        '--no-ticks',
        action='store_true',
        help='Skip ticks data update'
    )
    parser.add_argument(
        '--no-fundamentals',
        action='store_true',
        help='Skip fundamental data update'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=7,
        help='Days to look back for EDGAR filings (default: 7)'
    )

    args = parser.parse_args()

    if args.date:
        target_date = dt.datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target_date = None

    app = DailyUpdateApp()
    app.run_daily_update(
        target_date=target_date,
        update_ticks=not args.no_ticks,
        update_fundamentals=not args.no_fundamentals,
        fundamental_lookback_days=args.lookback
    )