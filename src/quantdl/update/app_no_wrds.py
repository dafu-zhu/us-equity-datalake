"""
Daily Update App (No WRDS Version) for US Equity Data Lake
===========================================================

WRDS-free version that uses:
- Nasdaq FTP for current universe (instead of CRSP historical)
- SEC API for CIK mapping (instead of WRDS SecurityMaster)

Suitable for GitHub Actions where WRDS has IP restrictions.
"""

import os
import logging
import datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import polars as pl
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import time
from threading import Semaphore

from quantdl.utils.logger import setup_logger
from quantdl.utils.calendar import TradingCalendar
from quantdl.storage.config_loader import UploadConfig
from quantdl.storage.s3_client import S3Client
from quantdl.storage.rate_limiter import RateLimiter
from quantdl.storage.data_collectors import DataCollectors
from quantdl.storage.data_publishers import DataPublishers
from quantdl.collection.alpaca_ticks import Ticks
from quantdl.collection.fundamental import SECClient
from quantdl.universe.current import fetch_all_stocks
from quantdl.master.security_master import SecurityMaster

load_dotenv()


class SimpleCIKResolver:
    """
    Lightweight CIK resolver using only SEC's public API (no WRDS).

    Suitable for recent data (2025+) where current SEC mapping is accurate.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._sec_cache: Optional[pl.DataFrame] = None

    def _fetch_sec_mapping(self) -> pl.DataFrame:
        """Fetch SEC's official CIK-Ticker mapping."""
        if self._sec_cache is not None:
            return self._sec_cache

        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            user_agent = os.getenv("SEC_USER_AGENT", "name@example.com")
            headers = {'User-Agent': user_agent}

            self.logger.info("Fetching SEC official CIK-Ticker mapping...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Convert to DataFrame
            records = []
            for key, value in data.items():
                ticker = value.get('ticker', '').replace('.', '').replace('-', '')  # Normalize
                cik_str = value.get('cik_str')
                if ticker and cik_str:
                    records.append({
                        'ticker': ticker,
                        'cik': str(cik_str).zfill(10)
                    })

            df = pl.DataFrame(records)
            self._sec_cache = df
            self.logger.info(f"Loaded {len(df)} ticker-CIK mappings from SEC")
            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch SEC CIK mapping: {e}")
            return pl.DataFrame({'ticker': [], 'cik': []})

    def get_cik(self, symbol: str, date: str = None, year: int = None) -> Optional[str]:
        """
        Get CIK for a symbol using SEC mapping.

        :param symbol: Symbol in any format (BRK-B, BRK.B, BRKB all work)
        :param date: Ignored (for API compatibility)
        :param year: Ignored (for API compatibility)
        :return: CIK string (zero-padded to 10 digits) or None
        """
        # Normalize symbol (remove dots and dashes)
        normalized = symbol.replace('.', '').replace('-', '')

        sec_map = self._fetch_sec_mapping()
        if sec_map.is_empty():
            return None

        match = sec_map.filter(pl.col('ticker') == normalized).select('cik').head(1)
        if not match.is_empty():
            return match.item()

        return None

    def batch_prefetch_ciks(self, symbols: List[str], year: int = None) -> Dict[str, Optional[str]]:
        """
        Batch resolve symbols to CIKs.

        :param symbols: List of symbols
        :param year: Ignored (for API compatibility)
        :return: Dict mapping symbol to CIK
        """
        sec_map = self._fetch_sec_mapping()
        if sec_map.is_empty():
            return {sym: None for sym in symbols}

        # Normalize all symbols
        symbol_map = {sym: sym.replace('.', '').replace('-', '') for sym in symbols}

        # Batch lookup
        result = {}
        for sym, normalized in symbol_map.items():
            match = sec_map.filter(pl.col('ticker') == normalized).select('cik').head(1)
            result[sym] = match.item() if not match.is_empty() else None

        return result


class DailyUpdateAppNoWRDS:
    """
    Daily update app that works without WRDS connection.

    Differences from standard DailyUpdateApp:
    - Uses Nasdaq FTP for universe (fetch_all_stocks)
    - Uses SEC API for CIK mapping (SimpleCIKResolver)
    - No CRSP/SecurityMaster/UniverseManager
    """

    def __init__(self, config_path: str = "configs/storage.yaml"):
        """Initialize the daily update app (no WRDS)."""
        # Setup config and clients
        self.config = UploadConfig(config_path)
        self.s3_client = S3Client(config_path).client

        # Setup logger
        self.logger = setup_logger(
            name="daily_update_no_wrds",
            log_dir=Path("data/logs/update"),
            level=logging.DEBUG,
            console_output=True
        )

        self.logger.info("Initializing WRDS-free daily update app...")

        # Initialize calendar
        self.calendar = TradingCalendar()

        # Initialize Alpaca ticks (no WRDS needed)
        self.alpaca_ticks = Ticks()

        # Load Alpaca credentials
        ALPACA_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET
        }

        # Rate limiters
        self.sec_rate_limiter = RateLimiter(max_rate=9.5)

        # Initialize CIK resolver (SEC-only)
        self.cik_resolver = SimpleCIKResolver(logger=self.logger)

        # Initialize SecurityMaster from S3 (read-only, no WRDS needed)
        self.security_master = SecurityMaster(
            s3_client=self.s3_client,
            bucket_name='us-equity-datalake'
        )

        # Initialize data collectors (pass None for crsp_ticks since we don't use it)
        self.data_collectors = DataCollectors(
            crsp_ticks=None,  # Not used in updates
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

        # Symbols cache
        self._symbols_cache: Optional[List[str]] = None

    def check_market_open(self, date: dt.date) -> bool:
        """Check if market was open on a given date."""
        is_open = self.calendar.is_trading_day(date)
        self.logger.info(f"Market {'was' if is_open else 'was NOT'} open on {date}")
        return is_open

    def _get_symbols(self) -> List[str]:
        """Get current universe from Nasdaq (cached)."""
        if self._symbols_cache is not None:
            return self._symbols_cache

        self.logger.info("Fetching current stock universe from Nasdaq...")
        df = fetch_all_stocks(with_filter=True, refresh=True, logger=self.logger)

        # Convert to Alpaca format (symbols as-is from Nasdaq)
        symbols = df['Ticker'].tolist()
        self._symbols_cache = symbols

        self.logger.info(f"Loaded {len(symbols)} symbols from Nasdaq")
        return symbols

    def get_recent_edgar_filings(self, cik: str, lookback_days: int = 7) -> List[Dict]:
        """Check EDGAR for recent filings for a given CIK."""
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

        try:
            self.sec_rate_limiter.acquire()
            response = requests.get(url, headers=self.sec_client.header)
            response.raise_for_status()
            data = response.json()

            recent_filings = data.get('filings', {}).get('recent', {})
            if not recent_filings:
                return []

            filing_dates = recent_filings.get('filingDate', [])
            forms = recent_filings.get('form', [])
            accession_numbers = recent_filings.get('accessionNumber', [])

            relevant_forms = {'10-K', '10-Q', '10-K/A', '10-Q/A', '8-K'}
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
        symbols: List[str],
        update_date: dt.date,
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
        """Update daily ticks for a specific date."""
        year = update_date.year
        month = update_date.month

        if symbols is None:
            symbols = self._get_symbols()

        symbol_bars = self.alpaca_ticks.fetch_daily_day_bulk(
            symbols=symbols,
            trade_day=update_date.isoformat(),
            sleep_time=0.2
        )

        self.logger.info(
            f"Updating daily ticks for {len(symbols)} symbols for {update_date} "
            f"(monthly partition: {year}/{month:02d})"
        )

        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        import io

        def process_symbol(sym: str) -> Dict[str, Optional[str]]:
            """Process single symbol update"""
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

                # Check if monthly file exists (security_id-based path)
                s3_key = f"data/raw/ticks/daily/{security_id}/{year}/{month:02d}/ticks.parquet"

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

                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        updated_df = new_df  # File doesn't exist yet - OK
                    else:
                        raise  # Re-raise to trigger retry or mark as failed
                except Exception as e:
                    raise  # Re-raise to trigger retry or mark as failed

                # Drop null rows
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
                metadata_prepared = {k: str(v) for k, v in metadata.items()}

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

    def update_minute_ticks(
        self,
        update_date: dt.date,
        symbols: Optional[List[str]] = None,
        max_workers: int = 50
    ) -> Dict[str, int]:
        """Update minute ticks for a specific trading day."""
        year = update_date.year
        trade_day_str = update_date.strftime('%Y-%m-%d')

        if symbols is None:
            symbols = self._get_symbols()

        self.logger.info(
            f"Updating minute ticks for {len(symbols)} symbols for {trade_day_str}"
        )

        # Fetch minute data
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

        pbar = tqdm(parsed_data.items(), desc="Minute ticks", unit="sym-day")
        for (sym, day), minute_df in pbar:
            try:
                if len(minute_df) == 0:
                    self.logger.debug(f"Skipping {sym} minute ticks for {day}: empty DataFrame")
                    stats['skipped'] += 1
                    pbar.set_postfix(ok=stats['success'], fail=stats['failed'], skip=stats['skipped'])
                    continue

                # Resolve symbol to security_id at trade day
                security_id = self.security_master.get_security_id(sym, day)

                # Upload to S3
                date_obj = dt.datetime.strptime(day, '%Y-%m-%d').date()
                year_str = date_obj.strftime('%Y')
                month_str = date_obj.strftime('%m')
                day_str = date_obj.strftime('%d')

                import io
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

                stats['success'] += 1

            except Exception as e:
                self.logger.error(f"Error uploading minute ticks for {sym} on {day}: {e}")
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
        """Update fundamental data for symbols with recent EDGAR filings (selectively)."""
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

    def run_daily_update(
        self,
        target_date: Optional[dt.date] = None,
        update_daily_ticks: bool = True,
        update_minute_ticks: bool = True,
        update_fundamental: bool = True,
        update_ttm: bool = True,
        update_derived: bool = True,
        fundamental_lookback_days: int = 7
    ):
        """Run complete daily update workflow (no WRDS required)."""
        if target_date is None:
            target_date = dt.date.today() - dt.timedelta(days=1)

        self.logger.info(f"=" * 80)
        self.logger.info(f"Starting daily update for {target_date} (WRDS-free mode)")
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

        # Get current universe
        symbols = self._get_symbols()

        # 1. Update ticks data (only if market was open)
        if update_daily_ticks or update_minute_ticks:
            market_open = self.check_market_open(target_date)

            if market_open:
                self.logger.info(f"Updating ticks data for {target_date}...")

                # Update daily ticks
                if update_daily_ticks:
                    self.logger.info("Updating daily ticks...")
                    daily_stats = self.update_daily_ticks(target_date, symbols)

                # Update minute ticks
                if update_minute_ticks:
                    self.logger.info("Updating minute ticks...")
                    minute_stats = self.update_minute_ticks(target_date, symbols)
            else:
                self.logger.info(
                    f"Market was closed on {target_date}, skipping ticks update"
                )

        # 2. Update fundamental data (check for recent filings)
        if update_fundamental or update_ttm or update_derived:
            self.logger.info("Checking EDGAR for recent filings...")

            symbols_with_fundamental_filings, symbols_with_filings, filing_stats = self.get_symbols_with_recent_filings(
                symbols=symbols,
                update_date=target_date,
                lookback_days=fundamental_lookback_days
            )

            if symbols_with_fundamental_filings:
                self.logger.info(
                    f"Updating fundamental data for {len(symbols_with_fundamental_filings)} symbols "
                    f"with 10-K/10-Q filings..."
                )

                end_date = dt.date.today().isoformat()
                start_date = (
                    dt.date.today() - dt.timedelta(days=fundamental_lookback_days)
                ).isoformat()

                fundamental_stats = self.update_fundamental(
                    symbols=list(symbols_with_fundamental_filings),
                    start_date=start_date,
                    end_date=end_date,
                    update_raw=update_fundamental,
                    update_ttm=update_ttm,
                    update_derived=update_derived
                )
            else:
                self.logger.info("No symbols with 10-K/10-Q filings, skipping fundamental update")

        self.logger.info(f"=" * 80)
        self.logger.info(f"Daily update completed for {target_date}")
        self.logger.info(f"=" * 80)
