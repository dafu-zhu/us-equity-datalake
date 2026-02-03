import os
from typing import List, Optional, Dict
from pathlib import Path
import logging
import datetime as dt
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from botocore.exceptions import ClientError
from dotenv import load_dotenv
import polars as pl

from quantdl.utils.logger import setup_logger
from quantdl.utils.calendar import TradingCalendar
from quantdl.storage.config_loader import UploadConfig
from quantdl.storage.s3_client import S3Client
from quantdl.storage.validation import Validator
from quantdl.storage.rate_limiter import RateLimiter
from quantdl.storage.cik_resolver import CIKResolver
from quantdl.storage.data_collectors import DataCollectors
from quantdl.storage.data_publishers import DataPublishers
from quantdl.storage.progress_tracker import UploadProgressTracker
from quantdl.collection.alpaca_ticks import Ticks
from quantdl.collection.crsp_ticks import CRSPDailyTicks
from quantdl.universe.manager import UniverseManager
from quantdl.master.security_master import SecurityMaster

load_dotenv()


class UploadApp:
    def __init__(
        self,
        alpaca_start_year: int = 2025
    ):
        # Setup config and client
        self.config = UploadConfig()
        self.client = S3Client().client

        # Setup logger
        self.logger = setup_logger(
            name=f"uploadapp",
            log_dir=Path("data/logs/upload"),
            level=logging.DEBUG,
            console_output = True
        )

        self.validator = Validator(self.client)

        # Initialize data fetchers once to reuse connections
        self.alpaca_ticks = Ticks()

        # Try WRDS connection (needed for <2025 backfill, optional for 2025+)
        try:
            self.crsp_ticks = CRSPDailyTicks(
                s3_client=self.client,
                bucket_name='us-equity-datalake',
                require_wrds=True  # Require WRDS for historical data (<2025)
            )
            self.security_master = self.crsp_ticks.security_master
            self._wrds_available = (self.crsp_ticks._conn is not None)

            if not self._wrds_available:
                self.logger.warning(
                    "WRDS connection unavailable. "
                    "Can upload 2025+ data (Alpaca), but cannot backfill <2025 (CRSP required)."
                )
        except Exception as e:
            self.logger.error(f"CRSPDailyTicks initialization failed: {e}")
            raise RuntimeError(
                "Cannot initialize UploadApp without SecurityMaster. "
                "Ensure S3 export exists: uv run quantdl-export-security-master --export"
            )

        # Initialize universe manager
        self.universe_manager = UniverseManager(
            crsp_fetcher=self.crsp_ticks if self._wrds_available else None,
            security_master=self.security_master
        )

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
            security_master=self.security_master,
            logger=self.logger
        )

        self.data_collectors = DataCollectors(
            crsp_ticks=self.crsp_ticks,
            alpaca_ticks=self.alpaca_ticks,
            alpaca_headers=self.headers,
            logger=self.logger,
            sec_rate_limiter=self.sec_rate_limiter,
            alpaca_start_year=alpaca_start_year
        )

        self.data_publishers = DataPublishers(
            s3_client=self.client,
            upload_config=self.config,
            logger=self.logger,
            data_collectors=self.data_collectors,
            security_master=self.crsp_ticks.security_master,
            alpaca_start_year=alpaca_start_year
        )

    # ===========================
    # Upload ticks
    # ===========================
    def _publish_single_daily_ticks(
            self,
            sym: str,
            year: int,
            month: Optional[int] = None,
            overwrite: bool = False,
            use_monthly_partitions: bool = True
        ) -> Dict[str, Optional[str]]:
        """
        Process daily ticks for a single symbol for entire year or specific month.
        Returns dict with status for progress tracking.

        Storage:
        - Monthly (current year): data/raw/ticks/daily/{security_id}/{YYYY}/{MM}/ticks.parquet
        - History (completed years): data/raw/ticks/daily/{security_id}/history.parquet

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :param month: Optional month (1-12) for monthly partitioning
        :param overwrite: If True, skip existence check and overwrite existing data
        :param use_monthly_partitions: If True, use monthly partitioning (default)
        """
        # Resolve symbol to security_id
        try:
            security_id = self.crsp_ticks.security_master.get_security_id(sym, f"{year}-12-31")
        except ValueError as e:
            # Symbol not active in this year - skip
            return {
                'symbol': sym,
                'status': 'skipped',
                'error': str(e)
            }

        # Validate: Check if data already exists (before collection)
        if not overwrite and self.validator.data_exists(sym, 'ticks', year, month, security_id=security_id):
            return {
                'symbol': sym,
                'status': 'canceled',
                'error': f'Symbol {sym} for {year}{f"/{month:02d}" if month else ""} already exists'
            }

        # Fetch data
        if month is not None:
            # Fetch specific month
            df = self.data_collectors.collect_daily_ticks_month(sym, year, month)
        else:
            # Fetch entire year
            df = self.data_collectors.collect_daily_ticks_year(sym, year)

        # Publish to S3
        if use_monthly_partitions and month is not None:
            return self.data_publishers.publish_daily_ticks(sym, year, security_id, df, month=month, by_year=False)
        else:
            return self.data_publishers.publish_daily_ticks(sym, year, security_id, df, by_year=False)

    def _upload_crsp_bulk_history(
        self,
        start_year: int = 2009,
        end_year: int = 2024,
        overwrite: bool = False,
        chunk_size: int = 500,
        resume: bool = True
    ):
        """
        Upload daily ticks for CRSP years using permno-centric bulk fetch.

        Fetches complete history per permno ONCE instead of year-by-year.
        Much more efficient: ~10 bulk queries vs ~80,000 symbol-year fetches.

        :param start_year: Start year for backfill (default: 2009)
        :param end_year: End year for backfill (default: 2024)
        :param overwrite: If True, ignore progress and re-fetch all
        :param chunk_size: Number of permnos per CRSP query (default: 500)
        :param resume: If True, skip already completed security_ids
        """
        if not self._wrds_available:
            raise RuntimeError(
                "CRSP bulk history upload requires WRDS connection. "
                "Ensure WRDS credentials are set."
            )

        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        self.logger.info(
            f"Starting CRSP bulk history upload: {start_year}-{end_year}"
        )
        self.logger.info(f"Strategy: permno-centric fetch, {chunk_size} permnos/batch")

        # Step 1: Build permno -> [(security_id, start, end), ...] mapping
        self.logger.info("Step 1/4: Building permno -> security_id mapping...")
        master_tb = self.security_master.master_tb

        # Get all unique permnos and their security_id mappings
        permno_sid_map: Dict[int, List[tuple]] = {}
        all_security_ids = set()

        for row in master_tb.iter_rows(named=True):
            permno = row['permno']
            sid = row['security_id']
            start = row['start_date']
            end = row['end_date']

            if sid is None or permno is None:
                continue

            all_security_ids.add(sid)
            if permno not in permno_sid_map:
                permno_sid_map[permno] = []
            permno_sid_map[permno].append((sid, start, end))

        unique_permnos = list(permno_sid_map.keys())
        self.logger.info(
            f"Found {len(unique_permnos)} unique permnos, "
            f"{len(all_security_ids)} unique security_ids"
        )

        # Step 2: Filter out completed security_ids (if resume)
        tracker = UploadProgressTracker(
            s3_client=self.client,
            bucket_name='us-equity-datalake',
            task_name='daily_ticks_crsp_bulk'
        )

        if resume and not overwrite:
            self.logger.info("Step 2/4: Checking progress for resume...")
            completed_sids = tracker.load()
            pending_sids = all_security_ids - completed_sids

            # Filter permnos to only those with pending security_ids
            permnos_to_fetch = []
            for permno, sid_list in permno_sid_map.items():
                # Check if any security_id for this permno is pending
                if any(sid in pending_sids for sid, _, _ in sid_list):
                    permnos_to_fetch.append(permno)

            self.logger.info(
                f"Resuming: {len(completed_sids)} security_ids completed, "
                f"{len(pending_sids)} pending, {len(permnos_to_fetch)} permnos to fetch"
            )
        else:
            permnos_to_fetch = unique_permnos
            pending_sids = all_security_ids
            if overwrite:
                tracker.reset()

        tracker.set_total(len(all_security_ids))

        if not permnos_to_fetch:
            self.logger.info("All security_ids already completed. Nothing to do.")
            return

        # Step 3: Batch fetch full history
        self.logger.info(
            f"Step 3/4: Fetching {len(permnos_to_fetch)} permnos "
            f"({start_date} to {end_date})..."
        )

        # Fetch in batches
        success = 0
        failed = 0
        skipped = 0

        with tracker:
            pbar = tqdm(
                total=len(pending_sids),
                desc="Uploading CRSP history",
                unit="sid"
            )

            for i in range(0, len(permnos_to_fetch), chunk_size):
                chunk = permnos_to_fetch[i:i + chunk_size]

                # Bulk fetch complete history for this chunk
                permno_data = self.crsp_ticks.collect_daily_ticks_full_history_bulk(
                    permnos=chunk,
                    start_date=start_date,
                    end_date=end_date,
                    chunk_size=chunk_size
                )

                # Step 4: Split by security_id and publish
                for permno in chunk:
                    if permno not in permno_data:
                        # No data for this permno (likely delisted before start_year)
                        for sid, _, _ in permno_sid_map.get(permno, []):
                            if sid in pending_sids:
                                skipped += 1
                                tracker.mark_skipped(sid)
                                pbar.update(1)
                        continue

                    df = permno_data[permno]

                    # Split data by security_id date ranges
                    for sid, sid_start, sid_end in permno_sid_map.get(permno, []):
                        if sid not in pending_sids:
                            continue  # Already completed

                        # Filter data for this security_id's date range
                        sid_df = df.filter(
                            (pl.col('timestamp') >= str(sid_start)) &
                            (pl.col('timestamp') <= str(sid_end))
                        )

                        if len(sid_df) == 0:
                            skipped += 1
                            tracker.mark_skipped(sid)
                            pbar.update(1)
                            pbar.set_postfix(ok=success, fail=failed, skip=skipped)
                            continue

                        # Get symbol for this security_id (for logging)
                        symbol_info = master_tb.filter(
                            pl.col('security_id') == sid
                        ).select('symbol').head(1)
                        symbol = symbol_info['symbol'][0] if len(symbol_info) > 0 else f"sid_{sid}"

                        # Publish to history.parquet
                        try:
                            result = self.data_publishers.publish_daily_ticks_to_history(
                                security_id=sid,
                                df=sid_df,
                                symbol=symbol
                            )
                            if result.get('status') == 'success':
                                success += 1
                                tracker.mark_completed(sid)
                            else:
                                failed += 1
                                tracker.mark_failed(sid)
                        except Exception as e:
                            self.logger.error(f"Failed to publish sid={sid}: {e}")
                            failed += 1
                            tracker.mark_failed(sid)

                        pbar.update(1)
                        pbar.set_postfix(ok=success, fail=failed, skip=skipped)

            pbar.close()

        self.logger.info(
            f"CRSP bulk history upload completed: {success} success, "
            f"{failed} failed, {skipped} skipped"
        )

    def upload_daily_ticks(
        self,
        year: int,
        overwrite: bool = False,
        use_monthly_partitions: bool = True,
        by_year: bool = False,
        chunk_size: int = 200,
        sleep_time: float = 0.2,
        current_year: Optional[int] = None
    ):
        """
        Upload daily ticks for all symbols for an entire year.

        Storage strategy:
        - Completed years (year < current_year): data/raw/ticks/daily/{security_id}/history.parquet (consolidated)
        - Current year (year == current_year): data/raw/ticks/daily/{security_id}/{YYYY}/{MM}/ticks.parquet (monthly)

        Completed years are appended to history.parquet incrementally during backfill.
        Current year uses monthly partitions for efficient daily updates.

        Uses CRSP for years < 2025, Alpaca for years >= 2025.
        Sequential processing to avoid rate limits.

        :param year: Year to fetch data for
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        :param use_monthly_partitions: If True, use monthly partitions (default: True)
        :param by_year: If True, collect year data once and publish monthly partitions in parallel
        :param current_year: Current year for determining storage mode (default: datetime.now().year)
        """
        # NEW: Check WRDS availability for historical years
        if year < 2025 and not self._wrds_available:
            raise RuntimeError(
                f"Cannot upload daily ticks for year {year}: WRDS connection required for CRSP data (years < 2025). "
                "Available: Alpaca data only (2025+). "
                "To backfill historical data, ensure WRDS credentials are set."
            )

        # Determine current year if not provided
        import datetime
        if current_year is None:
            current_year = datetime.datetime.now().year

        # Determine storage mode: history for completed years, monthly for current year
        is_completed_year = (year < current_year)

        # Override storage mode for completed years
        if is_completed_year:
            use_monthly_partitions = False
            self.logger.info(f"Year {year} is completed (< {current_year}), will upload to history.parquet")

        # Load symbols for this year
        alpaca_symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')

        total_symbols = len(alpaca_symbols)
        alpaca_start_year = self.data_collectors.ticks_collector.alpaca_start_year
        data_source = 'crsp' if year < alpaca_start_year else 'alpaca'

        # Pre-resolve security_ids for all symbols (used for data_exists and publishing)
        security_id_cache = {}
        for sym in alpaca_symbols:
            try:
                security_id_cache[sym] = self.crsp_ticks.security_master.get_security_id(sym, f"{year}-12-31")
            except ValueError:
                security_id_cache[sym] = None  # Symbol not active in this year

        if use_monthly_partitions:
            # Upload month by month for better organization
            self.logger.info(
                f"Starting {year} daily ticks upload for {total_symbols} symbols "
                f"(source={data_source}, monthly partitions, sequential processing, overwrite={overwrite}, "
                f"by_year={by_year})"
            )
            self.logger.info(f"Storage: data/raw/ticks/daily/{{security_id}}/{year}/{{MM}}/ticks.parquet")
            self.logger.info(f"Starting year {year} ({total_symbols} symbols)")

            if year >= alpaca_start_year:
                if by_year:
                    self.logger.info(
                        "Alpaca daily ticks will use monthly bulk fetch; ignoring by_year=True."
                    )

                for month in range(1, 13):
                    # Skip future months
                    today = dt.date.today()
                    if year == today.year and month > today.month:
                        self.logger.info(f"Skipping {year}-{month:02d}: future month")
                        continue

                    success = 0
                    failed = 0
                    canceled = 0
                    skipped = 0

                    pbar = tqdm(total=total_symbols, desc=f"Uploading {year}-{month:02d}", unit="sym")
                    for i in range(0, total_symbols, chunk_size):
                        chunk = alpaca_symbols[i:i + chunk_size]

                        if not overwrite:
                            symbols_to_fetch = []
                            for sym in chunk:
                                sec_id = security_id_cache.get(sym)
                                if sec_id is None:
                                    skipped += 1
                                    pbar.update(1)
                                elif self.validator.data_exists(sym, 'ticks', year, month, security_id=sec_id):
                                    canceled += 1
                                    pbar.update(1)
                                else:
                                    symbols_to_fetch.append(sym)
                            pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)
                            if not symbols_to_fetch:
                                continue
                            chunk = symbols_to_fetch

                        symbol_map = self.data_collectors.collect_daily_ticks_month_bulk(
                            chunk,
                            year,
                            month,
                            sleep_time=sleep_time
                        )

                        for sym in chunk:
                            security_id = security_id_cache[sym]
                            df = symbol_map.get(sym, pl.DataFrame())
                            result = self.data_publishers.publish_daily_ticks(
                                sym,
                                year,
                                security_id,
                                df,
                                month=month,
                                by_year=False
                            )

                            if result['status'] == 'success':
                                success += 1
                            elif result['status'] == 'canceled':
                                canceled += 1
                            elif result['status'] == 'skipped':
                                skipped += 1
                            else:
                                failed += 1

                            pbar.update(1)
                            pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)

                    pbar.close()
                    self.logger.info(
                        f"{year}-{month:02d} completed: {failed} failed, {success} success, "
                        f"{canceled} canceled, {skipped} skipped out of {total_symbols} total"
                    )
                return

            completed = 0
            success = 0
            failed = 0
            canceled = 0
            skipped = 0

            if by_year:
                bulk_year_map = {}
                if year < 2025:
                    bulk_year_map = self.data_collectors.collect_daily_ticks_year_bulk(
                        alpaca_symbols,
                        year
                    )

                pbar = tqdm(alpaca_symbols, desc=f"Uploading {year} symbols", unit="sym")
                for sym in pbar:
                    security_id = security_id_cache.get(sym)
                    if security_id is None:
                        skipped += 1
                        completed += 1
                        pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)
                        continue

                    if not overwrite:
                        any_exists = any(
                            self.validator.data_exists(sym, 'ticks', year, month, security_id=security_id)
                            for month in range(1, 13)
                        )
                        if any_exists:
                            canceled += 1
                            completed += 1
                            pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)
                            continue

                    year_df = None
                    if year < 2025:
                        year_df = bulk_year_map.get(sym, pl.DataFrame())

                    result = self.data_publishers.publish_daily_ticks(
                        sym,
                        year,
                        security_id,
                        df=None,
                        by_year=True,
                        year_df=year_df
                    )
                    completed += 1

                    if result['status'] == 'success':
                        success += 1
                    elif result['status'] == 'canceled':
                        canceled += 1
                    elif result['status'] == 'skipped':
                        skipped += 1
                    else:
                        failed += 1

                    pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)

                self.logger.info(
                    f"{year} completed: {failed} failed, {success} success, "
                    f"{canceled} canceled, {skipped} skipped out of {total_symbols} total"
                )
            else:
                # Process each month
                for month in range(1, 13):
                    completed = 0
                    success = 0
                    failed = 0
                    canceled = 0
                    skipped = 0

                    pbar = tqdm(
                        alpaca_symbols,
                        desc=f"Uploading {year}-{month:02d} symbols",
                        unit="sym"
                    )
                    for sym in pbar:
                        result = self._publish_single_daily_ticks(
                            sym, year, month=month, overwrite=overwrite, use_monthly_partitions=True
                        )
                        completed += 1

                        if result['status'] == 'success':
                            success += 1
                        elif result['status'] == 'canceled':
                            canceled += 1
                        elif result['status'] == 'skipped':
                            skipped += 1
                        else:
                            failed += 1

                        pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)

                    self.logger.info(
                        f"{year}-{month:02d} completed: {failed} failed, {success} success, "
                        f"{canceled} canceled, {skipped} skipped out of {total_symbols} total"
                    )

        else:
            # History mode (for completed years)
            storage_desc = "history.parquet (consolidated)" if is_completed_year else "yearly partitions (legacy)"
            self.logger.info(
                f"Starting {year} daily ticks upload for {total_symbols} symbols "
                f"(source={data_source}, {storage_desc}, sequential processing, overwrite={overwrite})"
            )
            storage_path = f"data/raw/ticks/daily/{{security_id}}/history.parquet" if is_completed_year else f"data/raw/ticks/daily/{{security_id}}/{year}/ticks.parquet"
            self.logger.info(f"Storage: {storage_path}")

            success = 0
            failed = 0
            canceled = 0
            skipped = 0

            if year >= alpaca_start_year:
                pbar = tqdm(total=total_symbols, desc=f"Uploading {year} history", unit="sym")
                for i in range(0, total_symbols, chunk_size):
                    chunk = alpaca_symbols[i:i + chunk_size]

                    if not overwrite:
                        symbols_to_fetch = []
                        for sym in chunk:
                            sec_id = security_id_cache.get(sym)
                            if sec_id is None:
                                skipped += 1
                                pbar.update(1)
                            elif self.validator.data_exists(sym, 'ticks', year, security_id=sec_id):
                                canceled += 1
                                pbar.update(1)
                            else:
                                symbols_to_fetch.append(sym)
                        pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)
                        if not symbols_to_fetch:
                            continue
                        chunk = symbols_to_fetch

                    symbol_map = self.data_collectors.collect_daily_ticks_year_bulk(
                        chunk,
                        year
                    )

                    for sym in chunk:
                        security_id = security_id_cache[sym]
                        df = symbol_map.get(sym, pl.DataFrame())
                        result = self.data_publishers.publish_daily_ticks(
                            sym,
                            year,
                            security_id,
                            df,
                            month=None,
                            by_year=False
                        )

                        if result['status'] == 'success':
                            success += 1
                        elif result['status'] == 'canceled':
                            canceled += 1
                        elif result['status'] == 'skipped':
                            skipped += 1
                        else:
                            failed += 1

                        pbar.update(1)
                        pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)

                pbar.close()
                self.logger.info(
                    f"{year} Daily ticks upload completed ({data_source}): {success} success, "
                    f"{failed} failed, {canceled} canceled, {skipped} skipped out of {total_symbols} total"
                )
                return

            pbar = tqdm(alpaca_symbols, desc=f"Uploading {year} symbols", unit="sym")
            for sym in pbar:
                result = self._publish_single_daily_ticks(
                    sym, year, overwrite=overwrite, use_monthly_partitions=False
                )

                if result['status'] == 'success':
                    success += 1
                elif result['status'] == 'canceled':
                    canceled += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                else:
                    failed += 1

                pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)

            self.logger.info(
                f"{year} Daily ticks upload completed ({data_source}): {success} success, "
                f"{failed} failed, {canceled} canceled, {skipped} skipped out of {total_symbols} total"
            )

    def upload_minute_ticks(self, year: int, month: int, overwrite: bool = False, resume: bool = False, num_workers: int = 50, chunk_size: int = 500, sleep_time: float = 0.0):
        """
        Upload minute ticks for a single month. Creates workers per call.
        For multi-month uploads, use upload_minute_ticks_year() instead.
        """
        self.upload_minute_ticks_year(
            year=year,
            months=[month],
            overwrite=overwrite,
            resume=resume,
            num_workers=num_workers,
            chunk_size=chunk_size,
            sleep_time=sleep_time
        )

    def upload_minute_ticks_year(
        self,
        year: int,
        months: list[int] = None,
        overwrite: bool = False,
        resume: bool = False,
        num_workers: int = 50,
        chunk_size: int = 500,
        sleep_time: float = 0.0
    ):
        """
        Upload minute ticks for multiple months with shared worker pool.
        Workers created once and reused across all months.

        :param year: Year to fetch data for
        :param months: List of months (1-12). Default: all 12 months
        :param overwrite: If True, overwrite existing data in S3
        :param resume: If True, skip symbols already verified complete
        :param num_workers: Number of concurrent processing workers
        :param chunk_size: Number of symbols to fetch at once
        :param sleep_time: Sleep time between API requests in seconds
        """
        if months is None:
            months = list(range(1, 13))

        # Load symbols once for the year
        alpaca_symbols_full = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')

        # Shared statistics across all months
        stats = {'success': 0, 'failed': 0, 'canceled': 0, 'skipped': 0, 'completed': 0}
        stats_lock = threading.Lock()

        # Single queue for all months
        data_queue = queue.Queue(maxsize=200)

        self.logger.info(f"Starting {year} minute ticks upload ({len(months)} months) | {num_workers} workers | chunk_size={chunk_size}")

        # Start consumer threads ONCE for all months
        workers = []
        for _ in range(num_workers):
            worker = threading.Thread(
                target=self.data_publishers.minute_ticks_worker,
                args=(data_queue, stats, stats_lock),
                daemon=True
            )
            worker.start()
            workers.append(worker)

        try:
            for month in months:
                self._upload_minute_ticks_month(
                    year=year,
                    month=month,
                    alpaca_symbols_full=alpaca_symbols_full,
                    overwrite=overwrite,
                    resume=resume,
                    chunk_size=chunk_size,
                    sleep_time=sleep_time,
                    data_queue=data_queue,
                    stats=stats,
                    stats_lock=stats_lock
                )
        finally:
            # Signal workers to stop
            for _ in range(num_workers):
                data_queue.put(None)

            # Wait for all workers to finish
            for worker in workers:
                worker.join()

        self.logger.info(
            f"Year {year} minute ticks completed: {stats['failed']} failed, {stats['success']} success, "
            f"{stats['canceled']} canceled, {stats['skipped']} skipped"
        )

    def _upload_minute_ticks_month(
        self,
        year: int,
        month: int,
        alpaca_symbols_full: list[str],
        overwrite: bool,
        resume: bool,
        chunk_size: int,
        sleep_time: float,
        data_queue: queue.Queue,
        stats: dict,
        stats_lock: threading.Lock
    ):
        """Process a single month using shared worker pool."""
        # Initialize progress tracker for this month
        tracker = UploadProgressTracker(
            s3_client=self.client,
            bucket_name='us-equity-datalake',
            task_name=f'minute_ticks_{year}_{month:02d}',
            key_type='str'
        )

        # Filter out already-completed symbols if resuming
        alpaca_symbols = alpaca_symbols_full.copy()
        if resume and not overwrite:
            completed_symbols = tracker.load()
            original_count = len(alpaca_symbols)
            alpaca_symbols = [s for s in alpaca_symbols if s not in completed_symbols]
            self.logger.info(f"{year}-{month:02d}: {len(completed_symbols)} complete, {len(alpaca_symbols)} pending")
        elif overwrite:
            tracker.reset()

        trading_days = self.calendar.load_trading_days(year, month)

        # Filter out future dates
        today = dt.date.today()
        today_str = today.strftime("%Y-%m-%d")
        trading_days = [d for d in trading_days if d <= today_str]

        if not trading_days:
            self.logger.info(f"Skipping {year}-{month:02d}: no valid trading days (future month)")
            return

        total_symbols = len(alpaca_symbols)
        total_days = len(trading_days)
        total_tasks = total_symbols * total_days

        if total_tasks == 0:
            self.logger.info(f"{year}-{month:02d}: Nothing to process")
            return

        pbar = tqdm(total=total_tasks, desc=f"Uploading {year}-{month:02d} minute", unit="task")
        try:
            with tracker:
                for i in range(0, total_symbols, chunk_size):
                    chunk = alpaca_symbols[i:i + chunk_size]

                    # Pre-filter symbols that need data
                    if not overwrite:
                        symbols_to_fetch = []
                        for sym in chunk:
                            try:
                                security_id = self.security_master.get_security_id(sym, trading_days[0])
                            except ValueError:
                                symbols_to_fetch.append(sym)
                                continue

                            if security_id is None:
                                symbols_to_fetch.append(sym)
                                continue

                            existing_days = self.validator.get_existing_minute_days(security_id, year, month)
                            trading_day_set = {day.split('-')[2] for day in trading_days}
                            if not trading_day_set.issubset(existing_days):
                                symbols_to_fetch.append(sym)
                            else:
                                tracker.mark_completed(sym)
                                canceled_count = len(trading_days)
                                with stats_lock:
                                    stats['canceled'] += canceled_count
                                    stats['completed'] += canceled_count
                                pbar.update(canceled_count)

                        pbar.set_postfix(ok=stats['success'], fail=stats['failed'], skip=stats['skipped'], cancel=stats['canceled'])
                        if not symbols_to_fetch:
                            continue
                        chunk = symbols_to_fetch

                    # Bulk fetch for the month
                    symbol_bars = self.data_collectors.fetch_minute_month(chunk, year, month, sleep_time=sleep_time)

                    # Parse and organize by (symbol, day)
                    parsed_data = self.data_collectors.parse_minute_bars_to_daily(symbol_bars, trading_days)

                    # Put parsed data into queue for upload
                    # Track symbols that actually have data to upload
                    symbols_with_data = set()
                    for (sym, day), minute_df in parsed_data.items():
                        if len(minute_df) == 0:
                            with stats_lock:
                                stats['skipped'] += 1
                                stats['completed'] += 1
                            pbar.update(1)
                        else:
                            symbols_with_data.add(sym)
                            data_queue.put((sym, day, minute_df))
                            with stats_lock:
                                stats['completed'] += 1
                            pbar.update(1)

                        pbar.set_postfix(ok=stats['success'], fail=stats['failed'], skip=stats['skipped'], cancel=stats['canceled'])

                    # Only mark symbols as completed if they had actual data
                    for sym in symbols_with_data:
                        tracker.mark_completed(sym)
        except Exception as e:
            self.logger.error(f"Error in {year}-{month:02d}: {e}", exc_info=True)
        finally:
            pbar.close()

    # ===========================
    # Upload fundamental
    # ===========================
    def _process_symbol_fundamental(
            self,
            sym: str,
            start_date: str,
            end_date: str,
            overwrite: bool = False,
            cik: Optional[str] = None
        ) -> dict:
        """
        Process fundamental data for a single symbol for a date range.
        Uses concept-based extraction with approved_mapping.yaml.
        Returns dict with status for progress tracking.

        Storage: data/raw/fundamental/{cik}/fundamental.parquet
        Stored in long format, no forward fill.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param start_date: Start date to fetch data for (YYYY-MM-DD)
        :param end_date: End date to fetch data for (YYYY-MM-DD)
        :param overwrite: If True, skip existence check and overwrite existing data
        :param cik: Pre-fetched CIK (if None, will look up)
        """
        # Use CIKResolver if CIK not provided
        if cik is None:
            reference_year = int(end_date[:4])
            reference_date = f"{reference_year}-06-30"
            cik = self.cik_resolver.get_cik(sym, reference_date, year=reference_year)

        if not overwrite and self.validator.data_exists(sym, 'fundamental', cik=cik):
            self.logger.info(
                f"Fundamental data for {cik} already exists; "
                "continuing to refresh requested date range."
            )

        # Publish fundamental data using concept-based extraction
        return self.data_publishers.publish_fundamental(
            sym=sym,
            start_date=start_date,
            end_date=end_date,
            cik=cik,
            sec_rate_limiter=self.sec_rate_limiter
        )

    def _process_symbol_ttm_fundamental(
            self,
            sym: str,
            start_date: str,
            end_date: str,
            overwrite: bool = False,
            cik: Optional[str] = None
        ) -> dict:
        """
        Process TTM fundamental data for a single symbol for a date range.
        Computes TTM in-memory from long-format fundamentals.

        Storage:
        - data/derived/features/fundamental/{cik}/ttm.parquet (long)

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param start_date: Start date to fetch data for (YYYY-MM-DD)
        :param end_date: End date to fetch data for (YYYY-MM-DD)
        :param overwrite: If True, skip existence check and overwrite existing data
        :param cik: Pre-fetched CIK (if None, will look up)
        """
        if cik is None:
            reference_year = int(end_date[:4])
            reference_date = f"{reference_year}-06-30"
            cik = self.cik_resolver.get_cik(sym, reference_date, year=reference_year)

        if not overwrite and self.validator.data_exists(sym, 'ttm', data_tier='derived', cik=cik):
            self.logger.info(
                f"TTM data for {sym} already exists; "
                "continuing to refresh requested date range."
            )

        return self.data_publishers.publish_ttm_fundamental(
            sym=sym,
            start_date=start_date,
            end_date=end_date,
            cik=cik,
            sec_rate_limiter=self.sec_rate_limiter
        )

    def upload_fundamental(
        self,
        start_date: str,
        end_date: str,
        max_workers: int = 50,
        overwrite: bool = False
    ):
        """
        Upload fundamental data for all symbols for a date range.
        Uses concept-based extraction with approved_mapping.yaml.

        Storage strategy: data/raw/fundamental/{cik}/fundamental.parquet
        - Stores all quarterly/annual filings (long format)
        - No forward filling - only actual filed data
        - One file per CIK
        - Uses all concepts from approved_mapping.yaml

        Performance optimizations:
        1. Batch pre-fetch CIKs to avoid per-symbol database queries
        2. Rate limiting to maximize SEC API throughput (9.5 req/sec)
        3. Increased worker pool (50 workers) - rate limiter controls actual request rate
        4. CIK caching for repeated use

        :param start_date: Start date (YYYY-MM-DD)
        :param end_date: End date (YYYY-MM-DD)
        :param max_workers: Number of concurrent threads (default: 50, rate limited to 9.5 req/sec)
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        start_time = time.time()
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])

        symbol_reference_year = {}
        for year in range(start_year, end_year + 1):
            symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')
            for sym in symbols:
                if sym not in symbol_reference_year:
                    symbol_reference_year[sym] = year

        alpaca_symbols = list(symbol_reference_year.keys())

        total = len(alpaca_symbols)
        self.logger.info(
            f"Starting fundamental upload for {total} symbols "
            f"from {start_date} to {end_date} with {max_workers} workers "
            f"(rate limited to 9.5 req/sec)"
        )
        self.logger.info(f"Using concept-based extraction with approved_mapping.yaml")
        self.logger.info("Storage: data/raw/fundamental/{cik}/fundamental.parquet")

        # OPTIMIZATION: Batch pre-fetch all CIKs before starting (avoids per-symbol DB queries)
        self.logger.info(f"Step 1/3: Pre-fetching CIKs for {total} symbols across {start_year}-{end_year}...")
        prefetch_start = time.time()
        cik_map = {}
        for year in range(start_year, end_year + 1):
            year_symbols = [
                sym for sym, ref_year in symbol_reference_year.items()
                if ref_year == year
            ]
            if not year_symbols:
                continue
            cik_map.update(
                self.cik_resolver.batch_prefetch_ciks(year_symbols, year, batch_size=100)
            )
        prefetch_time = time.time() - prefetch_start
        prefetch_rate = total / prefetch_time if prefetch_time > 0 else 0
        self.logger.info(
            f"CIK pre-fetch completed in {prefetch_time:.1f}s ({prefetch_rate:.1f} symbols/sec)"
        )

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
            self.logger.warning(
                f"No symbols with CIKs found between {start_date} and {end_date}, "
                "skipping fundamental upload"
            )
            return

        # Statistics
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
                    start_date,
                    end_date,
                    overwrite,
                    cik_map.get(sym)  # Pass pre-fetched CIK (guaranteed non-NULL)
                ): sym
                for sym in symbols_with_cik  # Only process symbols with CIKs
            }

            # Process completed tasks
            pbar = tqdm(as_completed(future_to_symbol), total=total, desc="Fundamental", unit="sym")
            for future in pbar:
                result = future.result()

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

                pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)

        # Final statistics
        total_time = time.time() - start_time
        fetch_time = time.time() - fetch_start
        avg_rate = total / fetch_time if fetch_time > 0 else 0

        self.logger.info(
            f"Fundamental upload for {start_date} to {end_date} completed in {total_time:.1f}s: "
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

    def upload_ttm_fundamental(
        self,
        start_date: str,
        end_date: str,
        max_workers: int = 50,
        overwrite: bool = False
    ):
        """
        Upload TTM fundamental data for all symbols for a date range.

        Storage: data/derived/features/fundamental/{cik}/ttm.parquet
        - Computes TTM in memory from long-format fundamentals
        - One file per CIK
        """
        start_time = time.time()
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])

        symbol_reference_year = {}
        for year in range(start_year, end_year + 1):
            symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')
            for sym in symbols:
                if sym not in symbol_reference_year:
                    symbol_reference_year[sym] = year

        alpaca_symbols = list(symbol_reference_year.keys())

        total = len(alpaca_symbols)
        self.logger.info(
            f"Starting TTM fundamental upload for {total} symbols "
            f"from {start_date} to {end_date} with {max_workers} workers "
            f"(rate limited to 9.5 req/sec)"
        )
        self.logger.info("Storage: data/derived/features/fundamental/{cik}/ttm.parquet (long)")

        self.logger.info(f"Step 1/3: Pre-fetching CIKs for {total} symbols...")
        prefetch_start = time.time()
        cik_map = {}
        for year in range(start_year, end_year + 1):
            year_symbols = [
                sym for sym, ref_year in symbol_reference_year.items()
                if ref_year == year
            ]
            if not year_symbols:
                continue
            cik_map.update(
                self.cik_resolver.batch_prefetch_ciks(year_symbols, year, batch_size=100)
            )
        prefetch_time = time.time() - prefetch_start
        prefetch_rate = total / prefetch_time if prefetch_time > 0 else 0
        self.logger.info(
            f"CIK pre-fetch completed in {prefetch_time:.1f}s ({prefetch_rate:.1f} symbols/sec)"
        )

        self.logger.info(f"Step 2/3: Filtering symbols with valid CIKs...")
        symbols_with_cik = [sym for sym in alpaca_symbols if cik_map.get(sym) is not None]
        symbols_without_cik = [sym for sym in alpaca_symbols if cik_map.get(sym) is None]

        self.logger.info(
            f"Symbol filtering complete: {len(symbols_with_cik)}/{total} have CIKs, "
            f"{len(symbols_without_cik)} are non-SEC filers (will be skipped)"
        )

        if len(symbols_without_cik) > 0:
            if len(symbols_without_cik) <= 30:
                self.logger.info(f"Non-SEC filers (skipped): {sorted(symbols_without_cik)}")
            else:
                self.logger.info(
                    f"Non-SEC filers (skipped, showing first 30/{len(symbols_without_cik)}): "
                    f"{sorted(symbols_without_cik)[:30]}"
                )

        total = len(symbols_with_cik)
        if total == 0:
            self.logger.warning(
                f"No symbols with CIKs found between {start_date} and {end_date}, skipping TTM upload"
            )
            return

        success = 0
        failed = 0
        canceled = 0
        skipped = 0

        skipped_symbols = []

        self.logger.info(f"Step 3/3: Computing TTM data for {total} symbols...")
        fetch_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._process_symbol_ttm_fundamental,
                    sym,
                    start_date,
                    end_date,
                    overwrite,
                    cik_map.get(sym)
                ): sym
                for sym in symbols_with_cik
            }

            pbar = tqdm(as_completed(future_to_symbol), total=total, desc="TTM", unit="sym")
            for future in pbar:
                result = future.result()

                if result['status'] == 'success':
                    success += 1
                elif result['status'] == 'canceled':
                    canceled += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                    skipped_symbols.append({
                        'symbol': result.get('symbol'),
                        'cik': result.get('cik'),
                        'error': result.get('error', 'Unknown reason')
                    })
                else:
                    failed += 1

                pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)

        total_time = time.time() - start_time
        fetch_time = time.time() - fetch_start
        avg_rate = total / fetch_time if fetch_time > 0 else 0

        self.logger.info(
            f"TTM upload for {start_date} to {end_date} completed in {total_time:.1f}s: "
            f"{success} success, {failed} failed, {canceled} canceled, {skipped} skipped out of {total} total"
        )
        self.logger.info(
            f"Performance: CIK fetch={prefetch_time:.1f}s, Compute+Upload={fetch_time:.1f}s, "
            f"Avg rate={avg_rate:.2f} sym/sec"
        )

    # ===========================
    # Upload derived fundamental
    # ===========================
    def _process_symbol_derived_fundamental(
            self,
            sym: str,
            start_date: str,
            end_date: str,
            overwrite: bool = False,
            cik: Optional[str] = None
        ) -> dict:
        """
        Process derived fundamental data for a single symbol for a date range.

        Workflow:
        1. Collect TTM fundamental data (long format)
        2. Compute derived metrics
        3. Publish derived data separately

        Storage: data/derived/features/fundamental/{cik}/metrics.parquet
        Contains ONLY derived metrics (keys + 24 derived columns).

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param start_date: Start date to fetch data for (YYYY-MM-DD)
        :param end_date: End date to fetch data for (YYYY-MM-DD)
        :param overwrite: If True, skip existence check and overwrite existing data
        :param cik: Pre-fetched CIK (if None, will look up)
        """
        # Use CIKResolver if CIK not provided
        if cik is None:
            reference_year = int(end_date[:4])
            reference_date = f"{reference_year}-06-30"
            cik = self.cik_resolver.get_cik(sym, reference_date, year=reference_year)

        # Validate: Check if derived data already exists
        if not overwrite and self.validator.data_exists(sym, 'fundamental', data_tier='derived', cik=cik):
            self.logger.info(
                f"Derived metrics for {sym} already exists; "
                "continuing to refresh requested date range."
            )

        if cik is None:
            return {
                'symbol': sym,
                'cik': None,
                'status': 'skipped',
                'error': f'No CIK found for {sym}'
            }

        # Step 1: Collect derived metrics (long format, in-memory)
        self.logger.debug(f"{sym}: Collecting derived metrics for {start_date} to {end_date}")
        derived_df, derived_reason = self.data_collectors.collect_derived_long(
            cik=cik,
            start_date=start_date,
            end_date=end_date,
            symbol=sym
        )

        if len(derived_df) == 0:
            self.logger.debug(f"{sym}: Derived metrics empty for {start_date} to {end_date}")
            return {
                'symbol': sym,
                'cik': cik,
                'status': 'skipped',
                'error': derived_reason or 'Failed to compute derived metrics'
            }

        # Step 3: Publish derived data (separate from raw)
        self.logger.debug(f"{sym}: Publishing derived metrics for {start_date} to {end_date}")
        return self.data_publishers.publish_derived_fundamental(
            sym=sym,
            start_date=start_date,
            end_date=end_date,
            derived_df=derived_df,
            cik=cik
        )

    def upload_derived_fundamental(
        self,
        start_date: str,
        end_date: str,
        max_workers: int = 50,
        overwrite: bool = False
    ):
        """
        Upload derived fundamental data for all symbols for a date range.

        Workflow for each symbol:
        1. Collect TTM fundamental data (long format)
        2. Compute 24 derived metrics
        3. Store derived data separately

        Storage strategy:
        - Raw: data/raw/fundamental/{cik}/fundamental.parquet (long format)
        - TTM: data/derived/features/fundamental/{cik}/ttm.parquet (long format)
        - Derived: data/derived/features/fundamental/{cik}/metrics.parquet (keys + 24 derived)
        One file per CIK for the full requested date range.

        :param start_date: Start date (YYYY-MM-DD)
        :param end_date: End date (YYYY-MM-DD)
        :param max_workers: Number of concurrent threads (default: 50, rate limited to 9.5 req/sec)
        :param overwrite: If True, overwrite existing data in S3 (default: False)
        """
        start_time = time.time()
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])

        symbol_reference_year = {}
        for year in range(start_year, end_year + 1):
            symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')
            if not symbols:
                self.logger.warning(f"No symbols loaded for {year}; skipping year in derived upload")
                continue
            for sym in symbols:
                if sym not in symbol_reference_year:
                    symbol_reference_year[sym] = year

        alpaca_symbols = list(symbol_reference_year.keys())

        total = len(alpaca_symbols)
        self.logger.info(
            f"Starting derived fundamental upload for {total} symbols "
            f"from {start_date} to {end_date} with {max_workers} workers"
        )
        self.logger.info("Storage: data/derived/features/fundamental/{cik}/metrics.parquet")

        # OPTIMIZATION: Batch pre-fetch all CIKs
        self.logger.info(
            f"Step 1/3: Pre-fetching CIKs for {total} symbols across {start_year}-{end_year}..."
        )
        prefetch_start = time.time()
        cik_map = {}
        for year in range(start_year, end_year + 1):
            year_symbols = [
                sym for sym, ref_year in symbol_reference_year.items()
                if ref_year == year
            ]
            if not year_symbols:
                continue
            cik_map.update(
                self.cik_resolver.batch_prefetch_ciks(year_symbols, year, batch_size=100)
            )
        prefetch_time = time.time() - prefetch_start
        self.logger.info(f"CIK pre-fetch completed in {prefetch_time:.1f}s")

        # Filter to only symbols with valid CIKs
        self.logger.info(f"Step 2/3: Filtering symbols with valid CIKs...")
        symbols_with_cik = [sym for sym in alpaca_symbols if cik_map.get(sym) is not None]
        symbols_without_cik = [sym for sym in alpaca_symbols if cik_map.get(sym) is None]

        self.logger.info(
            f"Symbol filtering complete: {len(symbols_with_cik)}/{total} have CIKs, "
            f"{len(symbols_without_cik)} are non-SEC filers (will be skipped)"
        )

        if len(symbols_without_cik) > 0:
            if len(symbols_without_cik) <= 30:
                self.logger.info(f"Non-SEC filers (skipped): {sorted(symbols_without_cik)}")
            else:
                self.logger.info(
                    f"Non-SEC filers (skipped, showing first 30/{len(symbols_without_cik)}): "
                    f"{sorted(symbols_without_cik)[:30]}"
                )

        # Update total
        total = len(symbols_with_cik)
        if total == 0:
            self.logger.warning(
                f"No symbols with CIKs found between {start_date} and {end_date}, skipping derived upload"
            )
            return

        # Statistics
        success = 0
        failed = 0
        canceled = 0
        skipped = 0

        self.logger.info(f"Step 3/3: Computing and uploading derived fundamentals for {total} symbols...")
        fetch_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with pre-fetched CIKs
            future_to_symbol = {
                executor.submit(
                    self._process_symbol_derived_fundamental,
                    sym,
                    start_date,
                    end_date,
                    overwrite,
                    cik_map.get(sym)
                ): sym
                for sym in symbols_with_cik
            }

            # Process completed tasks
            pbar = tqdm(as_completed(future_to_symbol), total=total, desc="Derived", unit="sym")
            for future in pbar:
                result = future.result()

                if result['status'] == 'success':
                    success += 1
                elif result['status'] == 'canceled':
                    canceled += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                else:
                    failed += 1

                pbar.set_postfix(ok=success, fail=failed, skip=skipped, cancel=canceled)

        # Final statistics
        total_time = time.time() - start_time
        fetch_time = time.time() - fetch_start
        avg_rate = total / fetch_time if fetch_time > 0 else 0

        self.logger.info(
            f"Derived fundamental upload for {start_date} to {end_date} completed in {total_time:.1f}s: "
            f"{success} success, {failed} failed, {canceled} canceled, {skipped} skipped out of {total} total"
        )
        self.logger.info(
            f"Performance: CIK fetch={prefetch_time:.1f}s, Compute+Upload={fetch_time:.1f}s, "
            f"Avg rate={avg_rate:.2f} sym/sec"
        )

    def run(
            self,
            start_year: int,
            end_year: int,
            max_workers: int=50,
            overwrite: bool=False,
            resume: bool=False,
            chunk_size: int=500,
            sleep_time: float=0.0,
            daily_chunk_size: int=200,
            daily_sleep_time: float=0.2,
            minute_ticks_start_year: int=2017,
            run_fundamental: bool=False,
            run_derived_fundamental: bool=False,
            run_ttm_fundamental: bool=False,
            run_daily_ticks: bool=False,
            run_minute_ticks: bool=False,
            run_top_3000: bool=False,
            run_all: bool=False
        ) -> None:
        """
        Run the complete workflow, fetch and upload fundamental, daily ticks and minute ticks data within the period

        Storage strategy:
        - Raw Fundamental: Once per CIK -> data/raw/fundamental/{cik}/fundamental.parquet
        - Derived Fundamental: Once per CIK -> data/derived/features/fundamental/{cik}/metrics.parquet
        - TTM Fundamental: Once per CIK -> data/derived/features/fundamental/{cik}/ttm.parquet (long)
        - Daily ticks: Monthly partitions -> data/raw/ticks/daily/{security_id}/{YYYY}/{MM}/ticks.parquet
        - Minute ticks: Daily partitions -> data/raw/ticks/minute/{security_id}/{YYYY}/{MM}/{DD}/ticks.parquet

        :param start_year: Starting year (inclusive)
        :param end_year: Ending year (inclusive)
        :param max_workers: Number of concurrent workers
        :param overwrite: If True, overwrite existing data
        :param resume: If True, skip already-completed symbols (for minute ticks)
        :param chunk_size: Number of symbols to fetch at once for minute data
        :param sleep_time: Sleep time between API requests
        :param run_fundamental: If True, upload raw fundamental data
        :param run_derived_fundamental: If True, upload derived fundamental data
        :param run_ttm_fundamental: If True, upload TTM fundamental data
        :param run_daily_ticks: If True, upload daily ticks data
        :param run_minute_ticks: If True, upload minute ticks data (all months for each year)
        :param run_top_3000: If True, upload the 3000 most liquid stock list
        """
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        if run_all:
            run_fundamental = True
            run_derived_fundamental = True
            run_ttm_fundamental = True
            run_daily_ticks = True
            run_minute_ticks = True
            run_top_3000 = True

        # Upload daily ticks (CRSP bulk for historical, year-by-year for current)
        if run_daily_ticks:
            alpaca_start = self.data_collectors.ticks_collector.alpaca_start_year
            crsp_end = min(end_year, alpaca_start - 1)

            # CRSP years: Use bulk history upload (permno-centric)
            if start_year < alpaca_start and self._wrds_available:
                self.logger.info(
                    f"Uploading CRSP daily ticks ({start_year}-{crsp_end}) "
                    "using permno-centric bulk fetch"
                )
                self._upload_crsp_bulk_history(
                    start_year=start_year,
                    end_year=crsp_end,
                    overwrite=overwrite,
                    chunk_size=500,
                    resume=True
                )

            # Alpaca years: Use existing year-by-year approach
            today = dt.date.today()
            alpaca_end = min(end_year, today.year)
            for year in range(max(start_year, alpaca_start), alpaca_end + 1):
                self.logger.info(f"Uploading Alpaca daily ticks for {year}")
                self.upload_daily_ticks(
                    year,
                    overwrite,
                    by_year=True,
                    chunk_size=daily_chunk_size,
                    sleep_time=daily_sleep_time
                )

        for year in range(start_year, end_year + 1):
            self.logger.info(f"Processing year {year}")

            if run_minute_ticks:
                # Skip future years for minute ticks
                today = dt.date.today()
                if year > today.year:
                    self.logger.info(f"Skipping minute ticks for {year}: future year")
                    continue

                # Upload minute ticks for all months in the year (only for years >= 2017)
                if year >= minute_ticks_start_year:
                    self.logger.info(f"Uploading minute ticks for {year} (all 12 months)")
                    self.upload_minute_ticks_year(year, overwrite=overwrite, resume=resume, num_workers=max_workers, chunk_size=chunk_size, sleep_time=sleep_time)
                else:
                    self.logger.info(f"Skipping minute ticks for {year} (data only available from 2017+)")

            if run_top_3000:
                self.upload_top_3000_monthly(year, overwrite=overwrite)

        # Upload fundamental data LAST (after all ticks are processed)
        if run_fundamental:
            self.logger.info(
                f"Uploading raw fundamental data for {start_date} to {end_date} "
                "(filing-year, long format)"
            )
            self.upload_fundamental(start_date, end_date, max_workers, overwrite)

        if run_ttm_fundamental:
            self.logger.info(
                f"Uploading TTM fundamental data for {start_date} to {end_date}"
            )
            self.upload_ttm_fundamental(start_date, end_date, max_workers, overwrite)

        if run_derived_fundamental:
            self.logger.info(
                f"Uploading derived fundamental data for {start_date} to {end_date}"
            )
            self.upload_derived_fundamental(start_date, end_date, max_workers, overwrite)

    def upload_top_3000_monthly(
        self,
        year: int,
        overwrite: bool = False,
        auto_resolve: bool = True
    ) -> None:
        """
        Upload top 3000 stocks for each month in a given year.

        Storage: data/symbols/{YYYY}/{MM}/top3000.txt
        :param auto_resolve: If True, resolve symbol changes when using CRSP
        """
        symbols = self.universe_manager.load_symbols_for_year(year, sym_type='alpaca')
        if not symbols:
            self.logger.warning(f"No symbols available for {year}, skipping top3000 upload")
            return

        alpaca_start_year = self.data_collectors.ticks_collector.alpaca_start_year
        source = 'crsp' if year < alpaca_start_year else 'alpaca'
        self.logger.info(
            f"Starting {year} top3000 monthly upload for {len(symbols)} symbols (source={source}, overwrite={overwrite})"
        )

        for month in range(1, 13):
            if not overwrite and self.validator.top_3000_exists(year, month):
                self.logger.info(f"Skipping {year}-{month:02d}: top3000 already exists")
                continue

            trading_days = self.calendar.load_trading_days(year, month)
            if not trading_days:
                self.logger.warning(f"No trading days found for {year}-{month:02d}, skipping")
                continue

            as_of = trading_days[-1]
            as_of_date = dt.datetime.strptime(as_of, "%Y-%m-%d").date()
            today = dt.date.today()
            if as_of_date > today and (as_of_date.year, as_of_date.month) > (today.year, today.month):
                self.logger.info(
                    f"Stopping at {year}-{month:02d}: as_of {as_of} is in a future month"
                )
                break

            top_3000 = self.universe_manager.get_top_3000(
                as_of,
                symbols,
                source,
                auto_resolve=auto_resolve
            )

            result = self.data_publishers.publish_top_3000(
                year=year,
                month=month,
                as_of=as_of,
                symbols=top_3000,
                source=source
            )

            if result['status'] == 'success':
                self.logger.info(
                    f"Uploaded top3000 for {year}-{month:02d} (as_of={as_of}, count={len(top_3000)})"
                )
            elif result['status'] == 'skipped':
                self.logger.warning(
                    f"Skipped top3000 for {year}-{month:02d}: {result.get('error')}"
                )
            else:
                self.logger.error(
                    f"Failed top3000 for {year}-{month:02d}: {result.get('error')}"
                )

    def close(self):
        """Close WRDS database connections"""
        if hasattr(self, 'universe_manager') and self.universe_manager is not None:
            self.universe_manager.close()
        if hasattr(self, 'crsp_ticks') and self.crsp_ticks is not None:
            if hasattr(self.crsp_ticks, 'conn') and self.crsp_ticks.conn is not None:
                self.crsp_ticks.conn.close()
                self.logger.info("WRDS connection closed")


