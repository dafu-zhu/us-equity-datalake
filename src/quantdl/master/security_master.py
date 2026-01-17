import wrds
import pandas as pd
import polars as pl
from dotenv import load_dotenv
import os
import time
import requests
import datetime as dt
from typing import List, Tuple, Optional, Dict, Set, Any
from pathlib import Path
import logging
import io
import pyarrow.parquet as pq

from quantdl.utils.logger import setup_logger
from quantdl.universe.current import fetch_all_stocks
from quantdl.utils.wrds import raw_sql_with_retry
from quantdl.storage.rate_limiter import RateLimiter

load_dotenv()

# OpenFIGI rate limits: 25 req/min (no key) or 250 req/min (with key)
OPENFIGI_RATE_LIMIT_NO_KEY = 25 / 60  # ~0.42 req/sec
OPENFIGI_RATE_LIMIT_WITH_KEY = 250 / 60  # ~4.2 req/sec
OPENFIGI_BATCH_SIZE = 25  # Max tickers per request (larger batches cause 413 errors)
OPENFIGI_MIN_BATCH_SIZE = 5  # Minimum batch size before giving up
OPENFIGI_MAX_RETRIES = 3  # Max retries per batch on transient errors


class SymbolNormalizer:
    """
    Deterministic symbol normalizer based on current Nasdaq stock list with SecurityMaster validation.

    Strategy:
    1. Load current stock list from Nasdaq (via fetch_all_stocks)
    2. For any incoming symbol with date context, verify security_id matches
    3. Prevents false matches (e.g., delisted ABCD ≠ current ABC.D)
    4. If verified same security: return Nasdaq format
    5. If different security or delisted: return original format

    Edge case handling:
        - ABCD (2021-2023, delisted, security_id=1000)
        - ABC.D (2025+, active, security_id=2000)
        - Both normalize to "ABCD" but different security_id
        - Solution: Keep historical ABCD as-is, don't convert to ABC.D
    
    Note:
        Nasdaq only covers currectly active stocks. If a stock is delisted, keep it in CRSP format as is. The symbol is for naming the storage folder. This is because the delisted stocks won't be updated, therefore they won't need to match the Nasdaq list.

    Examples:
        to_nasdaq_format('BRKB', '2024-01-01') -> 'BRK.B' (same security)
        to_nasdaq_format('ABCD', '2022-01-01') -> 'ABCD' (delisted, different from ABC.D)
    """

    # CRSP data coverage end date
    CRSP_LATEST_DATE = '2024-12-31'

    def __init__(self, security_master: Optional['SecurityMaster'] = None):
        """
        Initialize with current Nasdaq stock list and optional SecurityMaster.

        :param security_master: SecurityMaster instance for validation (optional)
        """
        # Load current stock list (cached in data/symbols/stock_exchange.csv)
        self.current_stocks_df = fetch_all_stocks(with_filter=True, refresh=False)

        # Create normalized lookup: {crsp_format: nasdaq_format}
        # e.g., {'BRKB': 'BRK.B', 'AAPL': 'AAPL', 'GOOGL': 'GOOGL'}
        self.sym_map = {}
        for ticker in self.current_stocks_df['Ticker']:
            # Skip NaN or non-string values
            if not isinstance(ticker, str):
                continue
            # Remove separators for lookup key
            crsp_key = ticker.replace('.', '').replace('-', '').upper()
            self.sym_map[crsp_key] = ticker

        self.security_master = security_master
        self.logger = setup_logger(
            name="master.SecurityNormalizer",
            log_dir=Path("data/logs/master"),
            level=logging.INFO
        )

    def to_nasdaq_format(self, symbol: str, day: Optional[str] = None) -> str:
        """
        Normalize symbol to Nasdaq format with security_id validation.

        :param symbol: Ticker symbol in any format (BRKB, BRK.B, BRK-B)
        :param day: Date context for validation (format: 'YYYY-MM-DD', optional)
        :return: Nasdaq format if same security, otherwise original

        Examples:
            to_nasdaq_format('BRKB', '2024-01-01') -> 'BRK.B' (verified same security)
            to_nasdaq_format('BRKB') -> 'BRK.B' (no validation, assume same)
            to_nasdaq_format('ABCD', '2022-01-01') -> 'ABCD' (different security, keep original)
        """
        if not symbol:
            return symbol

        # Remove separators
        crsp_key = symbol.replace('.', '').replace('-', '').upper()

        # Check if exists in current stock list
        if crsp_key not in self.sym_map:
            # Not in current list (delisted), return as-is
            return symbol.upper()

        nasdaq_format = self.sym_map[crsp_key]

        # If no date context or no SecurityMaster, return Nasdaq format (assume same security)
        if day is None or self.security_master is None:
            return nasdaq_format

        # Validate using SecurityMaster: check if same security
        try:
            # Get security_id for original symbol at given date
            original_sid = self.security_master.get_security_id(
                symbol=crsp_key,  # Use CRSP format for lookup
                day=day,
                auto_resolve=False  # Strict match only
            )

            # Get security_id for Nasdaq format at CRSP latest date
            nasdaq_sid = self.security_master.get_security_id(
                symbol=crsp_key,  # Use CRSP format for lookup
                day=self.CRSP_LATEST_DATE,
                auto_resolve=False
            )

            # If same security_id, safe to convert to Nasdaq format
            if original_sid == nasdaq_sid:
                return nasdaq_format
            else:
                # Different securities, keep original format
                return symbol.upper()

        except ValueError:
            self.logger.error(f"Symbol {symbol} not found in SecurityMaster at one of the dates, keep original")
            return symbol.upper()

    def batch_normalize(
        self,
        symbols: List[str],
        day: Optional[str] = None
    ) -> List[str]:
        """
        Normalize a batch of symbols with optional date validation.

        :param symbols: List of ticker symbols in any format
        :param day: Date context for validation (format: 'YYYY-MM-DD', optional)
        :return: List of normalized symbols (Nasdaq format if verified)
        """
        return [self.to_nasdaq_format(sym, day) for sym in symbols]

    @staticmethod
    def to_crsp_format(symbol: str) -> str:
        """
        Convert any format to CRSP format (remove separators).

        :param symbol: Ticker in any format (e.g., BRK.B, BRK-B, BRKB)
        :return: CRSP format (e.g., BRKB)
        """
        return symbol.replace('.', '').replace('-', '').upper()

    @staticmethod
    def to_sec_format(symbol: str) -> str:
        """
        Convert Nasdaq format to SEC format (period -> hyphen).

        :param symbol: Ticker in Nasdaq format (e.g., BRK.B)
        :return: SEC format (e.g., BRK-B)
        """
        return symbol.replace('.', '-').upper()


class SecurityMaster:
    """
    Map stock symbols, CIKs, CUSIPs across time horizon using WRDS
    """
    # CRSP data coverage end date
    CRSP_LATEST_DATE = '2024-12-31'

    def __init__(
        self,
        db: Optional[wrds.Connection] = None,
        s3_client: Optional[Any] = None,
        bucket_name: str = 'us-equity-datalake',
        s3_key: str = 'data/master/security_master.parquet',
        force_rebuild: bool = False
    ):
        """
        Initialize SecurityMaster with lazy S3 loading.

        Loading sequence:
        1. If force_rebuild: Build from WRDS
        2. Try S3 (fast path)
        3. If S3 fails: Build from WRDS (slow path)
        4. If built from WRDS: Auto-export to S3

        :param db: Optional WRDS connection
        :param s3_client: Optional S3 client for lazy loading
        :param bucket_name: S3 bucket name
        :param s3_key: S3 key for security master
        :param force_rebuild: If True, skip S3 and rebuild from WRDS
        """
        # Setup logger first (needed for all paths)
        self.logger = setup_logger(
            name="master.SecurityMaster",
            log_dir=Path("data/logs/master"),
            level=logging.INFO
        )

        # Cache for SEC CIK mapping (loaded on-demand)
        self._sec_cik_cache: Optional[pl.DataFrame] = None
        self._from_s3 = False

        # Try S3 lazy loading (FAST PATH)
        if not force_rebuild and s3_client:
            try:
                self.master_tb, metadata = self._load_from_s3(s3_client, bucket_name, s3_key)

                # Validate schema - must have permno column
                if 'permno' not in self.master_tb.columns:
                    self.logger.warning("S3 data missing 'permno' column, rebuilding from WRDS")
                    raise ValueError("Schema mismatch: missing permno")

                # Validate metadata
                crsp_end = metadata.get('crsp_end_date')
                if crsp_end != self.CRSP_LATEST_DATE:
                    self.logger.warning(
                        f"S3 data stale: crsp_end_date={crsp_end} vs expected={self.CRSP_LATEST_DATE}, rebuilding from WRDS"
                    )
                    raise ValueError("Stale S3 data")

                self._from_s3 = True
                self.logger.info(f"Loaded SecurityMaster from S3 ({len(self.master_tb)} rows, CRSP end: {crsp_end})")

                # Setup minimal WRDS connection for close() compatibility
                # (won't be used for queries if loaded from S3)
                self.db = db
                self.cik_cusip = None  # Not needed when loaded from S3
                return

            except Exception as e:
                self.logger.info(f"S3 load failed ({type(e).__name__}: {e}), falling back to WRDS")

        # Build from WRDS (SLOW PATH)
        if db is None:
            username = os.getenv('WRDS_USERNAME')
            password = os.getenv('WRDS_PASSWORD')
            if not username or not password:
                raise ValueError(
                    "WRDS credentials not found. Set WRDS_USERNAME and WRDS_PASSWORD environment variables. "
                    "Alternatively, export SecurityMaster to S3 first using: "
                    "uv run quantdl-export-security-master --export"
                )
            self.db = wrds.Connection(
                wrds_username=username,
                wrds_password=password
            )
        else:
            self.db = db

        self.cik_cusip = self.cik_cusip_mapping()
        self.master_tb = self.master_table()
        self._from_s3 = False

        # Auto-export to S3 if client provided
        if s3_client:
            try:
                self.export_to_s3(s3_client, bucket_name, s3_key)
                self.logger.info("Auto-exported SecurityMaster to S3 for next time")
            except Exception as e:
                self.logger.warning(f"Auto-export to S3 failed: {e}")
    
    def _fetch_sec_cik_mapping(self) -> pl.DataFrame:
        """
        Fetch SEC's official CIK-Ticker mapping as fallback for WRDS NULLs.

        Returns DataFrame with columns: [ticker, cik]
        Note: This is a snapshot mapping (current tickers only), not historical.

        Caches result to avoid repeated API calls.
        """
        if self._sec_cik_cache is not None:
            return self._sec_cik_cache

        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {'User-Agent': 'name@example.com'}  # SEC requires User-Agent

            self.logger.info("Fetching SEC official CIK-Ticker mapping...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse JSON structure: {0: {cik_str, ticker, title}, 1: {...}, ...}
            records = []
            for entry in data.values():
                # Normalize ticker to CRSP format (remove separators)
                ticker = str(entry.get('ticker', '')).replace('.', '').replace('-', '').upper()
                cik = str(entry.get('cik_str', '')).zfill(10)  # Zero-pad to 10 digits

                if ticker and cik != '0000000000':
                    records.append({'ticker': ticker, 'cik': cik})

            # Create DataFrame
            sec_df = pl.DataFrame(records)

            self.logger.info(f"Loaded {len(sec_df)} CIK mappings from SEC")
            self._sec_cik_cache = sec_df
            return sec_df

        except Exception as e:
            self.logger.error(f"Failed to fetch SEC CIK mapping: {e}", exc_info=True)
            # Return empty DataFrame on failure
            return pl.DataFrame({'ticker': [], 'cik': []}, schema={'ticker': pl.Utf8, 'cik': pl.Utf8})

    def _fetch_sec_mapping_full(self) -> pl.DataFrame:
        """
        Fetch SEC mapping with company title for SecurityMaster updates.

        Returns DataFrame with columns: [ticker, cik, title]
        - ticker: CRSP format (separators removed)
        - cik: Zero-padded 10-digit string
        - title: Company name
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {'User-Agent': os.getenv('SEC_USER_AGENT', 'name@example.com')}

        self.logger.info("Fetching SEC company tickers for SecurityMaster update...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()

        records = []
        for entry in data.values():
            # Normalize ticker to CRSP format (remove separators)
            ticker = str(entry.get('ticker', '')).replace('.', '').replace('-', '').upper()
            cik = str(entry.get('cik_str', '')).zfill(10)
            title = str(entry.get('title', ''))

            if ticker and cik != '0000000000':
                records.append({'ticker': ticker, 'cik': cik, 'title': title})

        self.logger.info(f"Loaded {len(records)} tickers from SEC")
        return pl.DataFrame(records)

    def cik_cusip_mapping(self) -> pl.DataFrame:
        """
        All historical mappings, updated until Dec 31, 2024

        Strategy:
        1. Use WRDS CIK mapping as primary source (historical, accurate when present)
        2. For NULL CIKs, fallback to SEC's official CIK-Ticker mapping (current snapshot)
        3. Keep NULL if both sources fail (non-SEC filers)

        Schema: (permno, symbol, company, cik, cusip, start_date, end_date)

        Note: CIK may be NULL for:
        - Foreign companies (use Form 6-K, not 10-K)
        - Small companies (below $10M threshold)
        - OTC/Pink Sheet stocks
        - Non-operating entities (shells, SPACs)
        """
        query = """
        SELECT DISTINCT
            a.kypermno,
            a.ticker,
            a.tsymbol,
            a.comnam,
            a.ncusip,
            b.cik,
            b.cikdate1,
            b.cikdate2,
            a.namedt,
            a.nameenddt
        FROM
            crsp.s6z_nam AS a
        LEFT JOIN
            wrdssec_common.wciklink_cusip AS b
            ON SUBSTR(a.ncusip, 1, 8) = SUBSTR(b.cusip, 1, 8)
            AND (b.cik IS NULL OR a.namedt <= b.cikdate2)
            AND (b.cik IS NULL OR a.nameenddt >= b.cikdate1)
        WHERE
            a.shrcd IN (10, 11)
        ORDER BY
            a.kypermno, a.namedt
        """

        # Execute and load into a DataFrame
        self.logger.info("Fetching CIK-CUSIP mapping from WRDS...")
        map_df = raw_sql_with_retry(self.db, query)
        map_df['namedt'] = pd.to_datetime(map_df['namedt'])
        map_df['nameenddt'] = pd.to_datetime(map_df['nameenddt'])
        map_df['cikdate1'] = pd.to_datetime(map_df['cikdate1'])
        map_df['cikdate2'] = pd.to_datetime(map_df['cikdate2'])

        # Calculate CIK validity period (prefer longer validity = more reliable)
        # Use total_seconds() / 86400 to get days (workaround for timedelta.days)
        map_df['cik_validity_days'] = (
            (map_df['cikdate2'] - map_df['cikdate1']).apply(
                lambda x: x.total_seconds() / 86400 if pd.notnull(x) else -1
            )
        )

        # Filter to keep only the most reliable CIK when multiple CIKs exist for same period
        # Strategy: Keep CIK with longest validity period (cikdate2 - cikdate1)
        map_df = map_df.sort_values(
            ['kypermno', 'tsymbol', 'namedt', 'nameenddt', 'cik_validity_days'],
            ascending=[True, True, True, True, False]  # Longest validity first
        )

        # Keep first (most reliable) CIK for each (permno, symbol, namedt, nameenddt)
        map_df = map_df.drop_duplicates(
            subset=['kypermno', 'tsymbol', 'namedt', 'nameenddt'],
            keep='first'
        )

        # Group by unique combinations to get period ranges
        # Note: dropna=False keeps NULL CIKs (will be handled by SEC fallback later)
        result = (
            map_df.groupby(
                ['kypermno', 'cik', 'ticker', 'tsymbol', 'comnam', 'ncusip'],
                dropna=False
            ).agg({
                'namedt': 'min',
                'nameenddt': 'max'
            })
            .reset_index()
            .sort_values(['kypermno', 'namedt'])
            .dropna(subset=['tsymbol'])
        )

        pl_map = pl.DataFrame(result).with_columns(
            pl.col('kypermno').cast(pl.Int32).alias('permno'),
            pl.col('tsymbol').alias('symbol'),
            pl.col('comnam').alias('company'),
            pl.col('ncusip').alias('cusip'),
            pl.col('namedt').cast(pl.Date).alias('start_date'),
            pl.col('nameenddt').cast(pl.Date).alias('end_date')
        ).select(['permno', 'symbol', 'company', 'cik', 'cusip', 'start_date', 'end_date'])

        # Count NULL CIKs before fallback
        null_count_before = pl_map.filter(pl.col('cik').is_null()).height
        total_count = pl_map.height

        if null_count_before > 0:
            self.logger.info(
                f"Found {null_count_before}/{total_count} records with NULL CIK from WRDS "
                f"({null_count_before/total_count*100:.1f}%), attempting SEC fallback..."
            )

            # Fetch SEC CIK mapping
            sec_mapping = self._fetch_sec_cik_mapping()

            if not sec_mapping.is_empty():
                # For records with NULL CIK, try to match against SEC mapping by symbol
                # Note: SEC mapping is current snapshot, so this is best-effort for historical data
                pl_map = pl_map.join(
                    sec_mapping,
                    left_on='symbol',
                    right_on='ticker',
                    how='left',
                    suffix='_sec'
                ).with_columns([
                    # Use WRDS CIK if available, otherwise SEC CIK
                    pl.when(pl.col('cik').is_not_null())
                    .then(pl.col('cik'))
                    .otherwise(pl.col('cik_sec'))
                    .alias('cik')
                ]).drop('cik_sec')

                # Count how many NULLs were filled
                null_count_after = pl_map.filter(pl.col('cik').is_null()).height
                filled = null_count_before - null_count_after

                if filled > 0:
                    self.logger.info(
                        f"SEC fallback filled {filled}/{null_count_before} NULL CIKs "
                        f"({filled/null_count_before*100:.1f}%)"
                    )

                self.logger.info(
                    f"Final result: {null_count_after}/{total_count} records still have NULL CIK "
                    f"({null_count_after/total_count*100:.1f}%) - these are non-SEC filers"
                )

                # Log details of symbols with NULL CIKs
                null_cik_records = pl_map.filter(pl.col('cik').is_null()).select(['symbol', 'company']).unique()
                if not null_cik_records.is_empty():
                    # Group by unique symbol (may have multiple company names due to history)
                    null_symbols_list = null_cik_records['symbol'].unique().to_list()

                    self.logger.info(f"Symbols without CIK ({len(null_symbols_list)} unique): {sorted(null_symbols_list)[:50]}")

                    if len(null_symbols_list) > 50:
                        self.logger.info(f"... and {len(null_symbols_list) - 50} more (see detailed log below)")

                    # Log detailed company information (first 20 examples)
                    self.logger.info("Examples of non-SEC filers with company names:")
                    for row in null_cik_records.head(20).iter_rows(named=True):
                        self.logger.info(f"  {row['symbol']:10} - {row['company']}")

                    if len(null_cik_records) > 20:
                        self.logger.info(f"  ... and {len(null_cik_records) - 20} more records")
            else:
                self.logger.warning("SEC fallback unavailable, keeping WRDS CIKs only")

                # Still log NULL symbols even if SEC fallback failed
                null_cik_records = pl_map.filter(pl.col('cik').is_null()).select(['symbol', 'company']).unique()
                if not null_cik_records.is_empty():
                    null_symbols_list = null_cik_records['symbol'].unique().to_list()
                    self.logger.warning(f"Symbols without CIK ({len(null_symbols_list)} unique): {sorted(null_symbols_list)[:30]}")
        else:
            self.logger.info("All records have CIK from WRDS, no fallback needed")

        return pl_map
    
    def security_map(self) -> pl.DataFrame:
        """
        Maps security_id based on BUSINESS continuity.

        Rules:
        1. If PERMNO changes → new security_id
        2. If PERMNO stays same:
           - If BOTH symbol AND CIK change (checking against adjacent period) → new security_id
           - Otherwise (only one or neither changes) → same security_id

        Note: Handles overlapping CIK periods by grouping records first.

        Schema: (security_id, permno, symbol, cik, start_date, end_date)
        """
        assert self.cik_cusip is not None, "cik_cusip not initialized (loaded from S3?)"

        # Step 1: Group by (permno, symbol) to collect ALL CIKs for each symbol period
        # This handles the case where the same symbol has multiple overlapping CIK records
        period_groups = (
            self.cik_cusip
            .group_by(['permno', 'symbol'])
            .agg([
                pl.col('cik').unique().alias('ciks'),  # ALL CIKs for this symbol
                pl.col('start_date').min().alias('start_date'),  # Earliest start
                pl.col('end_date').max().alias('end_date'),  # Latest end
                pl.col('company').first().alias('company'),
                pl.col('cusip').first().alias('cusip')
            ])
            .sort(['permno', 'start_date'])
        )

        # Step 2: Track previous period's data within each PERMNO
        period_groups = period_groups.with_columns([
            pl.col('permno').shift(1).alias('prev_permno'),
            pl.col('symbol').shift(1).alias('prev_symbol'),
            pl.col('ciks').shift(1).alias('prev_ciks'),
            pl.col('end_date').shift(1).alias('prev_end_date'),
        ])

        # Step 3: Determine if this period represents a new business
        # We need to check if ANY CIK from current period overlaps with ANY CIK from previous period
        def has_cik_overlap(row):
            """Check if any CIK from current period exists in previous period's CIKs"""
            if row['prev_ciks'] is None or row['ciks'] is None:
                return False
            curr_ciks = set(row['ciks'])
            prev_ciks = set(row['prev_ciks'])
            return len(curr_ciks & prev_ciks) > 0  # True if intersection is non-empty

        # Convert to pandas temporarily for complex logic
        pdf = period_groups.to_pandas()

        # Check CIK overlap
        pdf['cik_overlap'] = pdf.apply(has_cik_overlap, axis=1)

        # Determine new_business flag
        pdf['new_business'] = (
            pdf['prev_permno'].isna() |  # First row
            (pdf['permno'] != pdf['prev_permno']) |  # PERMNO changed
            (
                (pdf['permno'] == pdf['prev_permno']) &  # PERMNO same
                (pdf['symbol'] != pdf['prev_symbol']) &  # Symbol changed
                (~pdf['cik_overlap'])  # No CIK overlap (all CIKs different)
            )
        )

        # Assign security_ids
        pdf['security_id'] = (pdf['new_business'].cumsum() + 1000)

        # Step 4: Join security_id back to original cik_cusip data
        # This preserves the original start_date and end_date for each row
        security_assignments = pl.from_pandas(
            pdf[['permno', 'symbol', 'security_id']]
        )

        # Join back to original data based on (permno, symbol)
        result = self.cik_cusip.join(
            security_assignments,
            on=['permno', 'symbol'],
            how='left'
        ).select([
            'security_id',
            'permno',
            'symbol',
            'company',
            'cik',
            'cusip',
            'start_date',
            'end_date',
        ]).with_columns([
            pl.col('security_id').cast(pl.Int64)
        ])

        # Log some statistics
        n_securities = result['security_id'].n_unique()
        n_permnos = result['permno'].n_unique()
        self.logger.info(f"Created {n_securities} security_ids from {n_permnos} PERMNOs")

        return result

    def master_table(self) -> pl.DataFrame:
        """
        Create comprehensive table with security_id as master key, tracking business continuity.

        Schema: (security_id, permno, symbol, company, cik, cusip, start_date, end_date)
        """
        security_map = self.security_map()

        result = security_map.select([
            'security_id', 'permno', 'symbol', 'company', 'cik', 'cusip', 'start_date', 'end_date'
        ])

        return result
    
    def auto_resolve(self, symbol: str, day: str) -> int:
        """
        Smart resolve unmatched symbol and query day.
        Route to the security that is active on 'day', and have most recently used / use 'symbol' in the future
        """
        date_check = dt.datetime.strptime(day, '%Y-%m-%d').date()

        # Find all securities that ever used this symbol
        candidates = (
            self.master_tb.filter(pl.col('symbol').eq(symbol))
            .select('security_id')
            .unique()
            .filter(pl.col('security_id').is_not_null())
        )

        if candidates.is_empty():
            self.logger.debug(f"auto_resolve failed: symbol '{symbol}' never existed in security master")
            raise ValueError(f"Symbol '{symbol}' never existed in security master")

        # For each candidate, check if it was active on target date (under ANY symbol)
        active_securities = []
        null_candidates = 0
        for candidate_sid in candidates['security_id']:
            if candidate_sid is None:
                null_candidates += 1
                continue
            was_active = self.master_tb.filter(
                pl.col('security_id').eq(candidate_sid),
                pl.col('start_date').le(date_check),
                pl.col('end_date').ge(date_check)
            )
            if not was_active.is_empty():
                # Find when this security used the queried symbol
                symbol_usage = self.master_tb.filter(
                    pl.col('security_id').eq(candidate_sid),
                    pl.col('symbol').eq(symbol)
                ).select(['start_date', 'end_date']).head(1)

                active_securities.append({
                    'sid': candidate_sid,
                    'symbol_start': symbol_usage['start_date'][0],
                    'symbol_end': symbol_usage['end_date'][0]
                })
        if null_candidates > 0:
            self.logger.warning(
                f"auto_resolve: filtered {null_candidates} null security_id values for symbol='{symbol}'"
            )

        # Resolve ambiguity
        if len(active_securities) == 0:
            self.logger.debug(
                    f"auto_resolve failed: symbol '{symbol}' exists but associated security "
                    f"was not active on {day}"
                )
            raise ValueError(
                f"Symbol '{symbol}' exists but the associated security was not active on {day}"
            )
        elif len(active_securities) == 1:
            sid = active_securities[0]['sid']
        else:
            # Multiple securities used this symbol and were active on target date
            # Pick the one that used this symbol closest to the query date
            def distance_to_date(sec):
                """Calculate temporal distance from query date to when symbol was used"""
                if date_check < sec['symbol_start']:
                    return (sec['symbol_start'] - date_check).days
                elif date_check > sec['symbol_end']:
                    return (date_check - sec['symbol_end']).days
                else:
                    return 0

            # Pick security with minimum distance
            best_match = min(active_securities, key=distance_to_date)
            sid = best_match['sid']

            self.logger.info(
                f"auto_resolve: Multiple candidates found, selected security_id={sid} "
            )

        try:
            # Try to fetch the info
            cik = self.sid_to_info(sid, day, info='cik')
            company = self.sid_to_info(sid, day, info='company')
            
            self.logger.info(f"auto_resolve triggered for symbol='{symbol}' ({company}) on date='{day}', sid={sid}, cik={cik}")
        except Exception as e:
            # If it fails, log the specific error so you know WHY it crashed
            self.logger.error(f"auto_resolve triggered for symbol='{symbol}', sid={sid}: {str(e)}")

        return sid

    def get_security_id(self, symbol: str, day: str, auto_resolve: bool=True) -> int:
        """
        Finds the Internal ID for a specific Symbol at a specific point in time.

        :param auto_resolve: If the security has name change, resolve symbol to the nearest security
        """
        date_check = dt.datetime.strptime(day, '%Y-%m-%d').date()
            
        match = self.master_tb.filter(
            pl.col('symbol').eq(symbol),
            pl.col('start_date').le(date_check),
            pl.col('end_date').ge(date_check)
        )
        
        if match.is_empty():
            if not auto_resolve:
                raise ValueError(f"Symbol {symbol} not found in day {day}")
            else:
                return self.auto_resolve(symbol, day)

        result = match.head(1).select('security_id').item()

        # Validate that security_id is not None (data quality check)
        if result is None:
            raise ValueError(
                f"Symbol '{symbol}' found in security master for {day}, but security_id is None. "
                "This indicates corrupted data in the security master table."
            )

        return result
    
    def get_symbol_history(self, sid: int) -> List[Tuple[str, str, str]]:
        """
        Full list of symbol usage history for a given security_id

        Example: [('META', '2022-06-09', '2024-12-31'), ('FB', '2012-05-18', '2022-06-08')]
        """
        mtb = self.master_table()
        sid_df = mtb.filter(
            pl.col('security_id').eq(sid)
        ).group_by('symbol').agg(
            pl.col('start_date').min(),
            pl.col('end_date').max()
        )

        hist = sid_df.select(['symbol', 'start_date', 'end_date']).rows()
        isoformat_hist = [(sym, start.isoformat(), end.isoformat()) for sym, start, end in hist]

        return isoformat_hist
    
    def sid_to_permno(self, sid: Optional[int]) -> int:
        if sid is None:
            raise ValueError("security_id is None")
        permno = (
            self.master_tb.filter(
                pl.col('security_id').eq(sid)
            )
            .select('permno')
            .head(1)
            .item()
        )
        return permno

    def sid_to_info(self, sid: int, day: str, info: str):

        date_obj = dt.datetime.strptime(day, "%Y-%m-%d").date()

        master_tb = self.master_tb
        result = (
            master_tb.filter(
                pl.col('security_id').eq(sid),
                pl.col('start_date').le(date_obj),
                pl.col('end_date').ge(date_obj)
            ).select(info).head(1).item()
        )
        return result

    def export_to_s3(
        self,
        s3_client: Any,
        bucket_name: str = 'us-equity-datalake',
        s3_key: str = 'data/master/security_master.parquet'
    ) -> Dict[str, str]:
        """
        Export master_tb to S3 with embedded metadata.

        Metadata (Parquet custom_metadata):
        - crsp_end_date: 2024-12-31
        - export_timestamp: ISO8601 UTC
        - version: 1.0
        - row_count: Number of rows

        :param s3_client: Boto3 S3 client
        :param bucket_name: S3 bucket name
        :param s3_key: S3 key for export
        :return: Dict with status and timestamp
        """
        # Convert to Arrow table
        table = self.master_tb.to_arrow()

        # Embed metadata
        metadata = {
            b'crsp_end_date': self.CRSP_LATEST_DATE.encode(),
            b'export_timestamp': dt.datetime.utcnow().isoformat().encode(),
            b'version': b'1.0',
            b'row_count': str(len(self.master_tb)).encode()
        }
        existing_meta = table.schema.metadata or {}
        combined_meta = {**existing_meta, **metadata}
        table = table.replace_schema_metadata(combined_meta)

        # Write to buffer and upload
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        s3_client.upload_fileobj(
            buffer,
            bucket_name,
            s3_key
        )

        export_ts = dt.datetime.utcnow().isoformat()
        self.logger.info(f"Exported SecurityMaster to s3://{bucket_name}/{s3_key} ({len(self.master_tb)} rows)")
        return {'status': 'success', 'export_timestamp': export_ts}

    def _load_from_s3(
        self,
        s3_client: Any,
        bucket_name: str,
        s3_key: str
    ) -> Tuple[pl.DataFrame, Dict[str, str]]:
        """
        Load master_tb from S3 with metadata extraction.

        :param s3_client: Boto3 S3 client
        :param bucket_name: S3 bucket name
        :param s3_key: S3 key
        :return: Tuple of (master_tb DataFrame, metadata dict)
        :raises: Exception if load fails
        """
        # Download from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)

        # Read Parquet with metadata
        buffer = io.BytesIO(response['Body'].read())
        table = pq.read_table(buffer)

        # Extract custom metadata
        metadata = {}
        if table.schema.metadata:
            metadata = {
                k.decode(): v.decode()
                for k, v in table.schema.metadata.items()
            }

        # Convert to Polars (from_arrow on Table always returns DataFrame)
        df = pl.from_arrow(table)
        assert isinstance(df, pl.DataFrame)

        self.logger.debug(f"Loaded SecurityMaster from S3: {len(df)} rows, metadata: {metadata}")
        return df, metadata

    def update_from_sec(
        self,
        s3_client: Optional[Any] = None,
        bucket_name: str = 'us-equity-datalake'
    ) -> Dict[str, Any]:
        """
        Update master_tb from SEC company_tickers.json (WRDS-free updates).

        1. For existing securities with stale end_date: extend to today if still in SEC
        2. For new securities in SEC but not in master_tb: add with new security_id

        :param s3_client: Optional S3 client to export updated master_tb
        :param bucket_name: S3 bucket name for export
        :return: Dict with counts {'extended': N, 'added': N, 'unchanged': N} or 'error' on failure
        """
        try:
            sec_df = self._fetch_sec_mapping_full()
        except Exception as e:
            self.logger.error(f"Failed to fetch SEC data: {e}")
            return {'extended': 0, 'added': 0, 'unchanged': 0, 'error': str(e)}

        today = dt.date.today()
        stats = {'extended': 0, 'added': 0, 'unchanged': 0}

        # Create lookup set for SEC securities: (symbol, cik)
        sec_set = set(zip(sec_df['ticker'].to_list(), sec_df['cik'].to_list()))

        # 1. Extend end_date for existing securities still in SEC list
        # Build updated rows list
        updated_rows = []
        for row in self.master_tb.iter_rows(named=True):
            key = (row['symbol'], row['cik'])
            if key in sec_set and row['end_date'] < today:
                # Extend end_date to today
                updated_row = dict(row)
                updated_row['end_date'] = today
                updated_rows.append(updated_row)
                stats['extended'] += 1
            else:
                updated_rows.append(dict(row))
                stats['unchanged'] += 1

        # Rebuild master_tb with updated end_dates
        if stats['extended'] > 0:
            self.master_tb = pl.DataFrame(updated_rows)

        # 2. Add new securities not in master_tb
        existing_keys = set(zip(
            self.master_tb['symbol'].to_list(),
            self.master_tb['cik'].to_list()
        ))
        max_sid: int = self.master_tb['security_id'].max() or 1000  # type: ignore[assignment]

        new_rows = []
        for row in sec_df.iter_rows(named=True):
            key = (row['ticker'], row['cik'])
            if key not in existing_keys:
                max_sid += 1
                new_rows.append({
                    'security_id': max_sid,
                    'symbol': row['ticker'],
                    'company': row['title'],
                    'permno': None,
                    'cik': row['cik'],
                    'cusip': None,
                    'start_date': today,
                    'end_date': today
                })
                stats['added'] += 1

        if new_rows:
            new_df = pl.DataFrame(new_rows).cast({
                'security_id': pl.Int64,
                'start_date': pl.Date,
                'end_date': pl.Date
            })
            self.master_tb = pl.concat([self.master_tb, new_df], how='diagonal')

        # 3. Export to S3 if changes made
        if s3_client and (stats['extended'] > 0 or stats['added'] > 0):
            self.export_to_s3(s3_client, bucket_name)
            self.logger.info(
                f"SecurityMaster updated: {stats['extended']} extended, "
                f"{stats['added']} added, {stats['unchanged']} unchanged"
            )

        return stats

    def _fetch_openfigi_mapping(
        self,
        tickers: List[str],
        rate_limiter: Optional[RateLimiter] = None
    ) -> Dict[str, Optional[str]]:
        """
        Batch lookup ticker → shareClassFIGI via OpenFIGI API.

        API: POST https://api.openfigi.com/v3/mapping
        Rate limit: 25 req/min (no key) or 250 req/min (with key)
        Batch size: 25 tickers per request (reduced on 413 errors)

        Features:
        - Retry with exponential backoff on 429/5xx errors
        - Batch size reduction on 413 (payload too large)
        - Progress logging every 10 batches

        :param tickers: List of ticker symbols
        :param rate_limiter: Optional RateLimiter instance
        :return: Dict mapping ticker → shareClassFIGI (None if not found)
        """
        url = "https://api.openfigi.com/v3/mapping"
        headers = {"Content-Type": "application/json"}

        # Use API key if available
        api_key = os.getenv("OPENFIGI_API_KEY")
        if api_key:
            headers["X-OPENFIGI-APIKEY"] = api_key
            max_rate = OPENFIGI_RATE_LIMIT_WITH_KEY
        else:
            max_rate = OPENFIGI_RATE_LIMIT_NO_KEY

        # Create rate limiter if not provided
        if rate_limiter is None:
            rate_limiter = RateLimiter(max_rate=max_rate)

        results: Dict[str, Optional[str]] = {}
        total_batches = (len(tickers) + OPENFIGI_BATCH_SIZE - 1) // OPENFIGI_BATCH_SIZE
        batches_processed = 0

        self.logger.info(f"Starting OpenFIGI lookup for {len(tickers)} tickers ({total_batches} batches)")

        # Process in batches
        i = 0
        while i < len(tickers):
            batch = tickers[i:i + OPENFIGI_BATCH_SIZE]
            current_batch_size = len(batch)

            # Retry loop with exponential backoff
            success = False
            retry_count = 0
            working_batch = batch

            while not success and retry_count <= OPENFIGI_MAX_RETRIES:
                # Build request payload
                payload = [
                    {"idType": "TICKER", "idValue": t, "exchCode": "US"}
                    for t in working_batch
                ]

                try:
                    rate_limiter.acquire()
                    response = requests.post(url, json=payload, headers=headers, timeout=30)

                    # Handle specific HTTP errors
                    if response.status_code == 413:
                        # Payload too large - reduce batch size
                        new_size = max(len(working_batch) // 2, OPENFIGI_MIN_BATCH_SIZE)
                        if len(working_batch) <= OPENFIGI_MIN_BATCH_SIZE:
                            self.logger.warning(
                                f"413 error at minimum batch size ({OPENFIGI_MIN_BATCH_SIZE}), "
                                f"marking {len(working_batch)} tickers as unmapped"
                            )
                            for t in working_batch:
                                results[t] = None
                            success = True
                            break

                        self.logger.warning(
                            f"413 error, reducing batch size from {len(working_batch)} to {new_size}"
                        )
                        # Process first half now, second half will be picked up in next iteration
                        working_batch = working_batch[:new_size]
                        retry_count += 1
                        continue

                    if response.status_code == 429 or response.status_code >= 500:
                        # Rate limited or server error - exponential backoff
                        wait_time = 2 ** retry_count
                        self.logger.warning(
                            f"HTTP {response.status_code}, retrying in {wait_time}s "
                            f"(attempt {retry_count + 1}/{OPENFIGI_MAX_RETRIES + 1})"
                        )
                        time.sleep(wait_time)
                        retry_count += 1
                        continue

                    response.raise_for_status()
                    data = response.json()

                    # Parse response - each item corresponds to a ticker
                    for j, item in enumerate(data):
                        ticker = working_batch[j]
                        if "data" in item and item["data"]:
                            figi = item["data"][0].get("shareClassFIGI")
                            results[ticker] = figi
                        else:
                            results[ticker] = None

                    success = True

                    # If we reduced batch size, process remaining tickers from original batch
                    if len(working_batch) < current_batch_size:
                        remaining = batch[len(working_batch):]
                        if remaining:
                            # Insert remaining at front of unprocessed tickers
                            tickers = tickers[:i] + remaining + tickers[i + OPENFIGI_BATCH_SIZE:]
                            # Adjust total batches estimate
                            total_batches = (len(tickers) - i + OPENFIGI_BATCH_SIZE - 1) // OPENFIGI_BATCH_SIZE + batches_processed

                except requests.RequestException as e:
                    if retry_count < OPENFIGI_MAX_RETRIES:
                        wait_time = 2 ** retry_count
                        self.logger.warning(
                            f"Request error: {e}, retrying in {wait_time}s "
                            f"(attempt {retry_count + 1}/{OPENFIGI_MAX_RETRIES + 1})"
                        )
                        time.sleep(wait_time)
                        retry_count += 1
                    else:
                        self.logger.warning(
                            f"OpenFIGI batch failed after {OPENFIGI_MAX_RETRIES + 1} attempts: {e}"
                        )
                        for t in working_batch:
                            results[t] = None
                        success = True

                except Exception as e:
                    self.logger.error(f"Unexpected error in OpenFIGI lookup: {e}")
                    for t in working_batch:
                        results[t] = None
                    success = True

            # Mark exhausted retries
            if not success:
                self.logger.warning(
                    f"Batch exhausted retries, marking {len(working_batch)} tickers as unmapped"
                )
                for t in working_batch:
                    results[t] = None

            i += OPENFIGI_BATCH_SIZE
            batches_processed += 1

            # Progress logging every 10 batches
            if batches_processed % 10 == 0 or batches_processed == total_batches:
                pct = (batches_processed / total_batches) * 100
                self.logger.info(f"OpenFIGI progress: {batches_processed}/{total_batches} batches ({pct:.0f}%)")

        found = sum(1 for v in results.values() if v is not None)
        self.logger.info(f"OpenFIGI lookup complete: {found}/{len(results)} tickers mapped to FIGIs")

        return results

    def _fetch_nasdaq_universe(self) -> Set[str]:
        """
        Fetch current active stocks from Nasdaq FTP.

        :return: Set of active ticker symbols (Nasdaq format)
        """
        try:
            df = fetch_all_stocks(with_filter=True, refresh=True, logger=self.logger)
            tickers = set(df['Ticker'].tolist())
            self.logger.info(f"Fetched {len(tickers)} active tickers from Nasdaq")
            return tickers
        except Exception as e:
            self.logger.error(f"Failed to fetch Nasdaq universe: {e}")
            return set()

    def _detect_rebrands(
        self,
        disappeared: Set[str],
        appeared: Set[str],
        figi_mapping: Dict[str, Optional[str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Detect rebrands by matching shareClassFIGI between disappeared and appeared tickers.

        :param disappeared: Tickers that were in prev but not in current
        :param appeared: Tickers that are in current but not in prev
        :param figi_mapping: Dict mapping ticker → shareClassFIGI
        :return: List of (old_ticker, new_ticker, figi) tuples for detected rebrands
        """
        rebrands = []

        # Build reverse lookup: FIGI → disappeared ticker
        figi_to_old: Dict[str, str] = {}
        for ticker in disappeared:
            figi = figi_mapping.get(ticker)
            if figi:
                figi_to_old[figi] = ticker

        # Check if any appeared ticker has same FIGI as a disappeared ticker
        for ticker in appeared:
            figi = figi_mapping.get(ticker)
            if figi and figi in figi_to_old:
                old_ticker = figi_to_old[figi]
                rebrands.append((old_ticker, ticker, figi))
                self.logger.info(f"Detected rebrand: {old_ticker} → {ticker} (FIGI: {figi})")

        return rebrands

    def _load_prev_universe(
        self,
        s3_client: Any,
        bucket_name: str
    ) -> Tuple[Set[str], Optional[str]]:
        """
        Load previous universe from S3 metadata.

        :return: Tuple of (prev_universe set, prev_date string or None)
        """
        s3_key = "data/master/prev_universe.json"

        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            import json
            data = json.loads(response['Body'].read().decode('utf-8'))
            prev_universe = set(data.get('tickers', []))
            prev_date = data.get('date')
            self.logger.info(f"Loaded prev_universe: {len(prev_universe)} tickers from {prev_date}")
            return prev_universe, prev_date
        except s3_client.exceptions.NoSuchKey:
            self.logger.info("No prev_universe found, will bootstrap from current Nasdaq list")
            return set(), None
        except Exception as e:
            self.logger.warning(f"Failed to load prev_universe: {e}, bootstrapping")
            return set(), None

    def _save_prev_universe(
        self,
        s3_client: Any,
        bucket_name: str,
        universe: Set[str],
        date: str
    ) -> None:
        """
        Save current universe to S3 for next run.

        :param universe: Set of ticker symbols
        :param date: Date string (YYYY-MM-DD)
        """
        s3_key = "data/master/prev_universe.json"

        import json
        data = {
            'tickers': sorted(list(universe)),
            'date': date
        }

        body = json.dumps(data).encode('utf-8')
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=body,
            ContentType='application/json'
        )
        self.logger.info(f"Saved prev_universe: {len(universe)} tickers for {date}")

    def update_no_wrds(
        self,
        s3_client: Any,
        bucket_name: str = 'us-equity-datalake',
        grace_period_days: int = 14
    ) -> Dict[str, int]:
        """
        Update master_tb using Nasdaq + OpenFIGI (no WRDS required).

        Algorithm:
        1. EXTEND: Ticker in both prev and current → update end_date to today
        2. REBRAND: Old ticker disappeared, new appeared, same shareClassFIGI
           → Close old row, create new row with SAME security_id
        3. NEW IPO: New ticker with new FIGI → create row with NEW security_id
        4. DELIST: Ticker in prev but not current (for 14+ days) → freeze end_date

        :param s3_client: S3 client for persistence
        :param bucket_name: S3 bucket name
        :param grace_period_days: Days before treating missing ticker as delisted
        :return: Dict with counts {'extended', 'rebranded', 'added', 'delisted', 'unchanged'}
        """
        today = dt.date.today()
        today_str = today.isoformat()

        stats = {
            'extended': 0,
            'rebranded': 0,
            'added': 0,
            'delisted': 0,
            'unchanged': 0
        }

        # 1. Load current Nasdaq universe
        current_nasdaq = self._fetch_nasdaq_universe()
        if not current_nasdaq:
            self.logger.error("Failed to fetch Nasdaq universe, aborting update")
            return stats

        # 2. Load previous universe
        prev_universe, prev_date = self._load_prev_universe(s3_client, bucket_name)

        # Bootstrap: if no prev_universe, use current as baseline
        if not prev_universe:
            self.logger.info("Bootstrapping: using current Nasdaq list as prev_universe")
            self._save_prev_universe(s3_client, bucket_name, current_nasdaq, today_str)
            # Just extend end_dates for existing securities
            return self._extend_existing_securities(current_nasdaq, today, s3_client, bucket_name)

        # 3. Compute changes
        # Normalize to CRSP format for comparison
        def normalize(ticker):
            return ticker.replace('.', '').replace('-', '').upper()

        current_normalized = {normalize(t): t for t in current_nasdaq}
        prev_normalized = {normalize(t): t for t in prev_universe}

        current_set = set(current_normalized.keys())
        prev_set = set(prev_normalized.keys())

        still_active = current_set & prev_set  # In both
        disappeared = prev_set - current_set    # In prev, not in current
        appeared = current_set - prev_set       # In current, not in prev

        self.logger.info(
            f"Universe changes: {len(still_active)} active, "
            f"{len(disappeared)} disappeared, {len(appeared)} appeared"
        )

        # 4. Fetch OpenFIGI mappings for disappeared + appeared tickers
        tickers_to_lookup = list(disappeared | appeared)
        figi_mapping: Dict[str, Optional[str]] = {}
        if tickers_to_lookup:
            # Convert back to original format for lookup
            # t is guaranteed to be in one of the dicts since it came from their keys
            original_tickers: List[str] = [
                prev_normalized.get(t) or current_normalized[t]
                for t in tickers_to_lookup
            ]
            figi_results = self._fetch_openfigi_mapping(original_tickers)
            # Store with normalized keys
            for t in tickers_to_lookup:
                orig = prev_normalized.get(t) or current_normalized[t]
                figi_mapping[t] = figi_results.get(orig)

        # 5. Detect rebrands
        rebrands = self._detect_rebrands(disappeared, appeared, figi_mapping)
        rebrand_old = {r[0] for r in rebrands}
        rebrand_new = {r[1] for r in rebrands}

        # 6. Process updates
        updated_rows = []
        existing_keys = set()  # (symbol_normalized, cik)

        for row in self.master_tb.iter_rows(named=True):
            row_dict = dict(row)
            symbol_norm = normalize(row['symbol'])
            existing_keys.add((symbol_norm, row['cik']))

            if symbol_norm in still_active:
                # EXTEND: still active, update end_date
                row_dict['end_date'] = today
                stats['extended'] += 1

            elif symbol_norm in rebrand_old:
                # REBRAND (old ticker): close this row
                # Find the rebrand tuple
                for old, new, figi in rebrands:
                    if old == symbol_norm:
                        # Don't extend end_date (freeze it)
                        # The new ticker row will be added separately
                        break
                stats['rebranded'] += 1

            elif symbol_norm in disappeared:
                # DELIST: check grace period
                if prev_date:
                    prev_dt = dt.datetime.strptime(prev_date, '%Y-%m-%d').date()
                    days_missing = (today - prev_dt).days
                    if days_missing < grace_period_days:
                        # Still in grace period, extend end_date
                        row_dict['end_date'] = today
                        stats['extended'] += 1
                    else:
                        # Grace period passed, mark as delisted (freeze end_date)
                        stats['delisted'] += 1
                else:
                    # No prev_date, can't determine grace period
                    stats['unchanged'] += 1
            else:
                stats['unchanged'] += 1

            updated_rows.append(row_dict)

        # 7. Add rebrand new rows (same security_id as old)
        for old_norm, new_norm, figi in rebrands:
            # Find old row's security_id
            old_row = self.master_tb.filter(
                pl.col('symbol').str.replace_all(r'[.\-]', '').str.to_uppercase() == old_norm
            ).head(1)

            if old_row.is_empty():
                self.logger.warning(f"Rebrand old ticker {old_norm} not found in master_tb")
                continue

            old_security_id = old_row['security_id'][0]
            new_ticker = current_normalized[new_norm]

            # Create new row with same security_id
            new_row = {
                'security_id': old_security_id,
                'permno': old_row['permno'][0] if 'permno' in old_row.columns else None,
                'symbol': new_ticker,
                'company': old_row['company'][0] if 'company' in old_row.columns else '',
                'cik': old_row['cik'][0] if 'cik' in old_row.columns else None,
                'cusip': old_row['cusip'][0] if 'cusip' in old_row.columns else None,
                'share_class_figi': figi,
                'start_date': today,
                'end_date': today
            }
            updated_rows.append(new_row)

        # 8. Add truly new IPOs (new FIGI, not a rebrand)
        max_sid: int = self.master_tb['security_id'].max() or 1000  # type: ignore[assignment]
        new_ipos = appeared - rebrand_new

        for ticker_norm in new_ipos:
            ticker = current_normalized[ticker_norm]
            figi = figi_mapping.get(ticker_norm)

            max_sid += 1
            new_row = {
                'security_id': max_sid,
                'permno': None,
                'symbol': ticker,
                'company': '',
                'cik': None,
                'cusip': None,
                'share_class_figi': figi,
                'start_date': today,
                'end_date': today
            }
            updated_rows.append(new_row)
            stats['added'] += 1

        # 9. Rebuild master_tb
        if updated_rows:
            self.master_tb = pl.DataFrame(updated_rows).cast({
                'security_id': pl.Int64,
                'start_date': pl.Date,
                'end_date': pl.Date
            })

            # Ensure share_class_figi column exists
            if 'share_class_figi' not in self.master_tb.columns:
                self.master_tb = self.master_tb.with_columns(
                    pl.lit(None).cast(pl.Utf8).alias('share_class_figi')
                )

        # 10. Save updated prev_universe
        self._save_prev_universe(s3_client, bucket_name, current_nasdaq, today_str)

        # 11. Export to S3
        changes_made = stats['extended'] + stats['rebranded'] + stats['added'] + stats['delisted']
        if changes_made > 0:
            self.export_to_s3(s3_client, bucket_name)
            self.logger.info(
                f"SecurityMaster updated (no WRDS): "
                f"{stats['extended']} extended, {stats['rebranded']} rebranded, "
                f"{stats['added']} new IPOs, {stats['delisted']} delisted, "
                f"{stats['unchanged']} unchanged"
            )

        return stats

    def _extend_existing_securities(
        self,
        current_nasdaq: Set[str],
        today: dt.date,
        s3_client: Any,
        bucket_name: str
    ) -> Dict[str, int]:
        """
        Helper to extend end_dates for securities in current Nasdaq list.
        Used during bootstrap when no prev_universe exists.
        """
        stats = {'extended': 0, 'rebranded': 0, 'added': 0, 'delisted': 0, 'unchanged': 0}

        def normalize(ticker):
            return ticker.replace('.', '').replace('-', '').upper()

        current_normalized = {normalize(t) for t in current_nasdaq}

        updated_rows = []
        for row in self.master_tb.iter_rows(named=True):
            row_dict = dict(row)
            symbol_norm = normalize(row['symbol'])

            if symbol_norm in current_normalized:
                row_dict['end_date'] = today
                stats['extended'] += 1
            else:
                stats['unchanged'] += 1

            updated_rows.append(row_dict)

        if updated_rows:
            self.master_tb = pl.DataFrame(updated_rows).cast({
                'security_id': pl.Int64,
                'start_date': pl.Date,
                'end_date': pl.Date
            })

        if stats['extended'] > 0:
            self.export_to_s3(s3_client, bucket_name)
            self.logger.info(f"Bootstrap: extended {stats['extended']} securities to {today}")

        return stats

    def overwrite_from_crsp(
        self,
        db: Any,
        s3_client: Any,
        bucket_name: str = 'us-equity-datalake'
    ) -> Dict[str, int]:
        """
        Rebuild master_tb from CRSP data with shareClassFIGI (one-time overwrite).

        1. Fetch fresh data from WRDS CRSP
        2. Add shareClassFIGI column via OpenFIGI
        3. Export to S3, replacing existing master_tb

        :param db: WRDS database connection
        :param s3_client: S3 client
        :param bucket_name: S3 bucket name
        :return: Dict with stats {'rows', 'figi_mapped'}
        """
        self.logger.info("Rebuilding SecurityMaster from CRSP with OpenFIGI...")

        # Store old db connection
        old_db = self.db
        self.db = db

        # Rebuild from CRSP
        self.cik_cusip = self.cik_cusip_mapping()
        self.master_tb = self.master_table()

        # Restore db
        self.db = old_db

        # Get unique symbols
        symbols = self.master_tb['symbol'].unique().to_list()
        self.logger.info(f"Fetching OpenFIGI mappings for {len(symbols)} unique symbols...")

        # Fetch FIGI mappings
        figi_mapping = self._fetch_openfigi_mapping(symbols)

        # Add share_class_figi column
        self.master_tb = self.master_tb.with_columns(
            pl.col('symbol').map_elements(
                lambda s: figi_mapping.get(s),
                return_dtype=pl.Utf8
            ).alias('share_class_figi')
        )

        # Export to S3
        self.export_to_s3(s3_client, bucket_name)

        figi_count = sum(1 for v in figi_mapping.values() if v is not None)
        stats = {
            'rows': len(self.master_tb),
            'figi_mapped': figi_count
        }

        self.logger.info(
            f"CRSP rebuild complete: {stats['rows']} rows, "
            f"{stats['figi_mapped']}/{len(symbols)} FIGIs mapped"
        )

        return stats

    def close(self):
        """Close WRDS connection"""
        if self.db is not None:
            self.db.close()
    
