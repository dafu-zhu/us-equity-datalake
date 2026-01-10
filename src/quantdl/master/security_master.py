import wrds
import pandas as pd
import polars as pl
from dotenv import load_dotenv
import os
import time
import requests
import datetime as dt
from typing import List, Tuple, Optional
from pathlib import Path
import logging

from quantdl.utils.logger import setup_logger
from quantdl.universe.current import fetch_all_stocks

load_dotenv()


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
    def __init__(self, db: Optional[wrds.Connection] = None):
        if db is None:
            username = os.getenv('WRDS_USERNAME')
            password = os.getenv('WRDS_PASSWORD')
            self.db = wrds.Connection(
                wrds_username=username,
                wrds_password=password
            )
        else:
            self.db = db

        self.logger = setup_logger(
            name="master.SecurityMaster",
            log_dir=Path("data/logs/master"),
            level=logging.INFO
        )

        # Cache for SEC CIK mapping (loaded on-demand)
        self._sec_cik_cache: Optional[pl.DataFrame] = None

        self.cik_cusip = self.cik_cusip_mapping()
        self.master_tb = self.master_table()
    
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
        map_df = self.db.raw_sql(query)
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
            'security_id', 'permno', 'symbol', 'cik', 'start_date', 'end_date'
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

        Schema: (security_id, symbol, company, cik, cusip, start_date, end_date)
        """
        security_map = self.security_map()

        # Join with original cik_cusip to get company and cusip
        full_history = self.cik_cusip.join(
            security_map.select(['permno', 'symbol', 'cik', 'start_date', 'end_date', 'security_id']),
            on=['permno', 'symbol', 'cik', 'start_date', 'end_date'],
            how='left'
        )

        result = full_history.select([
            'security_id', 'symbol', 'company', 'cik', 'cusip', 'start_date', 'end_date'
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
        security_map = self.security_map()
        permno = (
            security_map.filter(
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

    def close(self):
        """Close WRDS connection"""
        self.db.close()
    

if __name__ == "__main__":
    # Example 1: Basic Symbol Normalization (without SecurityMaster validation)
    print("=" * 70)
    print("Example 1: Basic Symbol Normalization (No Validation)")
    print("=" * 70)

    normalizer = SymbolNormalizer()

    # Test cases demonstrating different formats for the same stock
    test_symbols = [
        ('BRKB', 'CRSP format'),
        ('BRK.B', 'Alpaca format'),
        ('BRK-B', 'SEC format'),
        ('AAPL', 'Simple ticker'),
        ('META', 'Simple ticker'),
        ('GOOGL', 'Simple ticker'),
        ('RKLB', 'No separator needed')
    ]

    print("\nNormalizing symbols to Nasdaq format (assumes same security):")
    print("-" * 70)
    for symbol, description in test_symbols:
        normalized = normalizer.to_nasdaq_format(symbol)
        print(f"{symbol:10} ({description:20}) -> {normalized}")

    # Demonstrate format conversions
    print("\n" + "=" * 70)
    print("Format Conversions:")
    print("-" * 70)
    nasdaq_symbol = 'BRK.B'
    print(f"Nasdaq format:  {nasdaq_symbol}")
    print(f"CRSP format:    {SymbolNormalizer.to_crsp_format(nasdaq_symbol)}")
    print(f"SEC format:     {SymbolNormalizer.to_sec_format(nasdaq_symbol)}")

    # Example 2: SecurityMaster Integration
    print("\n" + "=" * 70)
    print("Example 2: Symbol Normalization with SecurityMaster Validation")
    print("=" * 70)

    sm = SecurityMaster()
    normalizer_validated = SymbolNormalizer(security_master=sm)

    print("\nValidating symbol conversions with security_id:")
    print("-" * 70)

    # Test with date context
    test_cases_with_date = [
        ('BRKB', '2024-01-01', 'Active stock, same security'),
        ('BRKB', '2020-01-01', 'Active stock, historical date'),
    ]

    for symbol, date, description in test_cases_with_date:
        normalized = normalizer_validated.to_nasdaq_format(symbol, date)
        print(f"{symbol:10} on {date} ({description:25}) -> {normalized}")

    # Example 3: SecurityMaster Symbol Resolution
    print("\n" + "=" * 70)
    print("Example 3: SecurityMaster - Symbol Resolution Scenarios")
    print("=" * 70)

    # Scenario A: You ask for "AH" in 2012
    print(f"\nWho was AH in 2012? ID: {sm.get_security_id('AH', '2012-06-01')}")

    # Scenario B: You ask for "RCM" in 2012 (The Trap)
    print(f"Who was DNA in 2008? ID: {sm.get_security_id('DNA', '2008-06-01')}")

    # Scenario C: You ask for "RCM" in 2020
    print(f"Who was DNA in 2022? ID: {sm.get_security_id('DNA', '2022-01-01')}")

    print(f"\nWho was MSFT in 2024? ID: {sm.get_security_id('MSFT', '2024-01-01')}")
    print(f"Who was BRKB in 2022? ID: {sm.get_security_id('BRKB', '2022-01-01')}")

    # Example 3b: Auto-resolve distinguishes FB and META
    print("\n" + "-" * 70)
    print("Auto-resolve FB vs META (same company, different symbols)")
    print("-" * 70)
    fb_2023 = sm.get_security_id('FB', '2023-01-03', auto_resolve=True)
    meta_2023 = sm.get_security_id('META', '2023-01-03', auto_resolve=True)
    meta_2021 = sm.get_security_id('META', '2021-01-04', auto_resolve=True)
    fb_2021 = sm.get_security_id('FB', '2021-01-04', auto_resolve=True)
    print(f"FB on 2023-01-03 -> security_id: {fb_2023}")
    print(f"META on 2023-01-03 -> security_id: {meta_2023}")
    print(f"META on 2021-01-04 -> security_id: {meta_2021}")
    print(f"FB on 2021-01-04 -> security_id: {fb_2021}")
    print("Expect: FB 2023 == META 2023, META 2021 == FB 2021")

    # Example 4: Edge Case - Preventing False Matches
    print("\n" + "=" * 70)
    print("Example 4: Edge Case Prevention (ABCD vs ABC.D)")
    print("=" * 70)
    print("\nScenario: ABCD delisted in 2023, ABC.D started in 2025")
    print("Without validation: ABCD (2022) -> ABC.D (WRONG!)")
    print("With validation:    ABCD (2022) -> ABCD (CORRECT!)")
    print("\nValidation uses security_id to ensure same company:")
    print("- Look up security_id for 'ABCD' at historical date")
    print("- Look up security_id for 'ABCD' at 2024-12-31")
    print("- If different security_id -> keep original format")
    print("- If same security_id -> convert to Nasdaq format")

    sm.close()
    print("\n" + "=" * 70)
