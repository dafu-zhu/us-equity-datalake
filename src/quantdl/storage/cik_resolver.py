"""
CIK (Central Index Key) resolution functionality.

This module handles looking up CIKs for stock symbols using the SecurityMaster
database, with caching and fallback date logic to handle temporal mismatches.
"""

import datetime as dt
import logging
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl


class CIKResolver:
    """
    Resolves stock symbols to CIKs using SecurityMaster with caching and fallback logic.

    Handles:
    - Temporal mismatches (stocks that IPO'd mid-year)
    - Symbol format conversions (SEC vs CRSP)
    - Batch prefetching for performance
    - Persistent caching across lookups
    """

    def __init__(self, security_master, logger: logging.Logger):
        """
        Initialize CIK resolver.

        :param security_master: SecurityMaster instance from CRSP
        :param logger: Logger instance
        """
        self.security_master = security_master
        self.logger = logger
        self._cik_cache: Dict[Tuple[str, int], Optional[str]] = {}

    def get_cik(
        self,
        symbol: str,
        date: str,
        year: Optional[int] = None
    ) -> Optional[str]:
        """
        Get CIK for a symbol at a specific date using SecurityMaster.
        Handles both current and historical symbols through CRSP data.
        If the symbol is not active on the given date but year is provided,
        tries to find when it was active during that year.

        :param symbol: Symbol in SEC format (e.g., 'BRK-B', 'AAPL')
        :param date: Primary date in 'YYYY-MM-DD' format (e.g., '2010-01-15')
        :param year: Optional year to use for fallback date searches
        :return: CIK string (zero-padded to 10 digits) or None if not found
        """
        # Convert SEC format to CRSP format (BRK-B -> BRKB)
        crsp_symbol = symbol.replace('.', '').replace('-', '')
        date_year = int(date[:4])

        # For 2025+, prefer SEC official mapping (current snapshot)
        if (year is not None and year >= 2025) or date_year >= 2025:
            try:
                sec_map = self.security_master._fetch_sec_cik_mapping()
                sec_match = sec_map.filter(pl.col('ticker') == crsp_symbol).select('cik').head(1)
                if not sec_match.is_empty():
                    return sec_match.item()
            except Exception as e:
                self.logger.debug(f"SEC mapping lookup failed for {symbol}: {e}")

        # Try primary date first, then fallback dates if year is provided
        dates_to_try = [date]
        if year:
            # Add fallback dates: last day of year, mid-year, last day of each quarter
            dates_to_try.extend([
                f"{year}-12-31",  # End of year
                f"{year}-06-30",  # Mid-year
                f"{year}-09-30",  # Q3 end
                f"{year}-03-31",  # Q1 end
            ])
            # Remove duplicates while preserving order
            seen = set()
            dates_to_try = [d for d in dates_to_try if not (d in seen or seen.add(d))]

        for try_date in dates_to_try:
            try:
                # Use SecurityMaster to get security_id at the given date
                security_id = self.security_master.get_security_id(
                    symbol=crsp_symbol,
                    day=try_date,
                    auto_resolve=True  # Handle symbol changes automatically
                )
                if security_id is None:
                    continue

                # Query master table for CIK at this date
                master_tb = self.security_master.master_tb
                cik_record = master_tb.filter(
                    pl.col('security_id') == security_id,
                    pl.col('start_date') <= dt.datetime.strptime(try_date, '%Y-%m-%d').date(),
                    pl.col('end_date') >= dt.datetime.strptime(try_date, '%Y-%m-%d').date()
                ).select('cik').head(1)

                if cik_record.is_empty():
                    continue  # Try next date

                # Get CIK and ensure it's a string (may be int or None)
                cik_value = cik_record.item()
                if cik_value is None:
                    # NULL CIK in database - this is a non-SEC filer
                    # Don't try more dates, NULL is expected for this symbol
                    self.logger.debug(
                        f"Symbol {symbol} has NULL CIK in SecurityMaster - likely non-SEC filer "
                        f"(foreign company, small cap, OTC, etc.)"
                    )
                    return None

                # Success! Convert to zero-padded string
                cik_str = str(int(cik_value)).zfill(10)
                if try_date != date:
                    self.logger.debug(
                        f"Found CIK for {symbol} using fallback date {try_date} (primary: {date})"
                    )
                return cik_str

            except ValueError as e:
                # SecurityMaster couldn't resolve the symbol on this date
                if "not active on" in str(e):
                    continue  # Try next date
                else:
                    # Unexpected ValueError
                    self.logger.warning(
                        f"SecurityMaster error for {symbol} at {try_date}: {e}"
                    )
                    continue
            except Exception as e:
                self.logger.error(
                    f"Unexpected error getting CIK for {symbol} at {try_date}: {e}",
                    exc_info=True
                )
                continue

        # All dates failed
        self.logger.debug(
            f"Could not find CIK for {symbol} on any date in {year if year else date}"
        )
        return None

    def batch_prefetch_ciks(
        self,
        symbols: List[str],
        year: int,
        batch_size: int = 100
    ) -> Dict[str, Optional[str]]:
        """
        Batch pre-fetch CIKs for all symbols to avoid per-symbol database queries.
        Uses caching to avoid redundant lookups.

        :param symbols: List of symbols in SEC format (e.g., 'BRK-B', 'AAPL')
        :param year: Year for temporal context
        :param batch_size: Number of symbols to process in parallel
        :return: Dictionary mapping symbol -> CIK (or None if not found)
        """
        cik_map = {}
        symbols_to_fetch = []

        # Check cache first
        for sym in symbols:
            cache_key = (sym, year)
            if cache_key in self._cik_cache:
                cik_map[sym] = self._cik_cache[cache_key]
            else:
                symbols_to_fetch.append(sym)

        if not symbols_to_fetch:
            self.logger.info(f"All {len(symbols)} CIKs found in cache for {year}")
            return cik_map

        self.logger.info(
            f"Pre-fetching CIKs for {len(symbols_to_fetch)} symbols "
            f"(year={year}, cached={len(cik_map)})"
        )

        # Reference date for CIK lookup (mid-year is most likely to be active)
        reference_date = f"{year}-06-30"

        # Use ThreadPoolExecutor for parallel CIK fetching
        def fetch_single_cik(sym: str) -> Tuple[str, Optional[str]]:
            cik = self.get_cik(sym, reference_date, year=year)
            return (sym, cik)

        # Process in batches to control memory
        total_fetched = 0
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(fetch_single_cik, sym): sym
                for sym in symbols_to_fetch
            }

            for future in as_completed(futures):
                sym, cik = future.result()
                cik_map[sym] = cik

                # Update cache
                cache_key = (sym, year)
                self._cik_cache[cache_key] = cik

                total_fetched += 1
                if total_fetched % 100 == 0:
                    self.logger.info(
                        f"CIK pre-fetch progress: {total_fetched}/{len(symbols_to_fetch)}"
                    )

        # Log statistics
        found = sum(1 for cik in cik_map.values() if cik is not None)
        null_count = sum(1 for cik in cik_map.values() if cik is None)

        self.logger.info(
            f"CIK pre-fetch complete: {found}/{len(symbols)} found "
            f"({found/len(symbols)*100:.1f}%), {null_count} non-SEC filers"
        )

        # Log specific symbols without CIKs (with company names from SecurityMaster)
        if null_count > 0:
            symbols_without_cik = [sym for sym, cik in cik_map.items() if cik is None]

            # Try to get company names from SecurityMaster
            if null_count <= 50:
                # For small lists, show all with company names
                self.logger.info(f"Symbols without CIK ({null_count}): {sorted(symbols_without_cik)}")

                # Get company names for these symbols
                try:
                    master_tb = self.security_master.master_tb
                    null_details = master_tb.filter(
                        pl.col('symbol').is_in(symbols_without_cik),
                        pl.col('cik').is_null()
                    ).select(['symbol', 'company']).unique()

                    if not null_details.is_empty():
                        self.logger.info("Non-SEC filers details:")
                        for row in null_details.head(20).iter_rows(named=True):
                            self.logger.info(f"  {row['symbol']:10} - {row['company']}")

                        if len(null_details) > 20:
                            self.logger.info(f"  ... and {len(null_details) - 20} more")
                except Exception as e:
                    self.logger.debug(f"Could not fetch company names: {e}")
            else:
                # For large lists, just show first 50 symbols
                self.logger.info(
                    f"Symbols without CIK ({null_count}): {sorted(symbols_without_cik)[:50]} "
                    f"... and {null_count - 50} more"
                )

        return cik_map

    def clear_cache(self):
        """Clear the CIK cache."""
        self._cik_cache.clear()
        self.logger.info("CIK cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the cache."""
        return {
            'total_entries': len(self._cik_cache),
            'cached_ciks': sum(1 for v in self._cik_cache.values() if v is not None),
            'null_entries': sum(1 for v in self._cik_cache.values() if v is None)
        }
