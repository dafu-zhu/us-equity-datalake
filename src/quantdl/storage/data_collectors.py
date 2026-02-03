"""
Data collection functionality for market data.

This module handles fetching market data from various sources:
- Daily ticks from CRSP or Alpaca
- Minute ticks from Alpaca API
- Fundamental data from SEC EDGAR API
"""

import datetime as dt
import time
import logging
import threading
from collections import OrderedDict
from typing import List, Dict, Optional
from pathlib import Path
import yaml
import polars as pl

from quantdl.collection.models import TickField, DataCollector
from quantdl.collection.fundamental import Fundamental, DURATION_CONCEPTS
from quantdl.universe.current import fetch_all_stocks
from quantdl.utils.logger import setup_logger
from quantdl.utils.mapping import align_calendar
from quantdl.derived.ttm import compute_ttm_long
from quantdl.derived.metrics import compute_derived


class TicksDataCollector(DataCollector):
    """Handles tick data collection (daily and minute resolution)."""

    def __init__(
        self,
        crsp_ticks,
        alpaca_ticks,
        alpaca_headers: dict,
        logger: logging.Logger,
        alpaca_start_year: int = 2025
    ):
        super().__init__(logger=logger)
        self.crsp_ticks = crsp_ticks
        self.alpaca_ticks = alpaca_ticks
        self.alpaca_headers = alpaca_headers
        self.alpaca_start_year = alpaca_start_year

    def _normalize_daily_df(self, df: pl.DataFrame) -> pl.DataFrame:
        if len(df) > 0:
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                df = df.with_columns([pl.lit(None).alias(col) for col in missing_cols])

            df = df.with_columns([
                pl.col('timestamp').cast(pl.Utf8),
                pl.col('open').cast(pl.Float64).round(4),
                pl.col('high').cast(pl.Float64).round(4),
                pl.col('low').cast(pl.Float64).round(4),
                pl.col('close').cast(pl.Float64).round(4),
                pl.col('volume').cast(pl.Int64)
            ]).sort('timestamp')

        return df

    def _bars_to_daily_df(self, bars: List[Dict]) -> pl.DataFrame:
        if not bars:
            return pl.DataFrame()

        from dataclasses import asdict
        parsed_ticks = self.alpaca_ticks.parse_ticks(bars)
        ticks_data = [asdict(dp) for dp in parsed_ticks]

        df = pl.DataFrame(ticks_data).with_columns([
            pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S').dt.date(),
            pl.col('open').cast(pl.Float64),
            pl.col('high').cast(pl.Float64),
            pl.col('low').cast(pl.Float64),
            pl.col('close').cast(pl.Float64),
            pl.col('volume').cast(pl.Int64)
        ]).select(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        return df

    def collect_daily_ticks_year(self, sym: str, year: int) -> pl.DataFrame:
        """
        Fetch daily ticks for entire year from appropriate source and return as Polars DataFrame.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :return: Polars DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if year < self.alpaca_start_year:
            # Use CRSP for years < 2025 (avoids survivorship bias)
            crsp_symbol = sym.replace('.', '').replace('-', '')
            all_months_data = []

            # Fetch all 12 months
            for month in range(1, 13):
                try:
                    json_list = self.crsp_ticks.collect_daily_ticks(
                        symbol=crsp_symbol,
                        year=year,
                        month=month,
                        adjusted=True,
                        auto_resolve=True
                    )
                    all_months_data.extend(json_list)
                except ValueError as e:
                    if "not active on" in str(e):
                        # Symbol not active in this month, skip
                        continue
                    else:
                        raise

            # Convert to Polars DataFrame
            if not all_months_data:
                return pl.DataFrame()

            df = pl.DataFrame(all_months_data)
        else:
            # Use Alpaca for years >= 2025
            try:
                bars_map = self.alpaca_ticks.fetch_daily_year_bulk(
                    symbols=[sym],
                    year=year,
                    adjusted=True
                )
                df = self._bars_to_daily_df(bars_map.get(sym, []))
            except Exception as e:
                # Return empty DataFrame if fetch fails
                self.logger.warning(f"Failed to fetch {sym} for {year}: {e}")
                return pl.DataFrame()

        return self._normalize_daily_df(df)

    def collect_daily_ticks_year_bulk(self, symbols: List[str], year: int) -> Dict[str, pl.DataFrame]:
        """
        Bulk fetch daily ticks for a full year and return a mapping of symbol -> DataFrame.
        Uses CRSP for years < 2025 and Alpaca bulk fetch for years >= 2025.
        """
        if year < self.alpaca_start_year:
            return self.crsp_ticks.collect_daily_ticks_year_bulk(symbols, year, adjusted=True, auto_resolve=True)

        symbol_bars = self.alpaca_ticks.fetch_daily_year_bulk(symbols, year, adjusted=True)
        result = {}
        for sym in symbols:
            df = self._bars_to_daily_df(symbol_bars.get(sym, []))
            result[sym] = self._normalize_daily_df(df)
        return result

    def collect_daily_ticks_month(
        self,
        sym: str,
        year: int,
        month: int,
        year_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Fetch daily ticks for a specific month from appropriate source and return as Polars DataFrame.
        Directly fetches only the requested month from the source API (no year-level fetch).
        If year_df is provided, filters the month from that DataFrame instead of refetching.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :param month: Month to fetch data for (1-12)
        :return: Polars DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if year_df is not None:
            if len(year_df) == 0:
                return pl.DataFrame()
            month_prefix = f"{year}-{month:02d}"
            df = year_df.filter(
                pl.col('timestamp').cast(pl.Utf8).str.slice(0, 7).eq(month_prefix)
            )
            if len(df) == 0:
                return pl.DataFrame()
            calendar_path = getattr(self.crsp_ticks, "calendar_path", None)
            if isinstance(calendar_path, (str, Path)) and Path(calendar_path).exists():
                start_date_obj = dt.date(year, month, 1)
                if month == 12:
                    end_date_obj = dt.date(year, 12, 31)
                else:
                    end_date_obj = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
                df = pl.DataFrame(
                    align_calendar(
                        df.to_dicts(),
                        start_date_obj,
                        end_date_obj,
                        Path(calendar_path)
                    )
                )
        elif year < self.alpaca_start_year:
            # Use CRSP for years < 2025 (avoids survivorship bias)
            crsp_symbol = sym.replace('.', '').replace('-', '')
            try:
                json_list = self.crsp_ticks.collect_daily_ticks(
                    symbol=crsp_symbol,
                    year=year,
                    month=month,
                    adjusted=True,
                    auto_resolve=True
                )

                # Convert to Polars DataFrame
                if not json_list:
                    return pl.DataFrame()

                df = pl.DataFrame(json_list)
            except ValueError as e:
                if "not active on" in str(e):
                    # Symbol not active in this month
                    return pl.DataFrame()
                else:
                    raise
        else:
            # Use Alpaca for years >= 2025
            try:
                symbol_bars = self.alpaca_ticks.fetch_daily_month_bulk(
                    symbols=[sym],
                    year=year,
                    month=month,
                    adjusted=True
                )
                df = self._bars_to_daily_df(symbol_bars.get(sym, []))
            except Exception as e:
                # Return empty DataFrame if fetch fails
                self.logger.warning(f"Failed to fetch {sym} for {year}-{month:02d}: {e}")
                return pl.DataFrame()

        return self._normalize_daily_df(df)

    def collect_daily_ticks_month_bulk(
        self,
        symbols: List[str],
        year: int,
        month: int,
        sleep_time: float = 0.2
    ) -> Dict[str, pl.DataFrame]:
        """
        Bulk fetch daily ticks for a specific month and return a mapping of symbol -> DataFrame.
        """
        if year < self.alpaca_start_year:
            result = {}
            for sym in symbols:
                result[sym] = self.collect_daily_ticks_month(sym, year, month)
            return result

        symbol_bars = self.alpaca_ticks.fetch_daily_month_bulk(
            symbols=symbols,
            year=year,
            month=month,
            sleep_time=sleep_time,
            adjusted=True
        )
        result = {}
        for sym in symbols:
            df = self._bars_to_daily_df(symbol_bars.get(sym, []))
            result[sym] = self._normalize_daily_df(df)
        return result

    def fetch_minute_month(
        self,
        symbols: List[str],
        year: int,
        month: int,
        sleep_time: float = 0.2
    ) -> dict:
        """
        Bulk fetch minute data for multiple symbols for the specified month from Alpaca.
        Delegates to alpaca_ticks for the actual API interaction.

        :param symbols: List of symbols in Alpaca format (e.g., ['AAPL', 'MSFT'])
        :param year: Year to fetch
        :param month: Month to fetch (1-12)
        :param sleep_time: Sleep time between requests in seconds (default: 0.2)
        :return: Dict mapping symbol -> list of bars
        """
        return self.alpaca_ticks.fetch_minute_month_bulk(symbols, year, month, sleep_time)

    def fetch_minute_day(
        self,
        symbols: List[str],
        trade_day: str,
        sleep_time: float = 0.2
    ) -> dict:
        """
        Fetch minute data for multiple symbols for a specific trading day from Alpaca.
        Useful for daily incremental updates without refetching entire month.

        :param symbols: List of symbols in Alpaca format (e.g., ['AAPL', 'MSFT'])
        :param trade_day: Trading day (format: "YYYY-MM-DD")
        :param sleep_time: Sleep time between requests in seconds (default: 0.2)
        :return: Dict mapping symbol -> list of bars
        """
        return self.alpaca_ticks.fetch_minute_day_bulk(symbols, trade_day, sleep_time)

    def parse_minute_bars_to_daily(
        self,
        symbol_bars: Dict[str, List],
        trading_days: List[str]
    ) -> Dict[tuple, pl.DataFrame]:
        """
        Parse minute bars organized by symbol into (symbol, day) tuples with DataFrames.
        Uses vectorized operations for efficient processing.

        :param symbol_bars: Dict mapping symbol -> list of bars (from fetch_minute_month)
        :param trading_days: List of trading days in 'YYYY-MM-DD' format
        :return: Dict mapping (symbol, day) -> DataFrame of minute data
        """
        result = {}

        for sym, bars in symbol_bars.items():
            if not bars:
                # No data for this symbol, mark all days as empty
                for day in trading_days:
                    result[(sym, day)] = pl.DataFrame()
                continue

            try:
                # Convert all bars to DataFrame at once using vectorized operations
                timestamps = [bar[TickField.TIMESTAMP.value] for bar in bars]
                opens = [bar[TickField.OPEN.value] for bar in bars]
                highs = [bar[TickField.HIGH.value] for bar in bars]
                lows = [bar[TickField.LOW.value] for bar in bars]
                closes = [bar[TickField.CLOSE.value] for bar in bars]
                volumes = [bar[TickField.VOLUME.value] for bar in bars]
                num_trades_list = [bar[TickField.NUM_TRADES.value] for bar in bars]
                vwaps = [bar[TickField.VWAP.value] for bar in bars]

                # Create DataFrame and process with vectorized operations
                # Alpaca returns RFC-3339 timestamps with nanoseconds (e.g., "2024-01-03T14:30:00.123456789Z")
                # Strip nanoseconds to seconds precision before parsing
                all_bars_df = pl.DataFrame({
                    'timestamp_utc': timestamps,
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes,
                    'num_trades': num_trades_list,
                    'vwap': vwaps
                }, strict=False).with_columns([
                    # Strip fractional seconds: "2024-01-03T14:30:00.123456789Z" -> "2024-01-03T14:30:00Z"
                    pl.col('timestamp_utc')
                        .str.replace(r'\.\d+Z$', 'Z')
                        .alias('timestamp_clean')
                ]).with_columns([
                    # Parse timestamp: UTC -> ET, remove timezone
                    pl.col('timestamp_clean')
                        .str.strptime(pl.Datetime('us', 'UTC'), format='%Y-%m-%dT%H:%M:%SZ')
                        .dt.convert_time_zone('America/New_York')
                        .dt.replace_time_zone(None)
                        .alias('timestamp'),
                    # Cast types (vectorized)
                    pl.col('open').cast(pl.Float64),
                    pl.col('high').cast(pl.Float64),
                    pl.col('low').cast(pl.Float64),
                    pl.col('close').cast(pl.Float64),
                    pl.col('volume').cast(pl.Int64),
                    pl.col('num_trades').cast(pl.Int64),
                    pl.col('vwap').cast(pl.Float64)
                ]).drop(['timestamp_utc', 'timestamp_clean']).with_columns([
                    # Extract trade date from ET timestamp (not UTC!) for correct day grouping
                    pl.col('timestamp').dt.date().cast(pl.Utf8).alias('trade_date')
                ])

                # Process each trading day by filtering
                for day in trading_days:
                    day_df = all_bars_df.filter(pl.col('trade_date') == day)

                    if len(day_df) > 0:
                        # Select final columns for upload
                        minute_df = day_df.select([
                            'timestamp', 'open', 'high', 'low', 'close',
                            'volume', 'num_trades', 'vwap'
                        ])
                    else:
                        # Empty DataFrame for days with no data
                        minute_df = pl.DataFrame()

                    result[(sym, day)] = minute_df

            except Exception as e:
                self.logger.error(f"Error processing bars for {sym}: {e}", exc_info=True)
                # Mark all days as empty for this symbol
                for day in trading_days:
                    result[(sym, day)] = pl.DataFrame()

        return result


class FundamentalDataCollector(DataCollector):
    """Handles fundamental data collection from SEC EDGAR."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        sec_rate_limiter=None,
        fundamental_cache: Optional[OrderedDict] = None,
        fundamental_cache_lock: Optional[threading.Lock] = None,
        fundamental_cache_size: int = 128
    ):
        # Create logger if not provided (for backward compatibility)
        if logger is None:
            logger = setup_logger(
                name="storage.fundamental_data_collector",
                log_dir="data/logs/fundamental",
                level=logging.INFO
            )
        super().__init__(logger=logger)
        self.sec_rate_limiter = sec_rate_limiter

        # Use shared cache or create new one
        if fundamental_cache is not None and fundamental_cache_lock is not None:
            self._fundamental_cache = fundamental_cache
            self._fundamental_cache_lock = fundamental_cache_lock
            self._fundamental_cache_size = fundamental_cache_size
        else:
            self._fundamental_cache_size = max(int(fundamental_cache_size), 0)
            self._fundamental_cache = OrderedDict()
            self._fundamental_cache_lock = threading.Lock()

    def _load_concepts(
        self,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> List[str]:
        if concepts is not None:
            return concepts
        if config_path is None:
            config_path = Path("configs/approved_mapping.yaml")

        with open(config_path) as f:
            mappings = yaml.safe_load(f)
            return list(mappings.keys())

    def _get_or_create_fundamental(
        self,
        cik: str,
        symbol: Optional[str] = None
    ) -> Fundamental:
        """
        Get cached Fundamental object or create new one.
        Avoids redundant SEC API calls by reusing fetched data.

        :param cik: Company CIK number
        :param symbol: Optional symbol for logging
        :return: Fundamental object (cached or newly created)
        """
        if self._fundamental_cache_size <= 0:
            return Fundamental(
                cik=cik,
                symbol=symbol,
                rate_limiter=self.sec_rate_limiter
            )

        with self._fundamental_cache_lock:
            cached = self._fundamental_cache.get(cik)
            if cached is not None:
                self._fundamental_cache.move_to_end(cik)
                return cached

        created = Fundamental(
            cik=cik,
            symbol=symbol,
            rate_limiter=self.sec_rate_limiter
        )

        with self._fundamental_cache_lock:
            cached = self._fundamental_cache.get(cik)
            if cached is not None:
                self._fundamental_cache.move_to_end(cik)
                return cached
            self._fundamental_cache[cik] = created
            if len(self._fundamental_cache) > self._fundamental_cache_size:
                self._fundamental_cache.popitem(last=False)
        return created

    def collect_fundamental_long(
        self,
        cik: str,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> pl.DataFrame:
        """
        Fetch long-format fundamental data for a filing date range.

        Returns columns:
        [symbol, as_of_date, accn, form, concept, value, start, end, frame, is_instant]
        """
        try:
            concepts = self._load_concepts(concepts, config_path)

            # Use cached Fundamental object to avoid redundant API calls
            fnd = self._get_or_create_fundamental(cik=cik, symbol=symbol)
            start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
            end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

            records = []
            concepts_found = []
            concepts_missing = []

            for concept in concepts:
                try:
                    dps = fnd.get_concept_data(
                        concept,
                        start_date=start_date,
                        end_date=end_date
                    )
                    if dps:
                        concepts_found.append(concept)
                        for dp in dps:
                            if not (start_dt <= dp.timestamp <= end_dt):
                                continue
                            records.append(
                                {
                                    "symbol": symbol,
                                    "as_of_date": dp.timestamp.isoformat(),
                                    "accn": dp.accn,
                                    "form": dp.form,
                                    "concept": concept,
                                    "value": dp.value,
                                    "start": dp.start_date.isoformat() if dp.start_date else None,
                                    "end": dp.end_date.isoformat(),
                                    "frame": dp.frame,
                                    "is_instant": dp.is_instant,
                                }
                            )
                    else:
                        concepts_missing.append(concept)
                except Exception as e:
                    self.logger.debug(f"Failed to extract concept '{concept}' for CIK {cik}: {e}")
                    concepts_missing.append(concept)

            if not records:
                self.logger.warning(
                    f"No fundamental data found for CIK {cik} ({symbol}) "
                    f"from {start_date} to {end_date}"
                )
                return pl.DataFrame()

            self.logger.debug(
                f"CIK {cik} ({symbol}): {len(concepts_found)}/{len(concepts)} concepts available "
                f"({len(concepts_missing)} missing)"
            )

            return pl.DataFrame(records)

        except Exception as e:
            self.logger.error(
                f"Failed to collect fundamental data for CIK {cik} ({symbol}) "
                f"from {start_date} to {end_date}: {e}"
            )
            return pl.DataFrame()

    def collect_ttm_long_range(
        self,
        cik: str,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> pl.DataFrame:
        """
        Compute TTM long-format data in memory for a date range.
        """
        try:
            concepts = self._load_concepts(concepts, config_path)
            end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
            start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()

            fnd = self._get_or_create_fundamental(cik=cik, symbol=symbol)

            records = []
            for concept in concepts:
                try:
                    dps = fnd.get_concept_data(concept)
                    if not dps:
                        continue
                    for dp in dps:
                        if dp.timestamp > end_dt:
                            continue
                        records.append(
                            {
                                "symbol": symbol,
                                "as_of_date": dp.timestamp.isoformat(),
                                "accn": dp.accn,
                                "form": dp.form,
                                "concept": concept,
                                "value": dp.value,
                                "start": dp.start_date.isoformat() if dp.start_date else None,
                                "end": dp.end_date.isoformat(),
                                "frame": dp.frame,
                            }
                        )
                except Exception as e:
                    self.logger.debug(f"Failed to extract concept '{concept}' for CIK {cik}: {e}")

            if not records:
                self.logger.warning(
                    f"No fundamental data found for CIK {cik} ({symbol}) "
                    f"from {start_date} to {end_date}"
                )
                return pl.DataFrame()

            record_schema = {
                "symbol": pl.Utf8,
                "as_of_date": pl.Utf8,
                "accn": pl.Utf8,
                "form": pl.Utf8,
                "concept": pl.Utf8,
                "value": pl.Float64,
                "start": pl.Utf8,
                "end": pl.Utf8,
                "frame": pl.Utf8,
            }
            ttm_df = compute_ttm_long(
                pl.DataFrame(records, schema=record_schema, strict=False),
                logger=self.logger,
                symbol=symbol
            )
            if len(ttm_df) == 0:
                return pl.DataFrame()

            return ttm_df.filter(
                (pl.col("as_of_date") >= start_dt.strftime("%Y-%m-%d"))
                & (pl.col("as_of_date") <= end_dt.strftime("%Y-%m-%d"))
            )

        except Exception as e:
            self.logger.error(
                f"Failed to compute TTM for CIK {cik} ({symbol}) "
                f"from {start_date} to {end_date}: {e}"
            )
            return pl.DataFrame()

    def _build_metrics_wide(
        self,
        cik: str,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> tuple[pl.DataFrame, Optional[pl.DataFrame]]:
        concepts_list = self._load_concepts(concepts, config_path)
        stock_concepts = [c for c in concepts_list if c not in DURATION_CONCEPTS]
        duration_concepts = [c for c in concepts_list if c in DURATION_CONCEPTS]

        try:
            fund = self._get_or_create_fundamental(cik=cik, symbol=symbol)
        except Exception as e:
            self.logger.error(f"Failed to initialize Fundamental for CIK {cik} ({symbol}): {e}")
            return pl.DataFrame(), None

        start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

        record_schema = {
            "symbol": pl.Utf8,
            "as_of_date": pl.Utf8,
            "accn": pl.Utf8,
            "form": pl.Utf8,
            "concept": pl.Utf8,
            "value": pl.Float64,
            "start": pl.Utf8,
            "end": pl.Utf8,
            "frame": pl.Utf8,
        }

        ttm_records = []
        raw_records = []

        for concept in concepts_list:
            try:
                dps = fund.get_concept_data(concept)
            except Exception as e:
                self.logger.debug(f"Failed to extract concept '{concept}' for CIK {cik}: {e}")
                continue

            if not dps:
                continue

            for dp in dps:
                if start_dt <= dp.timestamp <= end_dt:
                    raw_records.append(
                        {
                            "symbol": symbol,
                            "as_of_date": dp.timestamp.isoformat(),
                            "accn": dp.accn,
                            "form": dp.form,
                            "concept": concept,
                            "value": dp.value,
                            "start": dp.start_date.isoformat() if dp.start_date else None,
                            "end": dp.end_date.isoformat(),
                            "frame": dp.frame,
                        }
                    )

                if concept in DURATION_CONCEPTS and dp.timestamp <= end_dt:
                    ttm_records.append(
                        {
                            "symbol": symbol,
                            "as_of_date": dp.timestamp.isoformat(),
                            "accn": dp.accn,
                            "form": dp.form,
                            "concept": concept,
                            "value": dp.value,
                            "start": dp.start_date.isoformat() if dp.start_date else None,
                            "end": dp.end_date.isoformat(),
                            "frame": dp.frame,
                        }
                    )

        if not ttm_records:
            return pl.DataFrame(), None
        ttm_df = compute_ttm_long(
            pl.DataFrame(ttm_records, schema=record_schema, strict=False),
            logger=self.logger,
            symbol=symbol
        )
        if len(ttm_df) == 0:
            return pl.DataFrame(), None
        ttm_df = ttm_df.filter(
            (pl.col("as_of_date") >= start_dt.strftime("%Y-%m-%d"))
            & (pl.col("as_of_date") <= end_dt.strftime("%Y-%m-%d"))
        )
        if len(ttm_df) == 0:
            return pl.DataFrame(), None

        metadata_df = (
            ttm_df.select(["symbol", "as_of_date", "accn", "form", "frame"])
            .group_by(["symbol", "as_of_date"])
            .agg([pl.col("accn").last(), pl.col("form").last(), pl.col("frame").last()])
        )

        ttm_wide = ttm_df.pivot(
            values="value",
            index=["symbol", "as_of_date"],
            on="concept",
            aggregate_function="first",
        )
        missing_duration = [c for c in duration_concepts if c not in ttm_wide.columns]
        if missing_duration:
            ttm_wide = ttm_wide.with_columns([pl.lit(None).alias(c) for c in missing_duration])

        base = ttm_wide.with_columns(
            pl.col("as_of_date").str.strptime(pl.Date, "%Y-%m-%d")
        ).sort(["symbol", "as_of_date"])

        raw_long = pl.DataFrame(raw_records, schema=record_schema, strict=False)
        if len(raw_long) == 0:
            for concept in stock_concepts:
                if concept not in base.columns:
                    base = base.with_columns(pl.lit(None).alias(concept))
            return base.with_columns(pl.col("as_of_date").dt.strftime("%Y-%m-%d")), metadata_df
        stock_long = raw_long.filter(pl.col("concept").is_in(stock_concepts)).with_columns(
            pl.col("as_of_date").str.strptime(pl.Date, "%Y-%m-%d")
        ).sort(["symbol", "as_of_date"])

        if len(stock_long) > 0:
            stock_wide = (
                stock_long
                .group_by(["symbol", "as_of_date", "concept"])
                .agg(pl.col("value").last())
                .pivot(
                    values="value",
                    index=["symbol", "as_of_date"],
                    on="concept",
                    aggregate_function="first",
                )
                .sort(["symbol", "as_of_date"])
            )
            base = base.join_asof(
                stock_wide,
                on="as_of_date",
                by="symbol",
                strategy="backward"
            )

        missing_cols = [c for c in stock_concepts if c not in base.columns]
        if missing_cols:
            base = base.with_columns([pl.lit(None).alias(c) for c in missing_cols])

        return base.with_columns(pl.col("as_of_date").dt.strftime("%Y-%m-%d")), metadata_df

    def collect_derived_long(
        self,
        cik: str,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> tuple[pl.DataFrame, Optional[str]]:
        """
        Compute derived metrics and return long-format data.

        Returns columns: [symbol, as_of_date, metric, value]
        """
        metrics_wide, _ = self._build_metrics_wide(
            cik=cik,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            concepts=concepts,
            config_path=config_path
        )
        if len(metrics_wide) == 0:
            return pl.DataFrame(), "metrics_wide_empty"

        id_cols = ["symbol", "as_of_date"]
        value_cols = [c for c in metrics_wide.columns if c not in id_cols]
        long_input = metrics_wide.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            variable_name="concept",
            value_name="value",
        ).drop_nulls(subset=["value"])

        derived_long = compute_derived(long_input, logger=self.logger, symbol=symbol)
        if len(derived_long) == 0:
            return pl.DataFrame(), "derived_empty"

        start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        return derived_long.filter(
            (pl.col("as_of_date") >= start_dt.strftime("%Y-%m-%d"))
            & (pl.col("as_of_date") <= end_dt.strftime("%Y-%m-%d"))
        ), None


class UniverseDataCollector(DataCollector):
    """Handles stock universe data collection."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Create logger if not provided (for backward compatibility)
        if logger is None:
            logger = setup_logger(
                name="storage.universe_data_collector",
                log_dir="data/logs/symbols",
                level=logging.INFO
            )
        super().__init__(logger=logger)

    def collect_current_universe(self, with_filter: bool = True, refresh: bool = False):
        """Fetch current universe of stocks."""
        return fetch_all_stocks(with_filter=with_filter, refresh=refresh)


class DataCollectors:
    """
    Orchestrator for data collection from various market data sources.
    Delegates to specialized collector instances.
    """

    def __init__(
        self,
        crsp_ticks,
        alpaca_ticks,
        alpaca_headers: dict,
        logger: logging.Logger,
        sec_rate_limiter=None,
        fundamental_cache_size: int = 128,
        alpaca_start_year: int = 2025
    ):
        """
        Initialize data collectors.

        :param crsp_ticks: CRSPDailyTicks instance
        :param alpaca_ticks: Alpaca Ticks instance
        :param alpaca_headers: Headers for Alpaca API requests
        :param logger: Logger instance
        :param sec_rate_limiter: Optional rate limiter for SEC API calls
        :param fundamental_cache_size: Size of LRU cache for Fundamental objects
        """
        self.logger = logger

        # Shared fundamental cache (managed centrally)
        self._fundamental_cache_size = max(int(fundamental_cache_size), 0)
        self._fundamental_cache: "OrderedDict[str, Fundamental]" = OrderedDict()
        self._fundamental_cache_lock = threading.Lock()

        # Create specialized collectors with dependency injection
        self.ticks_collector = TicksDataCollector(
            crsp_ticks=crsp_ticks,
            alpaca_ticks=alpaca_ticks,
            alpaca_headers=alpaca_headers,
            logger=logger,
            alpaca_start_year=alpaca_start_year
        )

        self.fundamental_collector = FundamentalDataCollector(
            logger=logger,
            sec_rate_limiter=sec_rate_limiter,
            fundamental_cache=self._fundamental_cache,
            fundamental_cache_lock=self._fundamental_cache_lock,
            fundamental_cache_size=fundamental_cache_size
        )

        self.universe_collector = UniverseDataCollector(logger=logger)

    # Delegation methods for fundamental collection
    def _load_concepts(
        self,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> List[str]:
        return self.fundamental_collector._load_concepts(concepts, config_path)

    def collect_fundamental_long(
        self,
        cik: str,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> pl.DataFrame:
        return self.fundamental_collector.collect_fundamental_long(
            cik, start_date, end_date, symbol, concepts, config_path
        )

    def collect_ttm_long_range(
        self,
        cik: str,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> pl.DataFrame:
        return self.fundamental_collector.collect_ttm_long_range(
            cik, start_date, end_date, symbol, concepts, config_path
        )

    def collect_derived_long(
        self,
        cik: str,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> tuple[pl.DataFrame, Optional[str]]:
        return self.fundamental_collector.collect_derived_long(
            cik, start_date, end_date, symbol, concepts, config_path
        )

    # Delegation methods for ticks collection
    def collect_daily_ticks_year(self, sym: str, year: int) -> pl.DataFrame:
        return self.ticks_collector.collect_daily_ticks_year(sym, year)

    def collect_daily_ticks_month(
        self,
        sym: str,
        year: int,
        month: int,
        year_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        return self.ticks_collector.collect_daily_ticks_month(sym, year, month, year_df=year_df)

    def collect_daily_ticks_year_bulk(self, symbols: List[str], year: int) -> Dict[str, pl.DataFrame]:
        return self.ticks_collector.collect_daily_ticks_year_bulk(symbols, year)

    def collect_daily_ticks_month_bulk(
        self,
        symbols: List[str],
        year: int,
        month: int,
        sleep_time: float = 0.2
    ) -> Dict[str, pl.DataFrame]:
        return self.ticks_collector.collect_daily_ticks_month_bulk(symbols, year, month, sleep_time)

    def fetch_minute_month(
        self,
        symbols: List[str],
        year: int,
        month: int,
        sleep_time: float = 0.2
    ) -> dict:
        return self.ticks_collector.fetch_minute_month(symbols, year, month, sleep_time)

    def fetch_minute_day(
        self,
        symbols: List[str],
        trade_day: str,
        sleep_time: float = 0.2
    ) -> dict:
        return self.ticks_collector.fetch_minute_day(symbols, trade_day, sleep_time)

    def parse_minute_bars_to_daily(
        self,
        symbol_bars: Dict[str, List],
        trading_days: List[str]
    ) -> Dict[tuple, pl.DataFrame]:
        return self.ticks_collector.parse_minute_bars_to_daily(symbol_bars, trading_days)
