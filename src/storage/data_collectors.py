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
from typing import List, Dict, Optional
from pathlib import Path
import yaml
import requests
import polars as pl

from collection.models import TickField
from collection.fundamental import Fundamental, DURATION_CONCEPTS
from derived.ttm import compute_ttm_long
from derived.metrics import compute_derived


class DataCollectors:
    """
    Handles data collection from various market data sources.
    """

    def __init__(
        self,
        crsp_ticks,
        alpaca_ticks,
        alpaca_headers: dict,
        logger: logging.Logger,
        sec_rate_limiter=None
    ):
        """
        Initialize data collectors.

        :param crsp_ticks: CRSPDailyTicks instance
        :param alpaca_ticks: Alpaca Ticks instance
        :param alpaca_headers: Headers for Alpaca API requests
        :param logger: Logger instance
        :param sec_rate_limiter: Optional rate limiter for SEC API calls
        """
        self.crsp_ticks = crsp_ticks
        self.alpaca_ticks = alpaca_ticks
        self.alpaca_headers = alpaca_headers
        self.logger = logger
        self.sec_rate_limiter = sec_rate_limiter

        # Cache for Fundamental objects to avoid redundant API calls
        self._fundamental_cache: Dict[str, Fundamental] = {}

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
        if cik not in self._fundamental_cache:
            self._fundamental_cache[cik] = Fundamental(
                cik=cik,
                symbol=symbol,
                rate_limiter=self.sec_rate_limiter
            )
        return self._fundamental_cache[cik]

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
        [symbol, as_of_date, accn, form, concept, value, start, end, fp]
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
                                    "fp": dp.fp,
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
                        if dp.end_date and dp.end_date > end_dt:
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
                                "fp": dp.fp,
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

            ttm_df = compute_ttm_long(pl.DataFrame(records), logger=self.logger, symbol=symbol)
            if len(ttm_df) == 0:
                return pl.DataFrame()

            return ttm_df.filter(
                (pl.col("as_of_date") >= start_dt.strftime("%Y-%m-%d"))
                & (pl.col("as_of_date") <= end_dt.strftime("%Y-%m-%d"))
                & (pl.col("end") >= start_dt.strftime("%Y-%m-%d"))
                & (pl.col("end") <= end_dt.strftime("%Y-%m-%d"))
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
    ) -> pl.DataFrame:
        concepts_list = self._load_concepts(concepts, config_path)
        stock_concepts = [c for c in concepts_list if c not in DURATION_CONCEPTS]
        duration_concepts = [c for c in concepts_list if c in DURATION_CONCEPTS]

        try:
            fund = self._get_or_create_fundamental(cik=cik, symbol=symbol)
        except Exception as e:
            self.logger.error(f"Failed to initialize Fundamental for CIK {cik} ({symbol}): {e}")
            return pl.DataFrame()

        start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

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
                            "fp": dp.fp,
                        }
                    )

                if concept in DURATION_CONCEPTS and dp.end_date and dp.end_date <= end_dt:
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
                            "fp": dp.fp,
                        }
                    )

        if not ttm_records:
            empty_cols = {
                "symbol": [],
                "as_of_date": [],
                "start": [],
                "end": [],
            }
            for concept in duration_concepts + stock_concepts:
                empty_cols[concept] = []
            return pl.DataFrame(empty_cols)

        ttm_df = compute_ttm_long(pl.DataFrame(ttm_records), logger=self.logger, symbol=symbol)
        if len(ttm_df) == 0:
            empty_cols = {
                "symbol": [],
                "as_of_date": [],
                "start": [],
                "end": [],
            }
            for concept in duration_concepts + stock_concepts:
                empty_cols[concept] = []
            return pl.DataFrame(empty_cols)
        ttm_df = ttm_df.filter(
            (pl.col("as_of_date") >= start_dt.strftime("%Y-%m-%d"))
            & (pl.col("as_of_date") <= end_dt.strftime("%Y-%m-%d"))
            & (pl.col("end") >= start_dt.strftime("%Y-%m-%d"))
            & (pl.col("end") <= end_dt.strftime("%Y-%m-%d"))
        )
        if len(ttm_df) == 0:
            empty_cols = {
                "symbol": [],
                "as_of_date": [],
                "start": [],
                "end": [],
            }
            for concept in duration_concepts + stock_concepts:
                empty_cols[concept] = []
            return pl.DataFrame(empty_cols)

        ttm_wide = ttm_df.pivot(
            values="value",
            index=["symbol", "as_of_date", "start", "end"],
            on="concept",
            aggregate_function="first",
        )
        missing_duration = [c for c in duration_concepts if c not in ttm_wide.columns]
        if missing_duration:
            ttm_wide = ttm_wide.with_columns([pl.lit(None).alias(c) for c in missing_duration])

        base = ttm_wide.with_columns(
            pl.col("as_of_date").str.strptime(pl.Date, "%Y-%m-%d")
        ).sort("as_of_date")

        if not raw_records:
            for concept in stock_concepts:
                if concept not in base.columns:
                    base = base.with_columns(pl.lit(None).alias(concept))
            return base.with_columns(pl.col("as_of_date").dt.strftime("%Y-%m-%d"))

        raw_long = pl.DataFrame(raw_records)
        stock_long = raw_long.filter(pl.col("concept").is_in(stock_concepts)).with_columns(
            pl.col("as_of_date").str.strptime(pl.Date, "%Y-%m-%d")
        ).sort("as_of_date")

        if len(stock_long) > 0:
            stock_wide = (
                stock_long
                .group_by(["as_of_date", "concept"])
                .agg(pl.col("value").last())
                .pivot(
                    values="value",
                    index="as_of_date",
                    on="concept",
                    aggregate_function="first",
                )
                .sort("as_of_date")
            )
            base = base.join_asof(stock_wide, on="as_of_date", strategy="backward")

        missing_cols = [c for c in stock_concepts if c not in base.columns]
        if missing_cols:
            base = base.with_columns([pl.lit(None).alias(c) for c in missing_cols])

        return base.with_columns(pl.col("as_of_date").dt.strftime("%Y-%m-%d"))

    def collect_derived_long(
        self,
        cik: str,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> pl.DataFrame:
        """
        Compute derived metrics and return long-format data.

        Returns columns: [symbol, as_of_date, start, end, accn, form, fp, concept, value]
        """
        metrics_wide = self._build_metrics_wide(
            cik=cik,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            concepts=concepts,
            config_path=config_path
        )
        if len(metrics_wide) == 0:
            return pl.DataFrame()

        id_cols = ["symbol", "as_of_date", "start", "end"]
        value_cols = [c for c in metrics_wide.columns if c not in id_cols]
        long_input = metrics_wide.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            variable_name="concept",
            value_name="value",
        ).drop_nulls(subset=["value"])

        # Use TTM metadata for derived rows (accn/form/fp)
        ttm_df = self.collect_ttm_long_range(
            cik=cik,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            concepts=concepts,
            config_path=config_path
        )
        metadata_df = None
        if len(ttm_df) > 0:
            metadata_df = (
                ttm_df.select(["symbol", "as_of_date", "start", "end", "accn", "form", "fp"])
                .group_by(["symbol", "as_of_date", "start", "end"])
                .agg([pl.col("accn").last(), pl.col("form").last(), pl.col("fp").last()])
            )

        if metadata_df is not None:
            long_input = long_input.join(
                metadata_df,
                on=["symbol", "as_of_date", "start", "end"],
                how="left",
            )

        derived_long = compute_derived(long_input, logger=self.logger, symbol=symbol)
        if len(derived_long) == 0:
            return pl.DataFrame()

        start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        return derived_long.filter(
            (pl.col("as_of_date") >= start_dt.strftime("%Y-%m-%d"))
            & (pl.col("as_of_date") <= end_dt.strftime("%Y-%m-%d"))
            & (pl.col("end") >= start_dt.strftime("%Y-%m-%d"))
            & (pl.col("end") <= end_dt.strftime("%Y-%m-%d"))
        )

    def collect_daily_ticks_year(self, sym: str, year: int) -> pl.DataFrame:
        """
        Fetch daily ticks for entire year from appropriate source and return as Polars DataFrame.

        :param sym: Symbol in Alpaca format (e.g., 'BRK.B')
        :param year: Year to fetch data for
        :return: Polars DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if year < 2025:
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
                df = self.alpaca_ticks.get_daily_year(
                    symbol=sym,
                    year=year,
                    adjusted=True
                )
            except Exception as e:
                # Return empty DataFrame if fetch fails
                self.logger.warning(f"Failed to fetch {sym} for {year}: {e}")
                return pl.DataFrame()

        # Apply consistent formatting and rounding (single logic for both sources)
        if len(df) > 0:
            df = df.with_columns([
                pl.col('timestamp').cast(pl.Utf8),
                pl.col('open').round(4),
                pl.col('high').round(4),
                pl.col('low').round(4),
                pl.col('close').round(4),
                pl.col('volume').cast(pl.Int64)
            ]).sort('timestamp')

        return df

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
                    # Parse timestamp: UTC -> ET, remove timezone
                    # Use strptime with explicit format and timezone to handle 'Z' marker
                    pl.col('timestamp_utc')
                        .str.strptime(pl.Datetime('us', 'UTC'), format='%Y-%m-%dT%H:%M:%SZ')
                        .dt.convert_time_zone('America/New_York')
                        .dt.replace_time_zone(None)
                        .alias('timestamp'),
                    # Extract trade date for filtering by day (fast string slice)
                    pl.col('timestamp_utc')
                        .str.slice(0, 10)  # Just extract 'YYYY-MM-DD' from '2024-01-03T14:30:00Z'
                        .alias('trade_date'),
                    # Cast types (vectorized)
                    pl.col('open').cast(pl.Float64),
                    pl.col('high').cast(pl.Float64),
                    pl.col('low').cast(pl.Float64),
                    pl.col('close').cast(pl.Float64),
                    pl.col('volume').cast(pl.Int64),
                    pl.col('num_trades').cast(pl.Int64),
                    pl.col('vwap').cast(pl.Float64)
                ]).drop('timestamp_utc')

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
