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
from collection.fundamental import Fundamental


class DataCollectors:
    """
    Handles data collection from various market data sources.
    """

    def __init__(
        self,
        crsp_ticks,
        alpaca_ticks,
        alpaca_headers: dict,
        logger: logging.Logger
    ):
        """
        Initialize data collectors.

        :param crsp_ticks: CRSPDailyTicks instance
        :param alpaca_ticks: Alpaca Ticks instance
        :param alpaca_headers: Headers for Alpaca API requests
        :param logger: Logger instance
        """
        self.crsp_ticks = crsp_ticks
        self.alpaca_ticks = alpaca_ticks
        self.alpaca_headers = alpaca_headers
        self.logger = logger

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

    def collect_fundamental_year(
        self,
        cik: str,
        year: int,
        symbol: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        config_path: Optional[Path] = None
    ) -> pl.DataFrame:
        """
        Fetch fundamental data for a specific year using approved_mapping.yaml concepts.
        Returns only actual filing dates (quarterly data points) without forward-filling.

        :param cik: Company CIK number
        :param year: Year to fetch data for
        :param symbol: Optional symbol for logging (e.g., 'AAPL')
        :param concepts: Optional list of concepts to fetch. If None, fetches all concepts from config.
        :param config_path: Optional path to approved_mapping.yaml (defaults to configs/approved_mapping.yaml)
        :return: Polars DataFrame with columns [timestamp, concept1, concept2, ...]
                 Returns empty DataFrame if no data available or error occurs

        Example:
            >>> collector = DataCollectors(...)
            >>> df = collector.collect_fundamental_year(cik='0001819994', year=2024, symbol='RKLB')
            >>> # Returns DataFrame with columns: timestamp, rev, net_inc, ta, tl, te, etc.
        """
        try:
            # Load concept mappings if not provided
            if concepts is None:
                if config_path is None:
                    config_path = Path("configs/approved_mapping.yaml")

                with open(config_path) as f:
                    mappings = yaml.safe_load(f)
                    concepts = list(mappings.keys())
                    self.logger.debug(f"Loaded {len(concepts)} concepts from {config_path}")

            # Create Fundamental instance
            fund = Fundamental(cik=cik, symbol=symbol)

            # Collect data for each concept
            fields_dict = {}
            concepts_found = []
            concepts_missing = []

            for concept in concepts:
                try:
                    dps = fund.get_concept_data(concept)
                    if dps:
                        fields_dict[concept] = fund.get_value_tuple(dps)
                        concepts_found.append(concept)
                    else:
                        # Add missing concept with empty list to ensure column exists with null values
                        fields_dict[concept] = []
                        concepts_missing.append(concept)
                except Exception as e:
                    self.logger.debug(f"Failed to extract concept '{concept}' for CIK {cik}: {e}")
                    # Add failed concept with empty list to ensure column exists with null values
                    fields_dict[concept] = []
                    concepts_missing.append(concept)

            # If no data found for any concept, return empty DataFrame
            if not concepts_found:
                self.logger.warning(f"No fundamental data found for CIK {cik} ({symbol}) in {year}")
                return pl.DataFrame()

            # Log coverage statistics
            self.logger.debug(
                f"CIK {cik} ({symbol}): {len(concepts_found)}/{len(concepts)} concepts available "
                f"({len(concepts_missing)} missing)"
            )

            # Aggregate data for the year (no forward-filling)
            start_day = f"{year}-01-01"
            end_day = f"{year}-12-31"
            df = fund.collect_fields_raw(start_day, end_day, fields_dict)

            # Convert timestamp to string for consistency
            if len(df) > 0 and 'timestamp' in df.columns:
                df = df.with_columns(
                    pl.col('timestamp').dt.strftime('%Y-%m-%d')
                )

            return df

        except Exception as e:
            self.logger.error(f"Failed to collect fundamental data for CIK {cik} ({symbol}) in {year}: {e}")
            return pl.DataFrame()
