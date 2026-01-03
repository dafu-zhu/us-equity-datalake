import os
import logging
import requests
import time
import zoneinfo
import datetime as dt
import warnings
from typing import Tuple, List, Dict, Any
from dataclasses import asdict
from dotenv import load_dotenv
import polars as pl
from pathlib import Path

from collection.models import TickField, TickDataPoint
from utils.logger import LoggerFactory
from utils.mapping import align_calendar

load_dotenv()


class Ticks:
    def __init__(self) -> None:

        # Setup logger
        self.logger = LoggerFactory(
            log_dir='data/logs/ticks',
            level=logging.INFO,
            daily_rotation=True,
            console_output=False
        ).get_logger(name=f'collection.ticks')

        # Load Alpaca API key and secrets
        ALPACA_KEY = os.getenv("ALPACA_API_KEY")
        ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET
        }

        self.calendar_dir = Path("data/calendar")
        self.calendar_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_trade_day_range(trade_day: str) -> Tuple[str, str]:
        """
        Get the full UTC time range for a trading day (9:30 AM - 4:00 PM ET)

        :param trade_day: Specify trade day
        :param type:
            str: format "YYYY-MM-DD"
            datetime.date:
        :return: start, end
        """
        trade_date = dt.datetime.strptime(trade_day, "%Y-%m-%d").date()

        # Declare ET time zone
        eastern = zoneinfo.ZoneInfo("America/New_York")
        start_et = dt.datetime.combine(trade_date, dt.time(9, 30), tzinfo=eastern)
        end_et = dt.datetime.combine(trade_date, dt.time(16, 0), tzinfo=eastern)

        # Transform into UTC time
        start_str = start_et.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = end_et.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

        return start_str, end_str
    
    def recent_daily_ticks(self, symbols: List[str], end_day: str, window: int=90) -> Dict[str, pl.DataFrame]:
        """
        Fetches recent daily data for a list of symbols using Alpaca's multi-symbol API.

        Note: Alpaca's limit parameter applies to TOTAL data points across all symbols,
        not per symbol. This function handles pagination with next_page_token.

        :param symbols: List of symbols to fetch
        :param end_day: End date in format 'YYYY-MM-DD'
        :param window: Number of calendar days to fetch
        :return: Dictionary {symbol: DataFrame} with price and volume data
        """
        # Calculate start date (add buffer for weekends/holidays)
        end_dt = dt.datetime.strptime(end_day, '%Y-%m-%d')
        start_dt = end_dt - dt.timedelta(days=int(window))

        # Convert to UTC timestamps for Alpaca API
        start_utc = dt.datetime.combine(start_dt.date(), dt.time(0, 0), tzinfo=dt.timezone.utc)
        end_utc = dt.datetime.combine(end_dt.date(), dt.time(23, 59, 59), tzinfo=dt.timezone.utc)
        start_str = start_utc.isoformat().replace("+00:00", "Z")
        end_str = end_utc.isoformat().replace("+00:00", "Z")

        self.logger.info(f"Fetching {len(symbols)} symbols from {start_dt.date()} to {end_day} (window: {window} days)")

        # Prepare symbol list for API
        symbols_str = ','.join(symbols)

        # Collect all data using pagination
        all_data = {}
        page_token = None
        page_count = 0

        while True:
            page_count += 1
            url = "https://data.alpaca.markets/v2/stocks/bars"
            params = {
                "symbols": symbols_str,
                "timeframe": "1Day",
                "start": start_str,
                "end": end_str,
                "limit": 10000,
                "adjustment": "split",
                "feed": "sip",
                "sort": "asc"
            }

            if page_token:
                params["page_token"] = page_token

            try:
                response = requests.get(url, headers=self.headers, params=params)
                time.sleep(0.1)  # Rate limiting: 200/min = ~3/sec, use 10/sec to be safe

                if response.status_code != 200:
                    self.logger.error(f"API Error [{response.status_code}]: {response.text}")
                    break

                data = response.json()

                # Accumulate bars for each symbol
                if 'bars' in data and data['bars']:
                    for symbol, bars in data['bars'].items():
                        if symbol not in all_data:
                            all_data[symbol] = []
                        all_data[symbol].extend(bars)

                # Check for next page
                page_token = data.get('next_page_token')
                if not page_token:
                    self.logger.info(f"Completed fetching data in {page_count} page(s)")
                    break

            except Exception as e:
                self.logger.error(f"Request failed on page {page_count}: {e}")
                break

        # Convert collected data to DataFrames
        result_dict = {}
        for symbol, bars in all_data.items():
            if bars:
                try:
                    parsed_ticks = self.parse_ticks(bars)
                    ticks_data = [asdict(dp) for dp in parsed_ticks]

                    df = (
                        pl.DataFrame(ticks_data)
                        .with_columns([
                            pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S'),
                            pl.col('open').cast(pl.Float64),
                            pl.col('high').cast(pl.Float64),
                            pl.col('low').cast(pl.Float64),
                            pl.col('close').cast(pl.Float64),
                            pl.col('volume').cast(pl.Int64)
                        ])
                        .select(['timestamp', 'close', 'volume'])
                    )

                    result_dict[symbol] = df

                except Exception as e:
                    self.logger.error(f"Failed to process data for {symbol}: {e}")

        self.logger.info(f"Successfully fetched {len(result_dict)}/{len(symbols)} symbols")
        return result_dict

    def get_ticks(
        self,
        symbol: str,
        start_day: str,
        end_day: str,
        timeframe: str = "1Min",
        adjusted: bool = True
    ) -> List[dict]:
        """
        Get tick data from Alpaca API for one symbol, with specified timeframe and date range.

        :param symbols: A list of symbol names to fetch
        :param start_day: Start datetime in UTC (format: "2025-01-03T14:30:00Z")
        :param end_day: End datetime in UTC (format: "2025-01-03T21:00:00Z")
        :param timeframe: Alpaca timeframe (e.g., "1Min", "1Day", "1Hour")
        :return: List of dictionaries with OHLCV data
        Example:
        {"c": 184.25, "h": 185.88, "l": 183.43, "n": 656956, "o": 184.22, "t": "2024-01-03T05:00:00Z", "v": 58418916, "vw": 184.319693}
        """
        url = f"https://data.alpaca.markets/v2/stocks/bars"
        params = {
            "symbols": str(symbol).upper(),
            "timeframe": timeframe,
            "start": start_day,
            "end": end_day,
            "limit": 10000,  # Max
            "adjustment": "split",
            "feed": "sip",
            "sort": "asc"
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)

            # Rate limit: 200/m or 10/s
            time.sleep(0.03)

            if response.status_code != 200:
                self.logger.error(f"API Error [{response.status_code}] for {symbol}: {response.text}")
                return []

            data = response.json()

            if 'bars' not in data or data['bars'] is None:
                return []

            if symbol not in data['bars']:
                return []

            return data['bars'][symbol]

        except Exception as e:
            self.logger.error(f"Request failed for {symbol}: {e}")
            return []

    def get_minute(self, symbol, trade_day: str) -> List[Dict[str, Any]]:
        """
        Get minute-level OHLCV data for a specific trading day.

        :param trade_day: Trading day (format: "YYYY-MM-DD" or date object)
        :return: List of dictionaries with OHLCV data
        """
        start, end = self.get_trade_day_range(trade_day)
        return self.get_ticks(symbol, start, end, "1Min")

    def get_daily(self, symbol: str, year: int, month: int, adjusted: bool=True) -> List[Dict[str, Any]]:
        """
        Get daily OHLCV data for a specific month with adjusted prices.

        :param symbol: Stock symbol
        :param year: Year (e.g., 2024)
        :param month: Month (1-12)
        :param adjusted: If True, apply split adjustments (default: True)
        :return: List of dictionaries with OHLCV data
        """
        # Calculate month range
        start_date = dt.date(year, month, 1)
        if month == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        # Convert to UTC timestamps for API
        start_utc = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=dt.timezone.utc)
        end_utc = dt.datetime.combine(end_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc)
        start = start_utc.isoformat().replace("+00:00", "Z")
        end = end_utc.isoformat().replace("+00:00", "Z")

        result = self.get_ticks(symbol, start, end, "1Day", adjusted=adjusted)

        return result

    @staticmethod
    def parse_ticks(ticks: List[Dict[str, Any]]) -> List[TickDataPoint]:
        """
        Parse raw tick data from Alpaca API into TickDataPoint objects.
        Converts timestamps from UTC to Eastern Time (timezone-naive).

        :param ticks: List of dictionaries with OHLCV data
        :return: List of TickDataPoint objects
        """
        eastern = zoneinfo.ZoneInfo("America/New_York")
        datapoints = []

        for tick in ticks:
            if not tick:
                raise ValueError(f"tick is empty before parsing, get {tick}")
            
            # Parse UTC timestamp and convert to ET
            timestamp_utc = dt.datetime.fromisoformat(
                tick[TickField.TIMESTAMP.value].replace('Z', '+00:00')
            )
            # Convert to Eastern Time and remove timezone info for storage
            timestamp_et = timestamp_utc.astimezone(eastern).replace(tzinfo=None)

            dp = TickDataPoint(
                timestamp=timestamp_et.isoformat(),
                open=tick[TickField.OPEN.value],
                high=tick[TickField.HIGH.value],
                low=tick[TickField.LOW.value],
                close=tick[TickField.CLOSE.value],
                volume=tick[TickField.VOLUME.value],
                num_trades=tick[TickField.NUM_TRADES.value],
                vwap=tick[TickField.VWAP.value]
            )
            datapoints.append(dp)

        return datapoints
    
    def collect_daily_ticks(self, symbol: str, year: int, month: int, adjusted: bool=True) -> List[Dict[str, Any]]:
        """
        Collect daily ticks for a specific month and return as list of dicts (JSON format), align with trading days.

        :param symbol: Stock symbol
        :param year: Year (e.g., 2024)
        :param month: Month (1-12)
        :param adjusted: If True, apply split adjustments (default: True)
        :return: List of dictionaries with daily OHLCV data
        """
        ticks = self.get_daily(symbol, year, month, adjusted)
        parsed_ticks: List[TickDataPoint] = self.parse_ticks(ticks)

        # Transform dataclass to dictionaries
        ticks_data = [asdict(dp) for dp in parsed_ticks]

        # Define date range for the month
        start_date = dt.date(year, month, 1)
        if month == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        # Merge with master calendar
        calendar_path = self.calendar_dir / "master.parquet"
        result = align_calendar(ticks_data, start_date, end_date, calendar_path)

        return result
    
    def collect_minute_ticks(self, symbol: str, trade_day: str) -> pl.DataFrame:
        """
        Given a trade day, collect the minute level tick data with dataframe

        :param symbol: Stock symbol
        :param trade_day: In format 'YYYY-MM-DD'
        :return: Polars DataFrame with minute-level OHLCV data
        """
        ticks = self.get_minute(symbol=symbol, trade_day=trade_day)
        parsed_ticks: List[TickDataPoint] = self.parse_ticks(ticks)

        # Convert datapoints to dictionaries
        ticks_data = [asdict(dp) for dp in parsed_ticks]

        # Handle empty data case
        if not ticks_data:
            self.logger.warning(f"No data available for {symbol} on {trade_day}")
            # Return empty DataFrame with correct schema
            df = pl.DataFrame({
                'timestamp': [],
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': [],
                'num_trades': [],
                'vwap': []
            }).with_columns([
                pl.col('timestamp').cast(pl.Datetime),
                pl.col('open').cast(pl.Float64),
                pl.col('high').cast(pl.Float64),
                pl.col('low').cast(pl.Float64),
                pl.col('close').cast(pl.Float64),
                pl.col('volume').cast(pl.Int64),
                pl.col('num_trades').cast(pl.Int64),
                pl.col('vwap').cast(pl.Float64)
            ])
            return df

        # Create DataFrame with appropriate schema based on storage type
        df = pl.DataFrame(ticks_data).with_columns([
            pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S'),
            pl.col('open').cast(pl.Float64),
            pl.col('high').cast(pl.Float64),
            pl.col('low').cast(pl.Float64),
            pl.col('close').cast(pl.Float64),
            pl.col('volume').cast(pl.Int64),
            pl.col('num_trades').cast(pl.Int64),
            pl.col('vwap').cast(pl.Float64)
        ])

        return df



if __name__ == "__main__":
    ticks = Ticks()

    # Example 1: Fetch recent daily ticks for multiple symbols (returns dict of DataFrames)
    print("=" * 70)
    print("Example 1: recent_daily_ticks")
    print("=" * 70)

    symbols = ["AAPL", "MSFT", "GOOGL"]
    end_day = "2024-12-31"
    window = 90  # 90 calendar days (~3 months)

    print(f"Fetching recent data for {len(symbols)} symbols...")
    recent_data = ticks.recent_daily_ticks(symbols=symbols, end_day=end_day, window=window)

    print(f"\nSuccessfully fetched {len(recent_data)} symbols")
    for symbol, df in recent_data.items():
        print(f"\n{symbol}: {len(df)} records")
        print(df.head(3))

    # Example 2: Fetch daily ticks for a full year (returns list of dictionaries)
    print("\n" + "=" * 70)
    print("Example 2: collect_daily_ticks")
    print("=" * 70)

    symbol = "AAPL"
    year = 2024
    month = 12

    print(f"Fetching daily ticks for {symbol} in {year}-{month:02d}...")
    daily_ticks = ticks.collect_daily_ticks(symbol=symbol, year=year, month=month, adjusted=True)

    print(f"\nTotal records: {len(daily_ticks)}")
    print("\nFirst 3 records:")
    print(daily_ticks[:3])

    # Example 3: Fetch minute ticks for a specific day (returns DataFrame)
    print("\n" + "=" * 70)
    print("Example 3: collect_minute_ticks")
    print("=" * 70)

    symbol = "AAPL"
    trade_day = "2024-12-31"

    print(f"Fetching minute ticks for {symbol} on {trade_day}...")
    minute_df = ticks.collect_minute_ticks(symbol=symbol, trade_day=trade_day)

    print(f"\nTotal minutes: {len(minute_df)}")
    print("\nFirst 5 minutes:")
    print(minute_df.head(5))
    print("\nLast 5 minutes:")
    print(minute_df.tail(5))