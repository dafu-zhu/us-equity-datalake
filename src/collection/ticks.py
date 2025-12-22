import os
import requests
import time
import zoneinfo
import datetime as dt
from typing import Tuple, List
from dataclasses import asdict
from dotenv import load_dotenv
import polars as pl
from pathlib import Path

from collection.models import TickField, TickDataPoint

load_dotenv()

class Ticks:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

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
        self.daily_ticks_df = None
        self.minute_ticks_df = None
        
        # Store path
        self.daily_dir = Path("data/raw/ticks/daily")
        self.minute_dir = Path("data/raw/ticks/minute")
        self.recent_dir = Path("data/raw/ticks/recent")
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.minute_dir.mkdir(parents=True, exist_ok=True)
        self.recent_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_trade_day_range(trade_day: str | dt.date) -> Tuple[str]:
        """
        Get the full UTC time range for a trading day (9:30 AM - 4:00 PM ET)

        :param trade_day: Specify trade day
        :param type:
            str: format "YYYY-MM-DD"
            datetime.date:
        :return: start, end
        """
        # Parse str day into dt.date object
        if isinstance(trade_day, str):
            trade_day = dt.datetime.strptime(trade_day, "%Y-%m-%d").date()

        # Declare ET time zone
        eastern = zoneinfo.ZoneInfo("America/New_York")
        start_et = dt.datetime.combine(trade_day, dt.time(9, 30), tzinfo=eastern)
        end_et = dt.datetime.combine(trade_day, dt.time(16, 0), tzinfo=eastern)

        # Transform into UTC time
        start_str = start_et.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        end_str = end_et.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")

        return start_str, end_str

    @staticmethod
    def get_year_range(year: str | int) -> Tuple[str]:
        """
        Get the UTC time range for a full year (Jan 1 - Dec 31)

        :param year: Year as string or integer
        :return: start, end in UTC format
        """
        if isinstance(year, str):
            year = int(year)

        # Create start and end dates in UTC
        start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
        end = dt.datetime(year, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)

        start_str = start.isoformat().replace("+00:00", "Z")
        end_str = end.isoformat().replace("+00:00", "Z")

        return start_str, end_str

    def get_ticks(
        self,
        start: str,
        end: str,
        timeframe: str = "1Min"
    ) -> List[dict]:
        """
        Get tick data from Alpaca API for specified timeframe and date range.

        :param start: Start datetime in UTC (format: "2025-01-03T14:30:00Z")
        :param end: End datetime in UTC (format: "2025-01-03T21:00:00Z")
        :param timeframe: Alpaca timeframe (e.g., "1Min", "1Day", "1Hour")
        :return: List of dictionaries with OHLCV data
        Example:
        {"c": 184.25, "h": 185.88, "l": 183.43, "n": 656956, "o": 184.22, "t": "2024-01-03T05:00:00Z", "v": 58418916, "vw": 184.319693}
        """
        symbol = str(self.symbol).upper()
        url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={symbol}&timeframe={timeframe}&start={start}&end={end}&limit=10000&adjustment=raw&feed=sip&sort=asc"

        try:
            response = requests.get(url, headers=self.headers)

            # Rate limit: 200/m or 10/s
            time.sleep(0.1)

            # 1. Check HTTP Status Code
            if response.status_code != 200:
                print(f"API Error [{response.status_code}] for {symbol}: {response.text}")
                return []

            data = response.json()

            # 2. Check if 'bars' key exists and is not None
            if 'bars' not in data or data['bars'] is None:
                return []
            
            # 3. Check if the specific symbol exists in the bars
            if symbol not in data['bars']:
                return []

            return data['bars'][symbol]
        
        except Exception as e:
            print(f"Request failed for {symbol}: {e}")
            return []

    def get_minute(self, trade_day: str | dt.date) -> List[dict]:
        """
        Get minute-level OHLCV data for a specific trading day.

        :param trade_day: Trading day (format: "YYYY-MM-DD" or date object)
        :return: List of dictionaries with OHLCV data
        """
        start, end = self.get_trade_day_range(trade_day)
        return self.get_ticks(start, end, timeframe="1Min")

    def get_daily(self, year: str | int) -> List[dict]:
        """
        Get daily OHLCV data for a full year.

        :param year: Year as string or integer (e.g., "2024" or 2024)
        :return: List of dictionaries with OHLCV data
        """
        start, end = self.get_year_range(year)
        return self.get_ticks(start, end, timeframe="1Day")

    def parse_ticks(self, ticks: List[dict]) -> List[TickDataPoint]:
        """
        Parse raw tick data from Alpaca API into TickDataPoint objects.
        Converts timestamps from UTC to Eastern Time (timezone-naive).

        :param ticks: List of dictionaries with OHLCV data
        :return: List of TickDataPoint objects
        """
        eastern = zoneinfo.ZoneInfo("America/New_York")
        datapoints = []

        for tick in ticks:
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
    
    def collect_daily_ticks(self, year: int) -> pl.DataFrame:
        """
        Given a year, collect daily ticks of the symbol for that year with a dataframe
        
        :param year: Specify trade year
        :type year: int
        """
        parsed_ticks: List[TickDataPoint] = self.parse_ticks(self.get_daily(year=year))
        
        # Define date range
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        # Transform dataclass to dictionaries
        ticks_data = [asdict(dp) for dp in parsed_ticks]

        # Merge with master calendar
        calendar_path = self.calendar_dir / "master.parquet"
        calendar_lf = (
            pl.scan_parquet(calendar_path)
            .filter(pl.col('Date').is_between(start_date, end_date))
            .sort('Date')
            .lazy()
        )
        ticks_lf = (
            pl.DataFrame(ticks_data)
            .with_columns([
                pl.col('timestamp').str.to_date(format='%Y-%m-%dT%H:%M:%S').alias('Date'),
                pl.col('open').cast(pl.Float64),
                pl.col('high').cast(pl.Float64),
                pl.col('low').cast(pl.Float64),
                pl.col('close').cast(pl.Float64),
                pl.col('volume').cast(pl.Int64),
                pl.col('num_trades').cast(pl.Int64),
                pl.col('vwap').cast(pl.Float64)
            ])
            .drop("timestamp")
            .lazy()
        )
        calendar_lf = calendar_lf.join_asof(ticks_lf, on='Date')

        result = calendar_lf.collect()
        self.daily_ticks_df = result
        
        return result
    
    def store_daily_ticks(self, year: int) -> None:
        """
        Store dataframe result from collect_daily_ticks with parquet
        
        :param year: Specify trade year
        :type year: int
        """
        if not isinstance(self.daily_ticks_df, pl.DataFrame):
            self.daily_ticks_df: pl.DataFrame = self.collect_daily_ticks(year=year)

        # Define storage path
        dir_path = self.daily_dir / f"{self.symbol}/{year}"
        dir_path.mkdir(parents=True, exist_ok=True)

        # Write to Parquet file
        file_path = dir_path / 'ticks.parquet'
        self.daily_ticks_df.write_parquet(file_path, compression='zstd')

        print(f"Stored {len(self.daily_ticks_df)} daily ticks to {file_path}")
    
    def collect_minute_ticks(self, trade_day: str) -> pl.DataFrame:
        """
        Given a trade day, collect the minute level tick data with dataframe
        
        :param trade_day: In format 'YYYY-MM-DD'
        :type trade_day: str
        """
        parsed_ticks: List[TickDataPoint] = self.parse_ticks(self.get_minute(trade_day=trade_day))

        # Convert datapoints to dictionaries
        ticks_data = [asdict(dp) for dp in parsed_ticks]

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

        # No need to merge with master cal
        self.minute_ticks_df = df
        
        return df

    def store_minute_ticks(self, trade_day: str) -> None:
        """
        Store dataframe result from collect_minute_ticks with parquet
        
        :param trade_day: In format 'YYYY-MM-DD'
        :type trade_day: str
        """
        if not isinstance(self.minute_ticks_df, pl.DataFrame):
            self.minute_ticks_df: pl.DataFrame = self.collect_minute_ticks(trade_day=trade_day)

        # Build directory path based on storage type
        date_obj = dt.datetime.strptime(trade_day, '%Y-%m-%d').date()
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        dir_path = self.minute_dir / f"{self.symbol}/{year}/{month}/{day}"
        dir_path.mkdir(parents=True, exist_ok=True)

        # Write to Parquet file
        file_path = dir_path / 'ticks.parquet'
        self.minute_ticks_df.write_parquet(file_path, compression='zstd')

        print(f"Stored {len(self.minute_ticks_df)} minute ticks to {file_path}")

    def update_liquidity_cache(self) -> bool:
        """
        Fetches the last 3 months of daily data and saves it to a 
        temporary 'recent' folder for liquidity scoring.

        If success, return True; otherwise return False
        """
        try:
            # Define 3-month window (UTC)
            end_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1) 
            start_dt = end_dt - dt.timedelta(weeks=12) 
            
            start_str = start_dt.isoformat().replace("+00:00", "Z")
            end_str = end_dt.isoformat().replace("+00:00", "Z")

            # Fetch Data
            raw_bars = self.get_ticks(start_str, end_str, timeframe="1Day")
            
            if not raw_bars:
                return False

            # Parse and Format for Liquidity Calculation
            parsed_bars = self.parse_ticks(raw_bars)
            bars_data = [asdict(dp) for dp in parsed_bars]
            
            # Create DataFrame
            df = (
                pl.LazyFrame(bars_data)
                .with_columns([
                    pl.col("timestamp").str.to_date(format='%Y-%m-%dT%H:%M:%S').alias("Date"),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Int64),
                    pl.lit(self.symbol).alias("symbol")
                ])
                .select(["Date", "symbol", "close", "volume"])
            ).collect()

            # Save to "Recent"
            df.write_parquet(self.recent_dir / f"{self.symbol}.parquet")
            return True

        except Exception as e:
            print(f"Error fetching recent data for {self.symbol}: {e}")
            return False


if __name__ == "__main__":
    SYMBOL = "AAPL"

    ticks = Ticks(SYMBOL)

    # Example 1: Fetch and store minute data for a specific day
    print("=" * 50)
    print("Fetching minute data...")
    trade_day = "2025-01-03"
    minutes = ticks.collect_minute_ticks(trade_day)
    ticks.store_minute_ticks(trade_day)
    minute_df = pl.read_parquet(ticks.minute_dir/"AAPL/2025/01/03/ticks.parquet")
    print(minute_df)

    # Example 2: Fetch and store daily data for a full year
    print("\n" + "=" * 50)
    print("Fetching daily data...")
    year = 2024
    daily_bars = ticks.collect_daily_ticks(year)
    ticks.store_daily_ticks(year)
    daily_df = pl.read_parquet(ticks.daily_dir/"AAPL/2024/ticks.parquet")
    print(daily_df)