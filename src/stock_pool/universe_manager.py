from pathlib import Path
import shutil
import time
import datetime as dt
import polars as pl
import logging
from collection.ticks import Ticks
from stock_pool.universe import fetch_all_stocks
from stock_pool.history_universe import get_hist_universe
from utils.logger import setup_logger

class UniverseManager:
    def __init__(self):
        self.recent_dir = Path("data/raw/ticks/recent")
        self.store_dir = Path("data/symbols")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.top_3000 = None
        log_dir = Path("data/logs/symbols")
        self.logger = setup_logger("symbols", log_dir, logging.INFO, console_output=True)

    def get_current_symbols(self, refresh=False) -> list[str]:
        """
        Get the current list of common stocks from Nasdaq Trader.
        """    
        csv_path = Path("data/symbols/stock_exchange.csv")
        if csv_path.exists() and not refresh:
            self.logger.info("Loading symbols from local CSV...")
            df = pl.read_csv(csv_path)
            symbols = df["Ticker"].to_list()
        else:
            pd_df = fetch_all_stocks()
            if pd_df is not None and not pd_df.empty:
                symbols = pd_df['Ticker'].tolist()
            else:
                raise ValueError("Failed to fetch symbols from SEC.")
                
        self.logger.info(f"Market Universe Size: {len(symbols)} tickers")
        return symbols
    
    def get_hist_symbols(self, year: int, month: int) -> list[str]:
        """
        Get history common stock list from CRSP database
        """
        symbols_df = get_hist_universe(year, month)
        symbols = symbols_df['Ticker'].to_list()

        if len(symbols) == 0:
            self.logger.warning(f"No symbols fetched for year {year}, month {month}")
        
        return symbols
    
    def remove_recent_data(self) -> None:
        if self.recent_dir.exists():
            shutil.rmtree(self.recent_dir)

    def fetch_recent_data(self, day: str, symbols: list[str], refresh=False) -> None:
        """
        Downloads 3-month daily history for ALL symbols using the efficient bulk fetcher.
        """
        if refresh:
            self.remove_recent_data()

        self.logger.info(f"Starting bulk fetch for {len(symbols)} symbols...")
        # Instantiate Ticks with a dummy symbol (symbol arg is ignored for bulk fetch)
        fetcher = Ticks(symbol="UNIVERSE")
        
        # Call the bulk method 
        fetcher.fetch_and_store_bulk(symbols, end_day=day)
        
        self.logger.info("Bulk fetch complete.")

    def filter_top_3000(self, day: str) -> list[str]:
        """
        Reads the cached data, calculates liquidity, and returns Top 3000.

        :param day: Date string in format "YYYY-MM-DD" to locate the correct data directory
        """
        try:
            # Extract year and month from day parameter
            day_dt = dt.datetime.strptime(day, '%Y-%m-%d')
            year = day_dt.strftime('%Y')
            month = day_dt.strftime('%m')

            # Build path to dated directory
            dated_dir = self.recent_dir / year / month

            if not dated_dir.exists():
                self.logger.error(f"Data directory does not exist: {dated_dir}")
                return []

            # Lazy Scan of all recent parquet files in the dated directory
            q = (
                pl.scan_parquet(dated_dir / "*.parquet")
                .group_by("symbol")
                .agg(
                    # Metric: Average Dollar Volume (Close * Volume)
                    (pl.col("close") * pl.col("volume")).mean().alias("avg_dollar_vol")
                )
                .filter(pl.col("avg_dollar_vol").is_not_null()) # Remove empty data
                .sort("avg_dollar_vol", descending=True)
                .head(3000)
            )
            df = q.collect()

            top_stock = df.row(0)
            bottom_stock = df.row(-1)
            self.logger.info(f"Top Liquid Stock: {top_stock[0]} (ADV: ${top_stock[1]:,.0f})")
            self.logger.info(f"Rank 3000 Stock:  {bottom_stock[0]} (ADV: ${bottom_stock[1]:,.0f})")

            result = df["symbol"].to_list()
            self.top_3000 = result
            return result

        except Exception as e:
            self.logger.error(f"Error calculating liquidity: {e}")
            return []
    
    # TODO: Refactor for upload
    def store_top_3000(self, day: str) -> None:
        """
        Store top 3000 symbols to dated directory.

        :param day: Date string in format "YYYY-MM-DD"
        """
        if not self.top_3000:
            self.top_3000 = self.filter_top_3000(day)

        # Extract year and month from day parameter
        day_dt = dt.datetime.strptime(day, '%Y-%m-%d')
        year = day_dt.strftime('%Y')
        month = day_dt.strftime('%m')

        # Create dated directory structure
        dated_dir = self.store_dir / year / month
        dated_dir.mkdir(parents=True, exist_ok=True)

        file_path = dated_dir / "universe_top3000.txt"
        with open(file_path, "w") as file:
            file.write("\n".join(self.top_3000))

        self.logger.info(f"Saved Top 3000 symbols to {file_path}")

    def run(self, day: str, refresh=False) -> None:
        """
        Run complete pipeline
        Output universe_top3000.txt at data/symbols/YYYY/MM/

        :param day: Date string in format "YYYY-MM-DD"
        :param refresh: Whether to refresh data from source
        """
        start = time.perf_counter()
        all_symbols = self.get_current_symbols(refresh=refresh)
        self.fetch_recent_data(day, all_symbols, refresh=refresh)
        self.store_top_3000(day)
        time_count = time.perf_counter() - start
        self.logger.info(f"Processing time: {time_count:.2f}s")

if __name__ == "__main__":
    um = UniverseManager()
    # Use today's date as default
    today = dt.datetime.now().strftime('%Y-%m-%d')
    um.run(day=today, refresh=True)