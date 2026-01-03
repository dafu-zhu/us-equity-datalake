from pathlib import Path
import shutil
import time
import datetime as dt
import polars as pl
import logging
from typing import Dict
from collection.crsp_ticks import CRSPDailyTicks
from collection.alpaca_ticks import Ticks
from stock_pool.universe import fetch_all_stocks
from stock_pool.history_universe import get_hist_universe_nasdaq
from utils.logger import setup_logger

class UniverseManager:
    def __init__(self):
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
            pd_df = fetch_all_stocks(logger=self.logger)
            if pd_df is not None and not pd_df.empty:
                symbols = pd_df['Ticker'].tolist()
            else:
                raise ValueError("Failed to fetch symbols from Nasdaq Trader.")
                
        self.logger.info(f"Market Universe Size: {len(symbols)} tickers")
        return symbols
    
    def get_hist_symbols(self, day: str) -> list[str]:
        """
        Get history common stock list from CRSP database
        """
        symbols_df = get_hist_universe_nasdaq(day)
        symbols = symbols_df['Ticker'].to_list()

        if len(symbols) == 0:
            self.logger.warning(f"No symbols fetched for day {day}")
        
        return symbols

    def get_top_3000(self, day: str, symbols: list[str], source: str) -> list[str]:
        """
        Fetch recent data and calculate top 3000 most liquid stocks (in-memory).

        :param day: Date string in format "YYYY-MM-DD"
        :param symbols: List of symbols to analyze
        :param source: 'crsp' or 'alpaca'
        :return: List of top 3000 symbols ranked by average dollar volume
        """
        self.logger.info(f"Fetching recent data for {len(symbols)} symbols using {source}...")

        # Fetch recent data (returns Dict[str, pl.DataFrame])
        if source.lower() == 'crsp':
            fetcher = CRSPDailyTicks()
            recent_data = fetcher.recent_daily_ticks(symbols, end_day=day)
        elif source.lower() == 'alpaca':
            fetcher = Ticks()
            recent_data = fetcher.recent_daily_ticks(symbols, end_day=day)
        else:
            raise ValueError(f"Invalid source: {source}. Must be 'crsp' or 'alpaca'")

        self.logger.info(f"Data fetched for {len(recent_data)} symbols, calculating liquidity...")

        # Calculate average dollar volume for each symbol
        liquidity_data = []
        for symbol, df in recent_data.items():
            if len(df) > 0:
                # Calculate average dollar volume: avg(close * volume)
                avg_dollar_vol = (df['close'] * df['volume']).mean()
                liquidity_data.append({
                    'symbol': symbol,
                    'avg_dollar_vol': avg_dollar_vol
                })

        # Create DataFrame and rank by liquidity
        liquidity_df = (
            pl.DataFrame(liquidity_data)
            .filter(pl.col('avg_dollar_vol') > 1000)
            .sort('avg_dollar_vol', descending=True)
            .head(3000)
        )

        if len(liquidity_df) == 0:
            self.logger.error("No symbols passed liquidity filter")
            return []

        # Log top and bottom stocks
        top_stock = liquidity_df.row(0)
        bottom_stock = liquidity_df.row(-1)
        self.logger.info(f"Top Liquid Stock: {top_stock[0]} (ADV: ${top_stock[1]:,.0f})")
        self.logger.info(f"Rank {len(liquidity_df)} Stock: {bottom_stock[0]} (ADV: ${bottom_stock[1]:,.0f})")

        result = liquidity_df['symbol'].to_list()
        self.top_3000 = result
        return result

if __name__ == "__main__":
    print("=" * 70)
    print("Example: Get Top 3000 Most Liquid Stocks (Alpaca)")
    print("=" * 70)

    um = UniverseManager()
    day = "2026-01-02"
    source = "alpaca"  # Use Alpaca for recent data

    # Parse date
    date = dt.datetime.strptime(day, '%Y-%m-%d')
    year = date.year

    print(f"\nTarget date: {day}")
    print(f"Year: {year}")
    print(f"Data source: {source}")

    # Step 1: Get universe (historical or current)
    print("\n" + "-" * 70)
    print("Step 1: Fetching stock universe...")
    print("-" * 70)

    if year < 2025:
        all_symbols = um.get_hist_symbols(day)
        print(f"Historical universe loaded: {len(all_symbols)} symbols")
    else:
        all_symbols = um.get_current_symbols(refresh=False)
        print(f"Current universe loaded: {len(all_symbols)} symbols")

    print(f"First 10 symbols: {all_symbols[:10]}")

    # Step 2: Get top 3000 (fetch recent data + calculate liquidity in-memory)
    print("\n" + "-" * 70)
    print("Step 2: Fetching recent data and calculating top 3000...")
    print("-" * 70)

    start = time.perf_counter()
    top_3000 = um.get_top_3000(day, all_symbols, source)
    elapsed = time.perf_counter() - start

    print(f"\nTop 3000 calculation complete!")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Total symbols in top 3000: {len(top_3000)}")

    # Step 3: Display results
    print("\n" + "-" * 70)
    print("Step 3: Results Summary")
    print("-" * 70)

    print(f"\nTop 10 most liquid stocks:")
    for i, symbol in enumerate(top_3000[:10], 1):
        print(f"  {i:2}. {symbol}")

    print(f"\nBottom 10 (ranked 2991-3000):")
    for i, symbol in enumerate(top_3000[-10:], len(top_3000) - 9):
        print(f"  {i:4}. {symbol}")

    # Step 4: Statistics
    print("\n" + "-" * 70)
    print("Step 4: Statistics")
    print("-" * 70)

    universe_size = len(all_symbols)
    top_3000_size = len(top_3000)
    coverage_rate = (top_3000_size / universe_size * 100) if universe_size > 0 else 0

    print(f"\nOriginal universe size:     {universe_size:5}")
    print(f"Top 3000 size:              {top_3000_size:5}")
    print(f"Coverage rate:              {coverage_rate:5.1f}%")
    print(f"Filtered out (illiquid):    {universe_size - top_3000_size:5}")

    print("\n" + "=" * 70)