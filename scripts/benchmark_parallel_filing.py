"""
Benchmark parallel filing checks performance.

Measures execution time and request rate for different symbol counts.
"""

import time
import datetime as dt
from typing import List
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantdl.update.app import DailyUpdateApp


def setup_logging():
    """Setup logging to file and console."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # File handler
    file_handler = logging.FileHandler('benchmark_parallel_filing.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Console handler (only WARNING+)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_test_symbols(updater: DailyUpdateApp, count: int) -> List[str]:
    """Get test symbols from universe."""
    year = dt.date.today().year
    universe = updater.universe_manager.load_symbols_for_year(year, sym_type='alpaca')
    return universe[:count]


def benchmark_filing_check(updater: DailyUpdateApp, symbols: List[str], lookback_days: int = 7):
    """
    Benchmark filing check performance.

    Returns: (symbols_with_filings, elapsed_time, req_per_sec)
    """
    update_date = dt.date.today() - dt.timedelta(days=1)

    start_time = time.time()
    symbols_with_filings = updater.get_symbols_with_recent_filings(
        update_date=update_date,
        symbols=symbols,
        lookback_days=lookback_days
    )
    elapsed_time = time.time() - start_time

    # Calculate request rate (assumes ~1 request per symbol)
    req_per_sec = len(symbols) / elapsed_time if elapsed_time > 0 else 0

    return symbols_with_filings, elapsed_time, req_per_sec


def run_benchmark():
    """Run benchmarks with different symbol counts."""
    setup_logging()

    print("=" * 80)
    print("PARALLEL FILING CHECK BENCHMARK")
    print("=" * 80)
    print()

    # Initialize updater
    print("Initializing updater...")
    updater = DailyUpdateApp(config_path='configs/storage.yaml')

    # Test configurations
    test_sizes = [100, 1000, 5000]
    lookback_days = 7

    results = []

    for size in test_sizes:
        print(f"\n{'='*80}")
        print(f"BENCHMARK: {size} symbols")
        print('='*80)

        # Get test symbols
        print(f"Getting {size} test symbols...")
        symbols = get_test_symbols(updater, size)
        print(f"Got {len(symbols)} symbols")

        # Run benchmark
        print(f"Checking filings (lookback: {lookback_days} days)...")
        symbols_with_filings, elapsed_time, req_per_sec = benchmark_filing_check(
            updater, symbols, lookback_days
        )

        # Store results
        result = {
            'symbol_count': len(symbols),
            'symbols_with_filings': len(symbols_with_filings),
            'elapsed_time': elapsed_time,
            'req_per_sec': req_per_sec,
            'symbols_per_sec': len(symbols) / elapsed_time
        }
        results.append(result)

        # Print results
        print()
        print(f"Results:")
        print(f"  Symbols checked:       {result['symbol_count']}")
        print(f"  Symbols with filings:  {result['symbols_with_filings']}")
        print(f"  Elapsed time:          {result['elapsed_time']:.2f}s")
        print(f"  Request rate:          {result['req_per_sec']:.2f} req/sec")
        print(f"  Symbols per second:    {result['symbols_per_sec']:.2f} symbols/sec")
        print()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print()
    print(f"{'Symbols':<15} {'Time (s)':<15} {'Rate (req/s)':<20} {'Speedup':<10}")
    print('-' * 80)

    baseline_time = None
    for result in results:
        if baseline_time is None:
            baseline_time = result['elapsed_time'] / result['symbol_count']  # Time per symbol
            speedup = "baseline"
        else:
            time_per_symbol = result['elapsed_time'] / result['symbol_count']
            speedup = f"{baseline_time / time_per_symbol:.2f}x"

        print(
            f"{result['symbol_count']:<15} "
            f"{result['elapsed_time']:<15.2f} "
            f"{result['req_per_sec']:<20.2f} "
            f"{speedup:<10}"
        )

    print()
    print("Rate Limiting Analysis:")
    print(f"  Target: <=10 req/sec (SEC limit)")
    for result in results:
        status = "✓ OK" if result['req_per_sec'] <= 10 else "✗ OVER LIMIT"
        print(f"  {result['symbol_count']:>4} symbols: {result['req_per_sec']:>6.2f} req/sec {status}")

    print()
    print("=" * 80)
    print()


if __name__ == '__main__':
    run_benchmark()
