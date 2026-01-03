#!/usr/bin/env python3
"""
Generate top 3000 stock universes for each month from 2010-01 to 2026-01.

For each month, uses YYYY-MM-01 to fetch the top 3000 most liquid stocks
and stores them in data/symbols/YYYY/MM/universe_top3000.txt

Usage:
    python scripts/generate_monthly_top3000.py
"""

import sys
from pathlib import Path
import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path so we can import modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from stock_pool.universe_manager import UniverseManager
from utils.logger import setup_logger


def process_single_month(month_date, um, total_months, month_index):
    """
    Process a single month: get universe and calculate top 3000.

    :param month_date: Date object for the month
    :param um: UniverseManager instance
    :param total_months: Total number of months
    :param month_index: Index of this month (1-based)
    :return: Tuple (success, stats_update, log_message)
    """
    year = month_date.year
    month = month_date.month
    fetch_date_str = f"{year}-{month:02d}-01"

    # Determine output path
    output_dir = Path(f"data/symbols/{year}/{month:02d}")
    output_file = output_dir / "universe_top3000.txt"

    # Skip if file already exists
    if output_file.exists():
        return (True, 'skipped', f"[{month_index}/{total_months}] {year}-{month:02d}: SKIPPED (file exists)")

    try:
        # Step 1: Get stock universe
        if year < 2025:
            all_symbols = um.get_hist_symbols(fetch_date_str)
            universe_type = "historical"
        else:
            all_symbols = um.get_current_symbols(refresh=False)
            universe_type = "current"

        # Step 2: Get top 3000 based on liquidity (includes I/O in recent_daily_ticks)
        source = 'crsp' if year < 2025 else 'alpaca'
        top_3000 = um.get_top_3000(fetch_date_str, all_symbols, source)

        if len(top_3000) == 0:
            return (False, 'failed', f"[{month_index}/{total_months}] {year}-{month:02d}: FAILED (no symbols)")

        # Step 3: Write to file
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for symbol in top_3000:
                f.write(f"{symbol}\n")

        msg = (f"[{month_index}/{total_months}] {year}-{month:02d}: SUCCESS "
               f"({len(all_symbols)} {universe_type} → {len(top_3000)} top, source: {source})")
        return (True, 'successful', msg)

    except Exception as e:
        return (False, 'failed', f"[{month_index}/{total_months}] {year}-{month:02d}: ERROR - {str(e)}")


def main():
    # Setup logging
    log_dir = Path("data/logs/universe_generation")
    logger = setup_logger(
        "universe_top3000_generator",
        log_dir,
        logging.INFO,
        console_output=True
    )

    logger.info("=" * 80)
    logger.info("Top 3000 Universe Generation Script")
    logger.info("=" * 80)
    logger.info("Generating monthly top 3000 universes from 2010-01 to 2026-01")
    logger.info("Using YYYY-MM-01 date for each month's universe")
    logger.info("")

    # Initialize UniverseManager
    um = UniverseManager()

    # Define date range: 2010-01-01 to 2026-01-01
    start_date = dt.date(2010, 1, 1)
    end_date = dt.date(2026, 1, 1)

    # Generate list of all months in the range
    current_date = start_date
    months = []
    while current_date <= end_date:
        months.append(current_date)
        # Move to next month
        if current_date.month == 12:
            current_date = dt.date(current_date.year + 1, 1, 1)
        else:
            current_date = dt.date(current_date.year, current_date.month + 1, 1)

    logger.info(f"Total months to process: {len(months)}")
    logger.info("")

    # Track statistics
    total_months = len(months)
    successful = 0
    failed = 0
    skipped = 0

    # Process months in parallel using ThreadPoolExecutor
    # Since recent_daily_ticks is I/O bound, threading will improve performance
    MAX_WORKERS = 10

    logger.info(f"Using {MAX_WORKERS} parallel workers for data fetching")
    logger.info("")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all months for processing
        future_to_month = {
            executor.submit(process_single_month, month_date, um, total_months, idx): (idx, month_date)
            for idx, month_date in enumerate(months, 1)
        }

        # Process results as they complete
        for future in as_completed(future_to_month):
            idx, month_date = future_to_month[future]
            try:
                success, stat_type, message = future.result()

                # Update statistics
                if stat_type == 'successful':
                    successful += 1
                elif stat_type == 'failed':
                    failed += 1
                elif stat_type == 'skipped':
                    skipped += 1

                # Log the result
                logger.info(message)

            except Exception as e:
                year = month_date.year
                month = month_date.month
                logger.error(f"[{idx}/{total_months}] {year}-{month:02d}: EXCEPTION - {str(e)}", exc_info=True)
                failed += 1

    # Final summary
    logger.info("=" * 80)
    logger.info("Generation Complete!")
    logger.info("=" * 80)
    logger.info(f"Total months:  {total_months}")
    logger.info(f"Successful:    {successful}")
    logger.info(f"Skipped:       {skipped}")
    logger.info(f"Failed:        {failed}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
