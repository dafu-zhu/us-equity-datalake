"""
Trading calendar utilities.

This module provides utilities for working with trading days and market calendars.
"""

import datetime as dt
from pathlib import Path
from typing import List
import polars as pl


class TradingCalendar:
    """
    Utility class for working with trading calendars.
    """

    def __init__(self, calendar_path: Path = Path("data/calendar/master.parquet")):
        """
        Initialize trading calendar.

        :param calendar_path: Path to master calendar parquet file
        """
        self.calendar_path = calendar_path

    def load_trading_days(self, year: int, month: int) -> List[str]:
        """
        Load trading days from master calendar for a specific month.

        :param year: Year to load trading days for
        :param month: Month to load trading days for (1-12)
        :return: List of trading days in 'YYYY-MM-DD' format
        """
        start_date = dt.date(year, month, 1)

        # Get last day of month
        if month == 12:
            end_date = dt.date(year, 12, 31)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)

        df = (
            pl.scan_parquet(self.calendar_path)
            .filter(pl.col('timestamp').is_between(start_date, end_date))
            .select('timestamp')
            .collect()
        )

        return [d.strftime('%Y-%m-%d') for d in df['timestamp'].to_list()]

    def load_trading_days_year(self, year: int) -> List[str]:
        """
        Load all trading days for an entire year.

        :param year: Year to load trading days for
        :return: List of trading days in 'YYYY-MM-DD' format
        """
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        df = (
            pl.scan_parquet(self.calendar_path)
            .filter(pl.col('timestamp').is_between(start_date, end_date))
            .select('timestamp')
            .collect()
        )

        return [d.strftime('%Y-%m-%d') for d in df['timestamp'].to_list()]

    def is_trading_day(self, date: dt.date) -> bool:
        """
        Check if a given date is a trading day.

        :param date: Date to check
        :return: True if trading day, False otherwise
        """
        df = (
            pl.scan_parquet(self.calendar_path)
            .filter(pl.col('timestamp') == date)
            .select('timestamp')
            .collect()
        )

        return len(df) > 0
