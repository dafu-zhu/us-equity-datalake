import datetime as dt
from quantdl.update.app import DailyUpdateApp

def main():
    """Main entry point for daily update."""
    import argparse

    parser = argparse.ArgumentParser(description="Run daily data lake update")
    parser.add_argument(
        '--date',
        type=str,
        help='Target date in YYYY-MM-DD format (default: yesterday)'
    )
    parser.add_argument(
        '--no-ticks',
        action='store_true',
        help='Skip ticks data update'
    )
    parser.add_argument(
        '--no-fundamentals',
        action='store_true',
        help='Skip fundamental data update'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=7,
        help='Days to look back for EDGAR filings (default: 7)'
    )

    args = parser.parse_args()

    # Parse target date
    if args.date:
        target_date = dt.datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target_date = None  # Will default to yesterday

    # Run update
    app = DailyUpdateApp()
    app.run_daily_update(
        target_date=target_date,
        update_ticks=not args.no_ticks,
        update_fundamentals=not args.no_fundamentals,
        fundamental_lookback_days=args.lookback
    )


if __name__ == "__main__":
    main()
