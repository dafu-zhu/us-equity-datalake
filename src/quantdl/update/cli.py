import datetime as dt
import os

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
    parser.add_argument(
        '--no-wrds',
        action='store_true',
        help='Use WRDS-free mode (Nasdaq universe + SEC CIK mapping). '
             'Suitable for CI/CD environments where WRDS has IP restrictions.'
    )

    args = parser.parse_args()

    # Parse target date
    if args.date:
        target_date = dt.datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target_date = None  # Will default to yesterday

    # Auto-detect WRDS-free mode if credentials missing
    use_wrds_free = args.no_wrds
    if not use_wrds_free:
        wrds_user = os.getenv('WRDS_USERNAME')
        wrds_pass = os.getenv('WRDS_PASSWORD')
        if not wrds_user or not wrds_pass:
            print("WRDS credentials not found, using WRDS-free mode")
            use_wrds_free = True

    # Import appropriate app
    if use_wrds_free:
        from quantdl.update.app_no_wrds import DailyUpdateAppNoWRDS
        print("Running in WRDS-free mode (Nasdaq + SEC API)")
        app = DailyUpdateAppNoWRDS()
    else:
        from quantdl.update.app import DailyUpdateApp
        print("Running with WRDS connection")
        app = DailyUpdateApp()

    # Run update
    app.run_daily_update(
        target_date=target_date,
        update_ticks=not args.no_ticks,
        update_fundamentals=not args.no_fundamentals,
        fundamental_lookback_days=args.lookback
    )