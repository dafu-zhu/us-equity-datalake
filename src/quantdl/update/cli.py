import datetime as dt
import os

MAX_BACKFILL_DAYS = 30


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
        '--backfill-from',
        type=str,
        help=f'Backfill from date (YYYY-MM-DD) to --date. Max {MAX_BACKFILL_DAYS} days.'
    )
    # Tick flags
    parser.add_argument(
        '--no-ticks',
        action='store_true',
        help='Skip all ticks updates (daily and minute)'
    )
    parser.add_argument(
        '--no-daily-ticks',
        action='store_true',
        help='Skip daily ticks update'
    )
    parser.add_argument(
        '--no-minute-ticks',
        action='store_true',
        help='Skip minute ticks update'
    )

    # Fundamental flags
    parser.add_argument(
        '--no-fundamental',
        action='store_true',
        help='Skip raw fundamental update'
    )
    parser.add_argument(
        '--no-ttm',
        action='store_true',
        help='Skip TTM fundamental update'
    )
    parser.add_argument(
        '--no-derived',
        action='store_true',
        help='Skip derived metrics update'
    )
    parser.add_argument(
        '--no-sentiment',
        action='store_true',
        help='Skip sentiment analysis update (FinBERT on MD&A)'
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
    parser.add_argument(
        '--no-top3000',
        action='store_true',
        help='Skip top 3000 universe refresh'
    )

    args = parser.parse_args()

    # Parse target date (end date for backfill range)
    if args.date:
        end_date = dt.datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        end_date = dt.date.today() - dt.timedelta(days=1)  # yesterday

    # Parse backfill start date
    if args.backfill_from:
        start_date = dt.datetime.strptime(args.backfill_from, '%Y-%m-%d').date()
        days_diff = (end_date - start_date).days
        if days_diff < 0:
            print(f"Error: --backfill-from ({start_date}) must be before --date ({end_date})")
            return
        if days_diff > MAX_BACKFILL_DAYS:
            print(f"Error: Backfill range ({days_diff} days) exceeds max ({MAX_BACKFILL_DAYS} days)")
            return
    else:
        start_date = end_date  # Single day update

    # Handle --no-ticks shorthand
    skip_daily_ticks = args.no_daily_ticks or args.no_ticks
    skip_minute_ticks = args.no_minute_ticks or args.no_ticks

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

    # Generate date range (sequential backfill)
    current_date = start_date
    dates_to_process = []
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += dt.timedelta(days=1)

    is_backfill = len(dates_to_process) > 1

    if is_backfill:
        print(f"Backfilling {len(dates_to_process)} days from {start_date} to {end_date}")

    # Run update for each date (ticks only during backfill)
    for target_date in dates_to_process:
        if is_backfill:
            print(f"\n{'='*60}")
            print(f"Processing {target_date} (ticks only)")
            print(f"{'='*60}")

        app.run_daily_update(
            target_date=target_date,
            update_daily_ticks=not skip_daily_ticks,
            update_minute_ticks=not skip_minute_ticks,
            # During backfill: skip fundamental/sentiment (run once at end)
            update_fundamental=not args.no_fundamental and not is_backfill,
            update_ttm=not args.no_ttm and not is_backfill,
            update_derived=not args.no_derived and not is_backfill,
            update_sentiment=not args.no_sentiment and not is_backfill,
            fundamental_lookback_days=args.lookback,
            update_top3000=not args.no_top3000,
        )

    # Backfill: run fundamental/TTM/derived/sentiment ONCE at end with full date range
    if is_backfill:
        run_fundamental = not args.no_fundamental
        run_ttm = not args.no_ttm
        run_derived = not args.no_derived
        run_sentiment = not args.no_sentiment

        if run_fundamental or run_ttm or run_derived or run_sentiment:
            backfill_lookback = (end_date - start_date).days + 1
            print(f"\n{'='*60}")
            print(f"Running fundamental/sentiment updates (lookback={backfill_lookback} days)")
            print(f"{'='*60}")

            app.run_daily_update(
                target_date=end_date,
                update_daily_ticks=False,
                update_minute_ticks=False,
                update_fundamental=run_fundamental,
                update_ttm=run_ttm,
                update_derived=run_derived,
                update_sentiment=run_sentiment,
                fundamental_lookback_days=backfill_lookback,
                update_top3000=False,
            )