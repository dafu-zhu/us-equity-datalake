import argparse

from quantdl.storage.app import UploadApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data lake upload workflows")
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (minute ticks)")

    parser.add_argument("--run-fundamental", action="store_true")
    parser.add_argument("--run-derived-fundamental", action="store_true")
    parser.add_argument("--run-ttm-fundamental", action="store_true")
    parser.add_argument("--run-daily-ticks", action="store_true")
    parser.add_argument("--run-minute-ticks", action="store_true")
    parser.add_argument("--run-top-3000", action="store_true")
    parser.add_argument("--run-all", action="store_true")

    parser.add_argument("--alpaca-start-year", type=int, default=2025)
    parser.add_argument("--minute-start-year", type=int, default=2017)
    parser.add_argument("--daily-chunk-size", type=int, default=200)
    parser.add_argument("--daily-sleep-time", type=float, default=0.2)

    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--minute-workers", type=int, default=50)
    parser.add_argument("--minute-chunk-size", type=int, default=30)
    parser.add_argument("--minute-sleep-time", type=float, default=0.02)

    args = parser.parse_args()

    app = UploadApp(
        alpaca_start_year=args.alpaca_start_year
    )
    try:
        app.run(
            start_year=args.start_year,
            end_year=args.end_year,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
            resume=args.resume,
            chunk_size=args.minute_chunk_size,
            sleep_time=args.minute_sleep_time,
            daily_chunk_size=args.daily_chunk_size,
            daily_sleep_time=args.daily_sleep_time,
            minute_ticks_start_year=args.minute_start_year,
            run_fundamental=args.run_fundamental,
            run_derived_fundamental=args.run_derived_fundamental,
            run_ttm_fundamental=args.run_ttm_fundamental,
            run_daily_ticks=args.run_daily_ticks,
            run_minute_ticks=args.run_minute_ticks,
            run_top_3000=args.run_top_3000,
            run_all=args.run_all
        )
    finally:
        app.close()