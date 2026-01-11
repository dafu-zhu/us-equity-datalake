from quantdl.storage.app import UploadApp

def main() -> None:
    app = UploadApp()
    try:
        app.run(
            start_year=2009,
            end_year=2025,
            max_workers=50,
            sleep_time=0.03,
            overwrite=True,
            daily_chunk_size=200,
            daily_sleep_time=0.2,
            minute_ticks_start_year=2017,
            run_daily_ticks=True,
            run_minute_ticks=False,
            run_fundamental=False, 
            run_derived_fundamental=False, 
            run_ttm_fundamental=False
        )
    finally:
        app.close()
