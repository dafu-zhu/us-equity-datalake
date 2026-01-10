from quantdl.storage.upload_app import UploadApp

def main() -> None:
    app = UploadApp()
    try:
        app.run(
            start_year=2009,
            end_year=2025,
            max_workers=50,
            sleep_time=0.03,
            overwrite=True,
            run_fundamental=False,
            run_derived_fundamental=False,
            run_ttm_fundamental=True,
        )
    finally:
        app.close()