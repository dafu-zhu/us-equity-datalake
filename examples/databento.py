from quantdl.storage.upload_app import UploadApp

app = UploadApp()
try:
    df, reason = app.data_collectors.collect_derived_long(
        cik="0000320193",  # AAPL
        start_date="2015-01-01",
        end_date="2018-12-31",
        symbol="AAPL",
    )
    if len(df) == 0:
        print(f"Derived empty: {reason}")
    else:
        roa_df = (
            df.filter(df["metric"] == "roa")
            .sort(["symbol", "as_of_date"])
        )
        print("Derived ROA:")
        print(roa_df)

        ttm_df = app.data_collectors.collect_ttm_long_range(
            cik="0000320193",  # AAPL
            start_date="2015-01-01",
            end_date="2018-12-31",
            symbol="AAPL",
        )
        ttm_inputs = (
            ttm_df
            .filter(ttm_df["concept"].is_in(["net_inc", "ta"]))
            .sort(["concept", "as_of_date"])
        )
        print("TTM inputs (net_inc, ta):")
        print(ttm_inputs)

        raw_df = app.data_collectors.collect_fundamental_long(
            cik="0000320193",  # AAPL
            start_date="2015-01-01",
            end_date="2018-12-31",
            symbol="AAPL",
            concepts=["net_inc", "ta"],
        )
        raw_inputs = (
            raw_df
            .filter(raw_df["concept"].is_in(["net_inc", "ta"]))
            .sort(["concept", "as_of_date"])
        )
        print("Raw inputs (net_inc, ta):")
        print(raw_inputs)
finally:
    app.close()
