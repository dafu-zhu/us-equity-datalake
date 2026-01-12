import requests
import polars as pl
import datetime as dt
from typing import List, Dict, Any
from pathlib import Path


def symbol_cik_mapping() -> dict:
        # Download the official ticker-CIK mapping file
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {'User-Agent': 'YourName/YourEmail@domain.com'} # SEC requires a proper User-Agent
        response = requests.get(url, headers=headers)

        symbol_cik = {}
        if response.status_code == 200:
            data = response.json()
            for key, value in data.items():
                symbol = value.get('ticker')
                if symbol:
                    symbol_cik[symbol] = value.get('cik_str')
        
        return symbol_cik

def align_calendar(
        data: List[Dict[str, Any]], 
        start_date: dt.date, 
        end_date: dt.date, 
        calendar_path: Path
    ) -> List[Dict[str, Any]]:
    # Align with trading days
    calendar_lf = (
        pl.scan_parquet(calendar_path)
        .filter(pl.col('timestamp').is_between(start_date, end_date))
        .sort('timestamp')
        .lazy()
    )

    # Handle the case when empty data is passed
    if not data:
        result = (
            calendar_lf.collect()
            .with_columns([
                pl.col('timestamp').dt.strftime('%Y-%m-%d'),
                pl.lit(None, dtype=pl.Float64).alias('open'),
                pl.lit(None, dtype=pl.Float64).alias('high'),
                pl.lit(None, dtype=pl.Float64).alias('low'),
                pl.lit(None, dtype=pl.Float64).alias('close'),
                pl.lit(None, dtype=pl.Int64).alias('volume'),
            ])
            .to_dicts()
        )
        return result

    # Parse different formats of timestamp
    ticks_df = pl.DataFrame(data).with_columns([
        pl.when(pl.col('timestamp').str.contains('T'))
        .then(pl.col('timestamp').str.to_datetime(format='%Y-%m-%dT%H:%M:%S', strict=False))
        .otherwise(
            pl.col('timestamp')
            .str.to_date(format='%Y-%m-%d', strict=False)
            .cast(pl.Datetime)
        )
        .dt.date()
        .alias('timestamp'),
        pl.col('open').cast(pl.Float64),
        pl.col('high').cast(pl.Float64),
        pl.col('low').cast(pl.Float64),
        pl.col('close').cast(pl.Float64),
        pl.col('volume').cast(pl.Int64)
    ])

    # Drop optional columns only if they exist
    optional_cols = ["num_trades", "vwap"]
    cols_to_drop = [col for col in optional_cols if col in ticks_df.columns]
    if cols_to_drop:
        ticks_df = ticks_df.drop(cols_to_drop)

    ticks_lf = ticks_df.sort('timestamp').lazy()
    calendar_lf = calendar_lf.join(ticks_lf, on='timestamp', how='left')

    # Collect, convert timestamp to string, and return as list of dicts
    result = (
        calendar_lf.collect()
        .with_columns(pl.col('timestamp').dt.strftime('%Y-%m-%d'))
        .to_dicts()
    )

    return result