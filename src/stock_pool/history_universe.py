import polars as pl
from pathlib import Path

year = 2010
month = 1


def get_hist_universe(year: int, month: int):
    """
    Historical universe common stock list from CRSP database
    Ticker name has no '-' or '.', e.g. BRK.B in alpaca, BRK-B in SEC, BRKB in CRSP
    """

    file_path = Path("data/symbols/history_symbols.csv")

    q = (
        pl.scan_csv(file_path)
        .with_columns(
            pl.col('date').str.strptime(pl.Date, '%Y-%m-%d'),
            pl.col('TSYMBOL').alias('Ticker'),
            pl.col('COMNAM').alias('Name')
        )
        .filter(
            pl.col('date').dt.year().eq(year),
            pl.col('date').dt.month().eq(month)
        )
        .select(['Ticker', 'Name'])
        .drop_nulls()
    )

    return q.collect()

if __name__ == "__main__":
    df = get_hist_universe(year, month)
    print(df['Ticker'].to_list())