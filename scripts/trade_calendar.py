import os
import requests
import datetime as dt
from pathlib import Path

from dotenv import load_dotenv
import polars as pl

load_dotenv()

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")

class TradeCalendar:
    def __init__(self, start_day: str, end_day: str):
        self.start_day = start_day
        self.end_day = end_day
        self.store_dir = Path('data/calendar')
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def get_calendar(self) -> pl.DataFrame:
        url = "https://paper-api.alpaca.markets/v2/calendar"

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET
        }

        params = {
            "start": f"{self.start_day}T00:00:00Z",
            "end": f"{self.end_day}T00:00:00Z",
            "date_type": "TRADING"
        }

        response = requests.get(url, headers=headers, params=params).json()

        date_list = []

        for res in response:
            date = dt.datetime.strptime(res['date'], '%Y-%m-%d').date()
            date_list.append(date)

        date_df = pl.DataFrame({'timestamp': date_list})

        return date_df

    def store_calendar(self, name='master'):
        date_df = self.get_calendar()
        file_path = self.store_dir / f'{name}.parquet'
        date_df.write_parquet(file_path)
        print(f"Trading calendar stored at {file_path}")

if __name__ == "__main__":
    start_day = "2009-01-01"
    end_day = "2029-12-31"

    tc = TradeCalendar(start_day, end_day)
    tc.store_calendar()

    calendar_path = Path("data/calendar/master.parquet")
    df = pl.read_parquet(calendar_path)
    print(df)