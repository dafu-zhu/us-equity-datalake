import requests
import json
import time
from collection.models import FndDataPoint
import datetime as dt
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

HEADER = {'User-Agent': 'name@example.com'}

class Fundamental:
    def __init__(self, cik: str) -> None:
        self.cik = cik

    def get_facts(self, field: str, sleep=True) -> List[dict]:
        """
        Get historical facts of a company using SEC XBRL
        
        :param cik: Company identifier
        :type cik: str
        :param field: Accounting data to fetch
        :type field: str
        """
        cik_padded = str(self.cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
        res = requests.get(url=url, headers=HEADER).json()

        # Avoid reaching api rate limit (10/s)
        if sleep:
            time.sleep(0.1)
        gaap = res['facts']['us-gaap']
        usd_result = gaap[field]['units']['USD']
        
        return usd_result

    def get_dps(self, field: str) -> List[FndDataPoint]:
        """
        Transform raw data point into FndDataPoint object
        """
        raw_data = self.get_facts(field)
        
        dps = []
        for dp in raw_data:
            # Reveal date
            filed_date = dt.datetime.strptime(dp['filed'], '%Y-%m-%d').date()
            
            # Fiscal calendar date, avoid look-ahead bias
            end_date = dt.datetime.strptime(dp['end'], '%Y-%m-%d').date()
            
            # Form to track amendment
            form = dp['form']

            dp_obj = FndDataPoint(
                timestamp=filed_date,
                value=dp['val'],
                end_date=end_date,
                fy=dp['fy'],
                fp=dp['fp'],
                form=form
            )
            dps.append(dp_obj)

        return dps

    def _deduplicate_dps(self, dps: List[FndDataPoint]) -> List[FndDataPoint]:
        """
        Deduplicate datapoints by keeping the most recent filing per fiscal period.

        Groups by (end_date, fy, fp) and selects the filing with the latest timestamp.
        This ensures amendments (10-K/A) supersede original filings (10-K).

        :param dps: List of FundamentalDataPoint objects
        :return: Deduplicated list of FundamentalDataPoint objects
        """
        # Group by fiscal period: (end_date, fy, fp)
        groups = defaultdict(list)
        for dp in dps:
            key = (dp.end_date, dp.fy, dp.fp)
            groups[key].append(dp)

        # Select most recent filing per group
        deduplicated = []
        for key, group in groups.items():
            # Sort by timestamp (filed date) descending, take the most recent
            most_recent = max(group, key=lambda x: x.timestamp)
            deduplicated.append(most_recent)

        # Sort by timestamp for chronological order
        deduplicated.sort(key=lambda x: x.timestamp)

        return deduplicated

    def generate_year_data(self, year: int, field: str, symbol: str) -> None:
        """
        Generate daily values for a given year and field, save to JSON.

        For every calendar day in the year, assigns the most recent filed value
        as of that date (forward-fill logic).

        :param year: Year to generate data for (e.g., 2024)
        :param field: XBRL field name (e.g., 'CashAndCashEquivalentsAtCarryingValue')
        :param symbol: Stock symbol (e.g., 'AAPL')
        """
        # Get and deduplicate datapoints
        raw_dps = self.get_dps(field)
        dps = self._deduplicate_dps(raw_dps)

        # Generate daily values for the year
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)

        daily_data = {}
        current_value = None

        # Iterate through each day of the year
        current_day = start_date
        dp_index = 0

        while current_day <= end_date:
            # Update current_value if a new filing was released on or before this day
            while dp_index < len(dps) and dps[dp_index].timestamp <= current_day:
                current_value = dps[dp_index].value
                dp_index += 1

            # Assign value for this day (None if no filing has been released yet)
            daily_data[current_day.isoformat()] = current_value

            current_day += dt.timedelta(days=1)

        # Save to JSON
        output_dir = Path(f"data/fundamental/{symbol}/{year}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{field}.json"

        with open(output_file, 'w') as file:
            json.dump(daily_data, file, indent=2)

        print(f"Saved {field} data for {symbol} ({year}) to {output_file}")





# Example usage
if __name__ == "__main__":
    cik = '320193'  # Apple Inc.
    symbol = 'AAPL'
    field = 'CostOfGoodsAndServicesSold'
    year = 2024

    # Create Fundamentals instance
    fund = Fundamental(cik)

    # Test deduplication
    print("Testing deduplication...")
    raw_dps = fund.get_dps(field)
    print(f"Raw datapoints: {len(raw_dps)}")

    deduplicated_dps = fund._deduplicate_dps(raw_dps)
    print(f"Deduplicated datapoints: {len(deduplicated_dps)}")

    # Generate and save year data
    print(f"\nGenerating daily data for {symbol} {year}...")
    fund.generate_year_data(year=year, field=field, symbol=symbol)
