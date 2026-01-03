import requests
import time
from collection.models import FndDataPoint
import datetime as dt
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from collections import defaultdict
import polars as pl
import json
import logging
from utils.logger import setup_logger

HEADER = {'User-Agent': 'name@example.com'}

class Fundamental:
    def __init__(self, cik: str, symbol: Optional[str] = None) -> None:
        self.cik = cik
        self.symbol = symbol
        self.log_dir = Path("data/logs/fundamental")
        self.calendar_path = Path("data/calendar/master.parquet")
        self.output_dir = Path("data/raw/fundamental")
        self.fields_df = None

        # Mkdir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger(
            name=f"fundamental.{cik}",
            log_dir=self.log_dir,
            level=logging.WARNING
        )

        # Request response
        self.req_response = self._req_response()
    
    def _req_response(self) -> dict:
        cik_padded = str(self.cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"

        try:
            response = requests.get(url=url, headers=HEADER)
            response.raise_for_status()
            res = response.json()
        except requests.RequestException as error:
            raise requests.RequestException(f"Failed to fetch data for CIK {cik_padded}: {error}")
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON response for CIK {cik_padded}: {error}")
        
        return res

    def get_field(self, field: str, fact_type: str) -> List[dict]:
        """
        Get the complete historical facts of a company using SEC XBRL

        :param field: Accounting data to fetch
        :param fact_type: Choose from "us-gaap" or "dei"
        :return: A list of dictionaries from SEC EDGAR
        :raises requests.RequestException: If HTTP request fails
        :raises KeyError: If field is not available for this company
        :raises ValueError: If data format is unexpected
        """
        res = self.req_response

        # Check if facts and us-gaap exist
        if 'facts' not in res:
            raise KeyError(f"No 'facts' data found for CIK {self.cik}")

        if fact_type not in res['facts']:
            raise KeyError(f"No '{fact_type}' data found for CIK {self.cik}")

        fact = res['facts'][fact_type]

        # Check if field exists
        if field not in fact:
            available_fields = list(fact.keys())
            raise KeyError(
                f"Field '{field}' not available for CIK {self.cik}. "
                f"Available fields: {len(available_fields)} total"
            )

        # Check if units exist for this field
        if 'units' not in fact[field]:
            raise KeyError(f"No 'units' data found for field '{field}' in CIK {self.cik}")

        # Check for USD first, then shares as fallback
        if 'USD' in fact[field]['units']:
            result = fact[field]['units']['USD']
        elif 'shares' in fact[field]['units']:
            result = fact[field]['units']['shares']
        else:
            available_units = list(fact[field]['units'].keys())
            raise KeyError(
                f"Neither USD nor shares units available for field '{field}' in CIK {self.cik}. "
                f"Available units: {available_units}"
            )

        return result

    def get_dps(self, field: str, fact_type: str) -> List[FndDataPoint]:
        """
        Transform raw data point into FndDataPoint object

        :param field: Accounting data to fetch
        :param fact_type: Choose from "us-gaap" or "dei"
        """
        raw_data = self.get_field(field, fact_type=fact_type)
        
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
    
    def get_value_tuple(self, dps: List[FndDataPoint]) -> List[Tuple[dt.date, float]]:
        """
        Process the result of get_dps, transform list of FndDataPoint into 
        a list of tuple in order to fit in dataframe
        
        :param dps: A list of FndDataPoint from get_dps
        :return: list of tuples with (date, value), ordered in date

        Example: [(date(2024, 9, 1), 9.1), (date(2024, 12, 1), 12.1)]
        """
        value_tuples = []
        for dp in dps:
            date: dt.date = dp.timestamp
            value = dp.value
            value_tuples.append((date, value))
        
        value_tuples.sort(key=lambda x: x[0])

        return value_tuples

    def collect_fields(
            self, 
            start_day: str,
            end_day: str,
            fields_dict: Dict[str, List[Tuple[dt.date, float]]],
        ) -> pl.DataFrame:
        """
        Collect multiple fields and put into one single dataframe
        
        :param start_day: Start day to align with master calendar, format "YYYY-MM-DD"
        :param end_day: End day to align with master calendar, format "YYYY-MM-DD"
        :param fields_dict: Key is the field name, Value is the value tuples from get_value_tuple
        :return: dataframe with columns [Date, Field1, Field2, ...]
        """
        # Define date range
        start_date = dt.datetime.strptime(start_day, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(end_day, "%Y-%m-%d").date()
        
        # Load master calendar and merge
        calendar_lf: pl.LazyFrame = (
            pl.scan_parquet(self.calendar_path)
            .filter(pl.col("timestamp").is_between(start_date, end_date))
            .sort('timestamp')
            .lazy()
        )

        # Main loop
        for field_name in fields_dict.keys():
            values = fields_dict[field_name]
            
            if not values:
                calendar_lf = calendar_lf.with_columns(
                    pl.lit(None, dtype=pl.Float64)
                    .alias(field_name)
                )
            
            else:
                tmp_lf = (
                    pl.DataFrame(
                        values,
                        schema=['timestamp', field_name],
                        orient='row'
                    )
                    .with_columns(
                        pl.col(field_name).cast(pl.Float64)
                    )
                    .sort('timestamp')
                    .drop_nulls(subset=[field_name])
                    .lazy()
                )

                calendar_lf = calendar_lf.join_asof(
                    tmp_lf,
                    on='timestamp',
                    strategy='backward'
                )
        
        result = calendar_lf.collect()
        self.fields_df = result

        return result


# Example usage
if __name__ == "__main__":
    cik = '1819994'  # Rocket Lab USA Inc.
    symbol = 'RKLB'
    fields = [
        'CostOfGoodsAndServicesSold',  # May not be available for all companies
        'Assets',
        'Liabilities',
        'StockholdersEquity',
        'Revenues',  # Alternative field names to test
        'CashAndCashEquivalentsAtCarryingValue'
    ]
    year = 2024

    # Create Fundamentals instance with symbol for better logging
    fund = Fundamental(cik, symbol=symbol)

    fields_dict = {}
    for field in fields:
        try:
            dps = fund.get_dps(field, 'us-gaap')
            fields_dict[field] = fund.get_value_tuple(dps)
        except KeyError as error:
            print(f"Field not found: {error}")

    start_day = f"{year}-01-01"
    end_day = f"{year}-12-31"
    collect_df = fund.collect_fields(start_day, end_day, fields_dict)
    print(collect_df.head())