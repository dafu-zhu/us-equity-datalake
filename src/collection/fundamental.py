import requests
import time
from collection.models import FndDataPoint
import datetime as dt
from typing import List, Optional, Tuple
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

    def get_gaap(self, field: str) -> List[dict]:
        """
        Get the complete historical facts of a company using SEC XBRL

        :param cik: Company identifier
        :type cik: str
        :param field: Accounting data to fetch
        :type field: str
        :raises requests.RequestException: If HTTP request fails
        :raises KeyError: If field is not available for this company
        :raises ValueError: If data format is unexpected
        """
        res = self.req_response

        # Check if facts and us-gaap exist
        if 'facts' not in res:
            raise KeyError(f"No 'facts' data found for CIK {self.cik}")

        if 'us-gaap' not in res['facts']:
            raise KeyError(f"No 'us-gaap' data found for CIK {self.cik}")

        gaap = res['facts']['us-gaap']

        # Check if field exists
        if field not in gaap:
            available_fields = list(gaap.keys())
            raise KeyError(
                f"Field '{field}' not available for CIK {self.cik}. "
                f"Available fields: {len(available_fields)} total"
            )

        # Check if units exist for this field
        if 'units' not in gaap[field]:
            raise KeyError(f"No 'units' data found for field '{field}' in CIK {self.cik}")

        # Check for USD first, then shares as fallback
        if 'USD' in gaap[field]['units']:
            result = gaap[field]['units']['USD']
        elif 'shares' in gaap[field]['units']:
            result = gaap[field]['units']['shares']
        else:
            available_units = list(gaap[field]['units'].keys())
            raise KeyError(
                f"Neither USD nor shares units available for field '{field}' in CIK {self.cik}. "
                f"Available units: {available_units}"
            )

        return result
    
    def get_dei(self, field:str) -> List[dict]:
        res = self.req_response

        # Check if facts and us-gaap exist
        if 'facts' not in res:
            raise KeyError(f"No 'facts' data found for CIK {self.cik}")

        if 'dei' not in res['facts']:
            raise KeyError(f"No 'dei' data found for CIK {self.cik}")
        
        dei = res['facts']['dei']

        # Check if field exists
        if field not in dei:
            available_fields = list(dei.keys())
            raise KeyError(
                f"Field '{field}' not available for CIK {self.cik}. "
                f"Available fields: {len(available_fields)} total"
            )
        
        if 'units' not in dei[field]:
            raise KeyError(f"No 'units'/'shares' data found for field '{field}' in CIK {self.cik}")
        
        if 'USD' in dei[field]['units']:
            result = dei[field]['units']['USD']
        elif 'shares' in dei[field]['units']:
            result = dei[field]['units']['shares']
        else:
            raise KeyError(
                f"USD units not available for field '{field}' in CIK {self.cik}. "
            )
        
        return result


    def get_dps(self, field: str, location: str) -> List[FndDataPoint]:
        """
        Transform raw data point into FndDataPoint object
        """
        if location == "dei":
            raw_data = self.get_dei(field)
        elif location == "us-gaap":
            raw_data = self.get_gaap(field)
        else:
            raise ValueError(f"Expected location 'dei' or 'us-gaap', get {location} instead")
        
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
    
    def get_value_tuple(self, field: str, location: str) -> List[Tuple[dt.date, float]]:
        """
        Process the result of get_dps, transform list of FndDataPoint into 
        a list of tuple in order to fit in dataframe
        
        :param field: Fundamental field name
        :type field: str
        :param location: 'dei' or 'us-gaap'
        :return: list of tuples with (date, value), ordered in date
        :rtype: List[Tuple[date, float]]

        Example: [(date(2024, 9, 1), 9.1), (date(2024, 12, 1), 12.1)]
        """
        try:
            dps = self.get_dps(field, location)
            value_tuples = []
            for dp in dps:
                date: dt.date = dp.timestamp
                value = dp.value
                value_tuples.append((date, value))
            
            value_tuples.sort(key=lambda x: x[0])

            return value_tuples
        
        except KeyError as error:
            # Field not available for this company
            self.logger.warning(
                f"Field not available - Symbol: {self.symbol}, Field: {field}, Error: {error}"
            )
            return []

        except requests.RequestException as error:
            # Network or API error
            self.logger.error(
                f"Request failed - Symbol: {self.symbol}, Field: {field}, Error: {error}"
            )
            return []

        except Exception as error:
            # Unexpected error
            self.logger.error(
                f"Unexpected error - Symbol: {self.symbol}, Field: {field}, Error: {error}",
                exc_info=True
            )
            return []

    def collect_fields(self, year: int, fields: List[str], location: str) -> pl.DataFrame:
        """
        Collect multiple fields and put into one single dataframe
        
        :param year: what year
        :type year: int
        :param fields: what fields
        :type fields: List[str]
        :param location: 'dei' or 'us-gaap'
        :return: dataframe with columns [Date, Field1, Field2, ...]
        :rtype: DataFrame
        """
        # Define date range
        start_date = dt.date(year, 1, 1)
        end_date = dt.date(year, 12, 31)
        
        # Load master calendar and merge
        calendar_lf: pl.LazyFrame = (
            pl.scan_parquet(self.calendar_path)
            .filter(pl.col("Date").is_between(start_date, end_date))
            .sort('Date')
            .lazy()
        )

        # Main loop
        for field_name in fields:
            dps = self.get_value_tuple(field_name, location)
            
            if not dps:
                calendar_lf = calendar_lf.with_columns(
                    pl.lit(None, dtype=pl.Float64)
                    .alias(field_name)
                )
            
            else:
                tmp_lf = (
                    pl.DataFrame(
                        dps,
                        schema=['Date', field_name],
                        orient='row'
                    )
                    .with_columns(
                        pl.col(field_name).cast(pl.Float64)
                    )
                    .sort('Date')
                    .drop_nulls(subset=[field_name])
                    .lazy()
                )

                calendar_lf = calendar_lf.join_asof(
                    tmp_lf,
                    on='Date',
                    strategy='backward'
                )
        
        result = calendar_lf.collect()
        self.fields_df = result

        return result

    def store_fields(self, symbol: str, year: int, fields: List[str]):
        if not self.symbol:
            self.symbol = symbol
        
        if not isinstance(self.fields_df, pl.DataFrame):
            self.fields_df = self.collect_fields(year=year, fields=fields)

        # Save to Parquet
        output_dir = self.output_dir / Path(f"{symbol}/{year}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "fundamental.parquet"
        self.fields_df.write_parquet(output_file, compression='zstd')

        print(f"Fundamental fields for {symbol} stored in {output_file} successful!")


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

    # Generate and save year data for all fields
    print(f"Generating daily data for {symbol} {year} with {len(fields)} fields...")
    print("=" * 60)
    res = fund.collect_fields(year=year, fields=fields)
    print(res)
    fund.store_fields(symbol, year, fields)