import requests
import time
from quantdl.collection.models import FndDataPoint, DataSource
import datetime as dt
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from collections import defaultdict
import polars as pl
import json
import yaml

HEADER = {'User-Agent': 'name@example.com'}

FIELD_CONFIG_PATH = Path("configs/approved_mapping.yaml")

with open(FIELD_CONFIG_PATH) as file:
    MAPPINGS = yaml.safe_load(file)

DURATION_CONCEPTS = {
    "rev",
    "cor",
    "op_inc",
    "net_inc",
    "ibt",
    "inc_tax_exp",
    "int_exp",
    "rnd",
    "sga",
    "dna",
    "cfo",
    "cfi",
    "cff",
    "capex",
    "div",
    "sto_isu",
}


def extract_concept(facts: dict, concept: str) -> Optional[dict]:
    """
    Extract a financial concept from SEC XBRL facts using mapping candidates.

    Searches through candidate tags in priority order and returns the first available field.

    Supports two mapping formats:
    - New format: concept: [tag1, tag2, ...]
    - Old format: concept: {gaap_candidates: [tag1, tag2, ...]}

    :param facts: Complete facts dictionary from SEC EDGAR API response
    :param concept: Concept name as defined in MAPPINGS (e.g., 'revenue', 'total assets')
    :return: Field data with units structure, or None if concept not found
    :raises KeyError: If concept not defined in MAPPINGS
    :raises ValueError: If tag format is invalid
    """
    if concept not in MAPPINGS:
        raise KeyError(f"Concept '{concept}' not defined in MAPPINGS")

    mapping = MAPPINGS[concept]

    # Support both old format {gaap_candidates: [...]} and new format [...]
    if isinstance(mapping, dict):
        # Old format: concept: {gaap_candidates: [...]}
        tags = mapping.get('gaap_candidates', [])
    elif isinstance(mapping, list):
        # New format: concept: [...]
        tags = mapping
    else:
        raise ValueError(f"Invalid mapping format for concept '{concept}': expected dict or list")

    for tag in tags:
        if ':' in tag:
            prefix, local = tag.split(':', 1)  # Use maxsplit=1 in case local name has ':'
        else:
            raise ValueError(f'Tag must include prefix: {tag}')

        # Check if prefix exists in facts
        if prefix not in facts:
            continue

        # Check if local tag exists in the prefix namespace
        if local in facts[prefix]:
            return facts[prefix][local]

    # No matching tag found for this concept
    return None


class SECClient:
    """Handles HTTP requests to SEC EDGAR API"""

    def __init__(self, header: Optional[dict] = None, rate_limiter=None):
        self.header = header or HEADER
        self.rate_limiter = rate_limiter

    def fetch_company_facts(self, cik: str) -> dict:
        """
        Fetch company facts from SEC EDGAR API.

        :param cik: Company CIK number (will be zero-padded to 10 digits)
        :return: Complete SEC EDGAR API response as dictionary
        :raises requests.RequestException: If HTTP request fails
        :raises ValueError: If JSON response is invalid
        """
        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"

        try:
            # Apply rate limiting before making API request
            if self.rate_limiter:
                self.rate_limiter.acquire()

            response = requests.get(url=url, headers=self.header)
            response.raise_for_status()
            res = response.json()
        except requests.RequestException as error:
            raise requests.RequestException(f"Failed to fetch data for CIK {cik_padded}: {error}")
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON response for CIK {cik_padded}: {error}")

        return res


class FundamentalExtractor:
    """Extracts specific fields from SEC EDGAR response data"""

    def _normalize_duration_raw(self, raw_data: List[dict]) -> List[dict]:
        """
        Normalize duration metrics to per-quarter values using start/end dates.
        """
        def _pick_frame(frames: Dict[str, dict], base: str) -> Optional[dict]:
            exact = frames.get(base)
            if exact is not None:
                return exact
            inst = frames.get(f"{base}I")
            if inst is not None:
                return inst
            for key, item in frames.items():
                if key.startswith(base):
                    return item
            return None

        # Derive Q4 from the annual frame using that year’s Q1–Q3 only.
        by_frame_year: Dict[int, List[dict]] = defaultdict(list)
        for dp in raw_data:
            start = dp.get("start")
            end = dp.get("end")
            filed = dp.get("filed")
            frame = dp.get("frame")
            if not start or not end or not filed or not frame:
                continue

            start_date = dt.datetime.strptime(start, "%Y-%m-%d").date()
            end_date = dt.datetime.strptime(end, "%Y-%m-%d").date()
            filed_date = dt.datetime.strptime(filed, "%Y-%m-%d").date()
            duration_days = (end_date - start_date).days
            frame_year = int(frame[2:6])

            by_frame_year[frame_year].append(
                {
                    "dp": dp,
                    "start_date": start_date,
                    "end_date": end_date,
                    "filed_date": filed_date,
                    "duration_days": duration_days,
                    "frame": frame,
                }
            )

        normalized: List[dict] = []
        for items in by_frame_year.values():
            items.sort(key=lambda item: item["filed_date"])

            latest_by_frame: Dict[str, dict] = {}
            for item in items:
                frame = item["frame"]
                prev = latest_by_frame.get(frame)
                if prev is None or item["filed_date"] > prev["filed_date"]:
                    latest_by_frame[frame] = item

            # stores quarterly frames that are already standalone (Q1–Q4)
            standalone_frames: Dict[str, dict] = {}
            # stores full‑year frames (used to derive Q4 for duration concepts)
            annual_frames: Dict[str, dict] = {}
            for frame, item in latest_by_frame.items():
                if any (q in frame for q in ('Q1', 'Q2', 'Q3', 'Q4')):
                    standalone_frames[frame] = item
                else:
                    annual_frames[frame] = item

            for frame, item in latest_by_frame.items():
                dp = item["dp"]
                if any (q in frame for q in ('Q1', 'Q2', 'Q3', 'Q4')):
                    normalized.append(dp)
                    continue
                if frame in annual_frames:
                    year = frame[2:6]
                    q1 = _pick_frame(standalone_frames, f"CY{year}Q1")
                    q2 = _pick_frame(standalone_frames, f"CY{year}Q2")
                    q3 = _pick_frame(standalone_frames, f"CY{year}Q3")
                    if q1 and q2 and q3:
                        adjusted = dict(dp)
                        adjusted["val"] = (
                            float(dp["val"])
                            - float(q1["dp"]["val"])
                            - float(q2["dp"]["val"])
                            - float(q3["dp"]["val"])
                        )
                        adjusted["start"] = (q3["end_date"] + dt.timedelta(days=1)).isoformat()
                        normalized.append(adjusted)

        deduped = {}
        for dp in normalized:
            frame = dp.get("frame")
            filed = dp.get("filed")
            if not frame or not filed:
                continue
            key = (filed, frame)
            deduped[key] = dp

        return list(deduped.values())

    def extract_field(self, facts_response: dict, field: str, fact_type: str, cik: str) -> List[dict]:
        """
        Extract raw field data from SEC EDGAR response.

        :param facts_response: Complete SEC EDGAR API response
        :param field: Accounting data to fetch (e.g., 'Assets', 'Revenues')
        :param fact_type: Choose from "us-gaap" or "dei"
        :param cik: CIK number (for error messages)
        :return: List of raw data dictionaries from SEC EDGAR
        :raises KeyError: If field is not available for this company
        """
        # Check if facts and fact_type exist
        if 'facts' not in facts_response:
            raise KeyError(f"No 'facts' data found for CIK {cik}")

        if fact_type not in facts_response['facts']:
            raise KeyError(f"No '{fact_type}' data found for CIK {cik}")

        fact = facts_response['facts'][fact_type]

        # Check if field exists
        if field not in fact:
            available_fields = list(fact.keys())
            raise KeyError(
                f"Field '{field}' not available for CIK {cik}. "
                f"Available fields: {len(available_fields)} total"
            )

        # Check if units exist for this field
        if 'units' not in fact[field]:
            raise KeyError(f"No 'units' data found for field '{field}' in CIK {cik}")

        # Check for USD first, then shares as fallback
        if 'USD' in fact[field]['units']:
            result = fact[field]['units']['USD']
        elif 'shares' in fact[field]['units']:
            result = fact[field]['units']['shares']
        else:
            available_units = list(fact[field]['units'].keys())
            raise KeyError(
                f"Neither USD nor shares units available for field '{field}' in CIK {cik}. "
                f"Available units: {available_units}"
            )

        return result

    def parse_datapoints(
        self,
        raw_data: List[dict],
        normalize_duration: bool = False,
        require_frame: bool = False,
    ) -> List[FndDataPoint]:
        """
        Transform raw SEC data points into FndDataPoint objects.

        :param raw_data: List of raw data dictionaries from SEC EDGAR
        :param normalize_duration: When True, normalize to per-quarter duration values
        :param require_frame: When True, only keep datapoints with a frame
        :return: List of FndDataPoint objects
        """
        if normalize_duration:
            raw_data = self._normalize_duration_raw(raw_data)
        if require_frame:
            raw_data = [dp for dp in raw_data if dp.get("frame")]

        dps = []
        for dp in raw_data:
            # Reveal date
            filed_date = dt.datetime.strptime(dp['filed'], '%Y-%m-%d').date()

            # Fiscal calendar date, avoid look-ahead bias
            end_date = dt.datetime.strptime(dp['end'], '%Y-%m-%d').date()
            start_date = (
                dt.datetime.strptime(dp['start'], '%Y-%m-%d').date()
                if dp.get('start')
                else None
            )

            # Form to track amendment
            form = dp['form']

            dp_obj = FndDataPoint(
                timestamp=filed_date,
                value=dp['val'],
                start_date=start_date,
                end_date=end_date,
                frame=dp.get('frame'),
                is_instant=bool(dp.get('frame') and 'I' in dp.get('frame', '')),
                form=form,
                accn=dp.get('accn')
            )
            dps.append(dp_obj)

        return dps


class FundamentalTransformer:
    """Transforms fundamental data into different formats and aggregations"""

    def __init__(self, calendar_path: Optional[Path] = None):
        self.calendar_path = calendar_path or Path("data/calendar/master.parquet")

    def to_value_tuples(self, dps: List[FndDataPoint]) -> List[Tuple[dt.date, float]]:
        """
        Convert FndDataPoint objects to (date, value) tuples.

        :param dps: List of FndDataPoint objects
        :return: List of tuples with (date, value), sorted by date
        """
        value_tuples = []
        for dp in dps:
            date: dt.date = dp.timestamp
            value = dp.value
            value_tuples.append((date, value))

        value_tuples.sort(key=lambda x: x[0])
        return value_tuples

    def aggregate_fields_raw(
        self,
        start_day: str,
        end_day: str,
        fields_dict: Dict[str, List[Tuple[dt.date, float]]],
    ) -> pl.DataFrame:
        """
        Aggregate multiple fields without forward-filling.
        Returns only actual filing dates (quarterly data points).

        :param start_day: Start day to filter filings, format "YYYY-MM-DD"
        :param end_day: End day to filter filings, format "YYYY-MM-DD"
        :param fields_dict: Dict of {field_name: [(date, value), ...]}
        :return: DataFrame with columns [timestamp, Field1, Field2, ...]
        """
        start_date = dt.datetime.strptime(start_day, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(end_day, "%Y-%m-%d").date()

        # Collect all unique timestamps from all fields
        all_timestamps = set()
        for field_values in fields_dict.values():
            for timestamp, _ in field_values:
                if start_date <= timestamp <= end_date:
                    all_timestamps.add(timestamp)

        # If no data, return empty DataFrame
        if not all_timestamps:
            return pl.DataFrame()

        # Sort timestamps
        sorted_timestamps = sorted(list(all_timestamps))

        # Build result dictionary
        result_dict = {'timestamp': sorted_timestamps}

        # For each field, map timestamp -> value
        for field_name, field_values in fields_dict.items():
            # Create a lookup dict for this field
            value_map = {
                timestamp: value
                for timestamp, value in field_values
                if start_date <= timestamp <= end_date
            }

            # Map each timestamp to its value (or None if not present)
            result_dict[field_name] = [value_map.get(ts) for ts in sorted_timestamps]

        # Create DataFrame with strict=False to allow mixed int/float types
        result_df = pl.DataFrame(result_dict, strict=False)

        # Cast numeric columns to Float64
        for col in result_df.columns:
            if col != 'timestamp':
                result_df = result_df.with_columns(pl.col(col).cast(pl.Float64))

        return result_df

    def aggregate_fields_ffill(
        self,
        start_day: str,
        end_day: str,
        fields_dict: Dict[str, List[Tuple[dt.date, float]]],
    ) -> pl.DataFrame:
        """
        Aggregate multiple fields with forward-filling across trading days.

        :param start_day: Start day to align with master calendar, format "YYYY-MM-DD"
        :param end_day: End day to align with master calendar, format "YYYY-MM-DD"
        :param fields_dict: Dict of {field_name: [(date, value), ...]}
        :return: DataFrame with columns [timestamp, Field1, Field2, ...]
        """
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
                    pl.lit(None, dtype=pl.Float64).alias(field_name)
                )
            else:
                tmp_lf = (
                    pl.DataFrame(values, schema=['timestamp', field_name], orient='row')
                    .with_columns(pl.col(field_name).cast(pl.Float64))
                    .sort('timestamp')
                    .drop_nulls(subset=[field_name])
                    .lazy()
                )

                calendar_lf = calendar_lf.join_asof(
                    tmp_lf, on='timestamp', strategy='backward'
                )

        return calendar_lf.collect()
    

class EDGARDataSource(DataSource):
    """EDGAR data source (2009+)"""

    def __init__(self, cik: str, response: Optional[dict] = None, extractor: Optional[FundamentalExtractor] = None):
        """
        Initialize EDGAR data source.

        :param cik: Company CIK number
        :param response: Pre-fetched SEC EDGAR response (optional, for efficiency)
        :param extractor: Pre-initialized FundamentalExtractor (optional, for reuse)
        """
        self.cik = cik
        self.extractor = extractor or FundamentalExtractor()

        # Use provided response or fetch new one
        if response:
            self.response = response
        else:
            client = SECClient()
            self.response = client.fetch_company_facts(cik)

    def supports_concept(self, concept: str) -> bool:
        """Check if concept exists in approved_mapping.yaml"""
        return concept in MAPPINGS

    def extract_concept(self, concept: str) -> Optional[List[FndDataPoint]]:
        """Extract using XBRL tag mapping"""
        # Use the global extract_concept function to get field data
        field_data = extract_concept(self.response['facts'], concept)

        if field_data is None:
            return None

        # Unit selection logic (USD > shares > first available)
        units = field_data['units']
        if 'USD' in units:
            raw_data = units['USD']
        elif 'shares' in units:
            raw_data = units['shares']
        else:
            raw_data = units[list(units.keys())[0]]

        normalize_duration = concept in DURATION_CONCEPTS

        return self.extractor.parse_datapoints(
            raw_data,
            normalize_duration=normalize_duration,
            require_frame=True
        )

    def get_coverage_period(self) -> tuple[str, str]:
        return ("2009-01-01", "2099-12-31")  # EDGAR coverage
    

class Fundamental:
    """
    Unified fundamental data collector with multi-source support.

    Primary data source: SEC EDGAR (2009+)
    Future: CRSP/Compustat for historical data (pre-2009)

    Supports both:
    - Concept-based extraction: get_concept_data('revenue')
    - Legacy field-based extraction: get_dps('Revenues', 'us-gaap')
    """

    def __init__(
        self,
        cik: str,
        symbol: Optional[str] = None,
        permno: Optional[str] = None,
        rate_limiter=None
    ) -> None:
        self.cik = cik
        self.symbol = symbol
        self.permno = permno  # For future CRSP integration
        self.log_dir = Path("data/logs/fundamental")
        self.calendar_path = Path("data/calendar/master.parquet")
        self.output_dir = Path("data/raw/fundamental")
        self.fields_df = None

        # Create service instances
        self.client = SECClient(rate_limiter=rate_limiter)
        self.extractor = FundamentalExtractor()
        self.transformer = FundamentalTransformer(calendar_path=self.calendar_path)

        # Mkdir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Fetch company facts from EDGAR
        self.req_response = self.client.fetch_company_facts(self.cik)

        # Initialize data sources
        self._sources: List[DataSource] = []

        # Add EDGAR data source (uses already-fetched response for efficiency)
        self._sources.append(
            EDGARDataSource(cik=self.cik, response=self.req_response, extractor=self.extractor)
        )

    def get_sec_field(self, field: str, fact_type: str) -> List[dict]:
        """
        Fetch raw field and field type from SEC EDGAR data for the given cik.

        :param field: Accounting data to fetch
        :param fact_type: Choose from "us-gaap" or "dei"
        :return: A list of dictionaries from SEC EDGAR
        :raises requests.RequestException: If HTTP request fails
        :raises KeyError: If field is not available for this company
        :raises ValueError: If data format is unexpected
        """
        return self.extractor.extract_field(
            self.req_response, field, fact_type, self.cik
        )

    def get_dps(self, field: str, fact_type: str) -> List[FndDataPoint]:
        """
        Transform raw data point into FndDataPoint object.

        :param field: Accounting data to fetch
        :param fact_type: Choose from "us-gaap" or "dei"
        :return: List of FndDataPoint objects
        """
        raw_data = self.get_sec_field(field, fact_type=fact_type)
        return self.extractor.parse_datapoints(raw_data, require_frame=True)

    def get_value_tuple(self, dps: List[FndDataPoint]) -> List[Tuple[dt.date, float]]:
        """
        Process the result of get_dps, transform list of FndDataPoint into
        a list of tuple in order to fit in dataframe.

        :param dps: A list of FndDataPoint from get_dps
        :return: list of tuples with (date, value), ordered in date

        Example: [(date(2024, 9, 1), 9.1), (date(2024, 12, 1), 12.1)]
        """
        return self.transformer.to_value_tuples(dps)

    def collect_fields_raw(
        self,
        start_day: str,
        end_day: str,
        fields_dict: Dict[str, List[Tuple[dt.date, float]]],
    ) -> pl.DataFrame:
        """
        Collect multiple fields without forward-filling - returns only actual filing dates.
        This returns quarterly data points (typically 4-5 per year) without interpolation.

        :param start_day: Start day to filter filings, format "YYYY-MM-DD"
        :param end_day: End day to filter filings, format "YYYY-MM-DD"
        :param fields_dict: Key is the field name, Value is the value tuples from get_value_tuple
        :return: DataFrame with columns [timestamp, Field1, Field2, ...] containing only actual filings
        """
        result_df = self.transformer.aggregate_fields_raw(
            start_day, end_day, fields_dict
        )
        self.fields_df = result_df
        return result_df

    def collect_fields_ffill(
        self,
        start_day: str,
        end_day: str,
        fields_dict: Dict[str, List[Tuple[dt.date, float]]],
    ) -> pl.DataFrame:
        """
        Collect multiple fields and put into one single dataframe with forward-filling.
        This method forward-fills quarterly data across all trading days.

        NOTE: For storage without forward-filling, use collect_fields_raw() instead.

        :param start_day: Start day to align with master calendar, format "YYYY-MM-DD"
        :param end_day: End day to align with master calendar, format "YYYY-MM-DD"
        :param fields_dict: Key is the field name, Value is the value tuples from get_value_tuple
        :return: dataframe with columns [Date, Field1, Field2, ...]
        """
        result = self.transformer.aggregate_fields_ffill(
            start_day, end_day, fields_dict
        )
        self.fields_df = result
        return result
    
    def get_concept_data(
        self,
        concept: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[List[FndDataPoint]]:
        """
        Extract data for a concept using the multi-source mapping system.

        Tries data sources in priority order:
        1. EDGAR (2009+)
        2. CRSP (historical, pre-2009) - when implemented

        :param concept: Concept name from fundamental.xlsx (e.g., 'revenue', 'total assets')
        :param start_date: Optional start date filter "YYYY-MM-DD" (filters on filing date)
        :param end_date: Optional end date filter "YYYY-MM-DD" (filters on filing date)
        :return: List of FndDataPoint objects, or None if not available
        """
        # Try each data source in priority order
        for source in self._sources:
            # Check if source supports this concept
            if not source.supports_concept(concept):
                continue

            # Check date range compatibility (for future multi-source routing)
            if start_date and end_date:
                src_start, src_end = source.get_coverage_period()
                # Simple overlap check: skip source if no overlap
                if start_date > src_end or end_date < src_start:
                    continue

            # Try to extract data
            data = source.extract_concept(concept)
            if data:
                if start_date or end_date:
                    start_dt = (
                        dt.datetime.strptime(start_date, "%Y-%m-%d").date()
                        if start_date
                        else None
                    )
                    end_dt = (
                        dt.datetime.strptime(end_date, "%Y-%m-%d").date()
                        if end_date
                        else None
                    )
                    data = [
                        dp for dp in data
                        if (start_dt is None or dp.timestamp >= start_dt)
                        and (end_dt is None or dp.timestamp <= end_dt)
                    ]
                return data

        # No source could provide this concept
        return None


# Example usage
if __name__ == "__main__":
    cik = '1819994'  # Rocket Lab USA Inc.
    symbol = 'RKLB'
    concepts = ['revenue', 'total assets', 'total liabilities']
    year = 2024

    # Create Fundamentals instance with symbol for better logging
    fund = Fundamental(cik, symbol=symbol)

    fields_dict = {}
    for concept in concepts:
        dps = fund.get_concept_data(concept)
        if dps:
            fields_dict[concept] = fund.get_value_tuple(dps)

    start_day = f"{year}-01-01"
    end_day = f"{year}-12-31"
    collect_df = fund.collect_fields_raw(start_day, end_day, fields_dict)
    print(collect_df.head())
