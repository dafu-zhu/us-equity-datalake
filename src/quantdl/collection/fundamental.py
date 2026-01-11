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
import os
from dotenv import load_dotenv

load_dotenv()

# SEC requires a valid User-Agent header (email or contact info)
# Set via environment variable SEC_USER_AGENT or fallback to placeholder
HEADER = {'User-Agent': os.getenv('SEC_USER_AGENT', 'your_name@example.com')}

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

    Searches through ALL candidate tags and merges data from all available fields.
    This handles cases where companies switch from deprecated to new XBRL tags
    (e.g., SalesRevenueNet -> Revenues in 2018).

    Mapping format in approved_mapping.yaml:
        concept: [tag1, tag2, ...]

    :param facts: Complete facts dictionary from SEC EDGAR API response
    :param concept: Concept name as defined in MAPPINGS (e.g., 'rev', 'ta')
    :return: Field data with merged units from all matching tags, or None if no tags found
    :raises KeyError: If concept not defined in MAPPINGS
    :raises ValueError: If tag format is invalid
    """
    if concept not in MAPPINGS:
        raise KeyError(f"Concept '{concept}' not defined in MAPPINGS")

    tags = MAPPINGS[concept]

    if not isinstance(tags, list):
        raise ValueError(f"Invalid mapping format for concept '{concept}': expected list of tags")

    # Collect all matching fields' data
    all_field_data = []
    for tag in tags:
        if ':' in tag:
            prefix, local = tag.split(':', 1)  # Use maxsplit=1 in case local name has ':'
        else:
            raise ValueError(f'Tag must include prefix: {tag}')

        # Check if this tag exists
        if prefix in facts and local in facts[prefix]:
            all_field_data.append(facts[prefix][local])

    if not all_field_data:
        return None

    # If only one field found, return it directly
    if len(all_field_data) == 1:
        return all_field_data[0]

    # Merge data from multiple fields (handles deprecated tag transitions)
    # Structure: {label: "...", description: "...", units: {USD: [...], shares: [...]}}
    merged = {
        'label': all_field_data[0].get('label', ''),
        'description': all_field_data[0].get('description', ''),
        'units': {}
    }

    # Merge units from all matching fields
    for field_data in all_field_data:
        if 'units' not in field_data:
            continue
        for unit_type, unit_data in field_data['units'].items():
            if unit_type not in merged['units']:
                merged['units'][unit_type] = []
            merged['units'][unit_type].extend(unit_data)

    # Deduplicate data points by (accn, frame, filed) to avoid double-counting
    # This handles edge cases where the same data might appear under multiple tags
    for unit_type in merged['units']:
        seen = {}
        for dp in merged['units'][unit_type]:
            # Use accn (filing accession number) + frame + filed as unique key
            key = (dp.get('accn'), dp.get('frame'), dp.get('filed'))
            # Keep the first occurrence (or could use latest filed date if needed)
            if key not in seen:
                seen[key] = dp
        merged['units'][unit_type] = list(seen.values())

    return merged


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

