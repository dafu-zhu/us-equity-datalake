"""
SEC EDGAR text extraction for sentiment analysis.

Extracts MD&A (Management's Discussion and Analysis) sections from 10-K and 10-Q filings.
"""

import re
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

# SEC requires a valid User-Agent header
HEADER = {'User-Agent': os.getenv('SEC_USER_AGENT', 'your_name@example.com')}


@dataclass
class FilingText:
    """Extracted text from an SEC filing."""

    cik: str
    accession_number: str
    filing_date: str  # YYYY-MM-DD
    filing_type: str  # "10-K" or "10-Q"
    section: str  # "MD&A"
    text: str
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "cik": self.cik,
            "accession_number": self.accession_number,
            "filing_date": self.filing_date,
            "filing_type": self.filing_type,
            "section": self.section,
            "text": self.text,
            "fiscal_year": self.fiscal_year,
            "fiscal_quarter": self.fiscal_quarter,
        }


class SECTextExtractor:
    """
    Extracts text sections from SEC EDGAR filings.

    Downloads full-submission.txt and extracts MD&A section using regex patterns.
    Handles both 10-K and 10-Q filings.
    """

    # Regex patterns for MD&A section boundaries
    # SEC filings use standard item numbering
    MDA_START_PATTERNS = [
        # 10-K Item 7: Management's Discussion and Analysis
        r"(?i)item\s+7[\.\s:]*management['\u2019]?s\s+discussion\s+and\s+analysis",
        r"(?i)item\s+7[\.\s:]*md\s*&\s*a",
        r"(?i)item\s+7[\.\s:]*management['\u2019]?s\s+discussion",
        # 10-Q Item 2: Management's Discussion and Analysis
        r"(?i)item\s+2[\.\s:]*management['\u2019]?s\s+discussion\s+and\s+analysis",
        r"(?i)item\s+2[\.\s:]*md\s*&\s*a",
    ]

    MDA_END_PATTERNS = [
        # 10-K Item 7A or 8 typically follows Item 7
        r"(?i)item\s+7a[\.\s:]*quantitative\s+and\s+qualitative",
        r"(?i)item\s+8[\.\s:]*financial\s+statements",
        # 10-Q Item 3 typically follows Item 2
        r"(?i)item\s+3[\.\s:]*quantitative\s+and\s+qualitative",
        r"(?i)item\s+4[\.\s:]*controls\s+and\s+procedures",
    ]

    def __init__(
        self,
        rate_limiter=None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize SEC text extractor.

        :param rate_limiter: Rate limiter for SEC API (9.5 req/sec)
        :param logger: Optional logger instance
        """
        self.rate_limiter = rate_limiter
        self.logger = logger or logging.getLogger(__name__)

        # HTTP session with retry
        self.session = requests.Session()

        retry_strategy = Retry(
            total=5,
            backoff_factor=2,  # 0, 2, 4, 8, 16 seconds between retries (moderate)
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=50,
        )

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.timeout = (10, 60)  # Longer read timeout for large filings

    def _normalize_accession(self, accession: str) -> str:
        """Convert accession number to URL format (no dashes)."""
        return accession.replace("-", "")

    def fetch_filing_text(self, cik: str, accession: str) -> Optional[str]:
        """
        Fetch full text of SEC filing.

        :param cik: Company CIK (zero-padded to 10 digits)
        :param accession: Accession number (with or without dashes)
        :return: Full filing text or None if fetch fails
        """
        cik_padded = str(cik).zfill(10)
        accession_clean = self._normalize_accession(accession)

        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_padded}/{accession_clean}/{accession}.txt"
        )

        try:
            if self.rate_limiter:
                self.rate_limiter.acquire()

            response = self.session.get(
                url=url,
                headers=HEADER,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.text

        except requests.RequestException as e:
            self.logger.warning(f"Failed to fetch filing {accession}: {e}")
            return None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove XBRL tags
        text = re.sub(r'<[^>]*:[^>]+>', ' ', text)
        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#[0-9]+;', ' ')
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _decode_html_entities(self, text: str) -> str:
        """Decode common HTML entities for pattern matching."""
        # Decode numeric entities
        text = re.sub(r'&#8217;', "'", text)  # Right single quote
        text = re.sub(r'&#8216;', "'", text)  # Left single quote
        text = re.sub(r'&#8220;', '"', text)  # Left double quote
        text = re.sub(r'&#8221;', '"', text)  # Right double quote
        text = re.sub(r'&#160;', ' ', text)   # Non-breaking space
        text = re.sub(r'&#38;', '&', text)    # Ampersand
        text = re.sub(r'&#[0-9]+;', ' ', text)  # Other numeric entities
        # Named entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&apos;', "'")
        text = text.replace('&rsquo;', "'")
        text = text.replace('&lsquo;', "'")
        text = text.replace('&rdquo;', '"')
        text = text.replace('&ldquo;', '"')
        return text

    def extract_mda(self, filing_text: str, filing_type: str) -> Optional[str]:
        """
        Extract MD&A section from filing text.

        :param filing_text: Full filing text
        :param filing_type: "10-K" or "10-Q"
        :return: Extracted MD&A text or None if not found
        """
        # Decode HTML entities for better pattern matching
        search_text = self._decode_html_entities(filing_text)

        # Try each start pattern
        start_match = None
        for pattern in self.MDA_START_PATTERNS:
            match = re.search(pattern, search_text)
            if match:
                start_match = match
                break

        if not start_match:
            self.logger.debug("MD&A start not found")
            return None

        # Get text after start marker (use decoded text for matching)
        text_after_start = search_text[start_match.end():]

        # Try each end pattern
        end_match = None
        for pattern in self.MDA_END_PATTERNS:
            match = re.search(pattern, text_after_start)
            if match:
                if end_match is None or match.start() < end_match.start():
                    end_match = match

        if end_match:
            mda_text = text_after_start[:end_match.start()]
        else:
            # If no end marker, take up to 100KB
            mda_text = text_after_start[:100000]

        # Clean HTML and normalize
        mda_text = self._clean_html(mda_text)

        # Validate minimum length
        if len(mda_text) < 500:
            self.logger.debug(f"MD&A too short: {len(mda_text)} chars")
            return None

        return mda_text

    def extract_filing(
        self,
        cik: str,
        accession: str,
        filing_date: str,
        filing_type: str,
        fiscal_year: Optional[int] = None,
        fiscal_quarter: Optional[int] = None
    ) -> Optional[FilingText]:
        """
        Extract MD&A from a single filing.

        :param cik: Company CIK
        :param accession: Filing accession number
        :param filing_date: Filing date (YYYY-MM-DD)
        :param filing_type: "10-K" or "10-Q"
        :param fiscal_year: Fiscal year of the filing
        :param fiscal_quarter: Fiscal quarter (None for 10-K)
        :return: FilingText with extracted MD&A or None
        """
        # Fetch full filing
        filing_text = self.fetch_filing_text(cik, accession)
        if not filing_text:
            return None

        # Extract MD&A
        mda_text = self.extract_mda(filing_text, filing_type)
        if not mda_text:
            return None

        return FilingText(
            cik=cik,
            accession_number=accession,
            filing_date=filing_date,
            filing_type=filing_type,
            section="MD&A",
            text=mda_text,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter
        )


class SentimentCollector:
    """
    Orchestrates sentiment data collection from SEC filings.

    Coordinates text extraction with fundamental filing metadata.
    """

    def __init__(
        self,
        rate_limiter=None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize sentiment collector.

        :param rate_limiter: Rate limiter for SEC API
        :param logger: Optional logger instance
        """
        self.rate_limiter = rate_limiter
        self.logger = logger or logging.getLogger(__name__)
        self.extractor = SECTextExtractor(
            rate_limiter=rate_limiter,
            logger=logger
        )

    def collect_filing_texts(
        self,
        cik: str,
        filings: List[Dict],
        max_filings: Optional[int] = None
    ) -> List[FilingText]:
        """
        Collect MD&A texts from multiple filings.

        :param cik: Company CIK
        :param filings: List of filing metadata dicts with keys:
                       accession, filing_date, form, fiscal_year, fiscal_quarter
        :param max_filings: Maximum number of filings to process (None for all)
        :return: List of FilingText objects with extracted text
        """
        results = []
        filings_to_process = filings[:max_filings] if max_filings else filings

        for filing in filings_to_process:
            # Only process 10-K and 10-Q
            form = filing.get('form', '')
            if form not in ('10-K', '10-Q', '10-K/A', '10-Q/A'):
                continue

            filing_type = '10-K' if '10-K' in form else '10-Q'

            filing_text = self.extractor.extract_filing(
                cik=cik,
                accession=filing.get('accession', ''),
                filing_date=filing.get('filing_date', ''),
                filing_type=filing_type,
                fiscal_year=filing.get('fiscal_year'),
                fiscal_quarter=filing.get('fiscal_quarter')
            )

            if filing_text:
                results.append(filing_text)
                self.logger.debug(
                    f"Extracted MD&A: {cik} {filing_type} "
                    f"{filing.get('filing_date')} ({len(filing_text.text)} chars)"
                )

        return results

    def get_filings_metadata(
        self,
        cik: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get filing metadata from SEC EDGAR for a CIK.

        Uses the same SEC API as fundamental data collection.

        :param cik: Company CIK
        :param start_date: Start date filter (YYYY-MM-DD)
        :param end_date: End date filter (YYYY-MM-DD)
        :return: List of filing metadata dicts
        """
        from quantdl.collection.fundamental import SECClient

        client = SECClient(rate_limiter=self.rate_limiter)

        try:
            company_data = client.fetch_company_facts(cik)
        except requests.RequestException as e:
            self.logger.warning(f"Failed to fetch company facts for {cik}: {e}")
            return []

        # Extract filing info from any concept that has filings
        filings = []
        seen_accessions = set()

        facts = company_data.get('facts', {})

        # Check us-gaap and dei for filings
        for namespace in ['us-gaap', 'dei']:
            if namespace not in facts:
                continue

            for concept_data in facts[namespace].values():
                if 'units' not in concept_data:
                    continue

                for unit_data in concept_data['units'].values():
                    for dp in unit_data:
                        accn = dp.get('accn')
                        if not accn or accn in seen_accessions:
                            continue

                        form = dp.get('form', '')
                        if form not in ('10-K', '10-Q', '10-K/A', '10-Q/A'):
                            continue

                        filed = dp.get('filed')
                        frame = dp.get('frame', '')

                        # Parse fiscal info from frame (e.g., CY2023Q3)
                        fiscal_year = None
                        fiscal_quarter = None
                        if frame:
                            year_match = re.search(r'CY(\d{4})', frame)
                            if year_match:
                                fiscal_year = int(year_match.group(1))
                            quarter_match = re.search(r'Q(\d)', frame)
                            if quarter_match:
                                fiscal_quarter = int(quarter_match.group(1))

                        seen_accessions.add(accn)
                        filings.append({
                            'accession': accn,
                            'filing_date': filed,
                            'form': form,
                            'fiscal_year': fiscal_year,
                            'fiscal_quarter': fiscal_quarter
                        })

        # Filter by date range
        if start_date or end_date:
            filtered = []
            for f in filings:
                fd = f.get('filing_date', '')
                if start_date and fd < start_date:
                    continue
                if end_date and fd > end_date:
                    continue
                filtered.append(f)
            filings = filtered

        # Sort by filing date descending
        filings.sort(key=lambda x: x.get('filing_date', ''), reverse=True)

        # Remove duplicates (keep most recent per accession)
        unique_filings = []
        seen = set()
        for f in filings:
            accn = f.get('accession')
            if accn not in seen:
                seen.add(accn)
                unique_filings.append(f)

        return unique_filings
