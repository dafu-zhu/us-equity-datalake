"""Unit tests for sentiment text extraction."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from quantdl.collection.sentiment import (
    FilingText,
    SECTextExtractor,
    SentimentCollector
)


class TestFilingText:
    """Tests for FilingText dataclass."""

    def test_filing_text_creation(self):
        """Test basic FilingText creation."""
        ft = FilingText(
            cik="0000320193",
            accession_number="0000320193-24-000001",
            filing_date="2024-01-15",
            filing_type="10-K",
            section="MD&A",
            text="Sample text content",
            fiscal_year=2023,
            fiscal_quarter=None
        )

        assert ft.cik == "0000320193"
        assert ft.filing_type == "10-K"
        assert ft.fiscal_year == 2023
        assert ft.fiscal_quarter is None

    def test_filing_text_to_dict(self):
        """Test FilingText.to_dict() method."""
        ft = FilingText(
            cik="0000320193",
            accession_number="test",
            filing_date="2024-01-15",
            filing_type="10-Q",
            section="MD&A",
            text="Content",
            fiscal_year=2024,
            fiscal_quarter=1
        )

        d = ft.to_dict()

        assert isinstance(d, dict)
        assert d["cik"] == "0000320193"
        assert d["filing_type"] == "10-Q"
        assert d["fiscal_quarter"] == 1


class TestSECTextExtractor:
    """Tests for SECTextExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create SECTextExtractor instance."""
        return SECTextExtractor()

    def test_normalize_accession(self, extractor):
        """Test accession number normalization."""
        assert extractor._normalize_accession("0000320193-24-000001") == "000032019324000001"
        assert extractor._normalize_accession("0001234567-23-999999") == "000123456723999999"

    def test_clean_html_basic(self, extractor):
        """Test HTML cleaning."""
        html = "<p>Hello</p><br>World"
        cleaned = extractor._clean_html(html)

        assert "<p>" not in cleaned
        assert "<br>" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned

    def test_clean_html_entities(self, extractor):
        """Test HTML entity decoding."""
        html = "A&amp;B &nbsp; C&lt;D"
        cleaned = extractor._clean_html(html)

        assert "A&B" in cleaned
        assert "C<D" in cleaned

    def test_clean_html_whitespace(self, extractor):
        """Test whitespace normalization."""
        html = "A   \n\n   B   \t\t   C"
        cleaned = extractor._clean_html(html)

        # Should collapse multiple whitespace
        assert "  " not in cleaned

    def test_extract_mda_not_found(self, extractor):
        """Test MD&A extraction when section not found."""
        text = "This is a filing without any Item 7 or management discussion."
        result = extractor.extract_mda(text, "10-K")

        assert result is None

    def test_extract_mda_10k(self, extractor):
        """Test MD&A extraction from 10-K."""
        # MD&A content must be at least 500 chars to pass validation
        mda_content = """
        Our company had a great year with significant revenue growth.
        We expanded into new markets and improved operational efficiency.
        The management team is pleased to report strong performance across all segments.
        Revenue increased by 25% year over year driven by strong demand.
        Operating margins improved due to cost optimization initiatives.
        We expect continued growth in the coming fiscal year.
        Our strategic investments in technology have paid dividends.
        Customer satisfaction scores reached all-time highs this year.
        """ * 2  # Repeat to ensure >500 chars

        text = f"""
        ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

        {mda_content}

        ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK

        Market risk section follows.
        """

        result = extractor.extract_mda(text, "10-K")

        assert result is not None
        assert "great year" in result
        assert "ITEM 7A" not in result

    def test_extract_mda_10q(self, extractor):
        """Test MD&A extraction from 10-Q."""
        # MD&A content must be at least 500 chars to pass validation
        mda_content = """
        Quarterly results showed improvement across all business segments.
        Revenue increased quarter over quarter driven by strong demand.
        Operating expenses were well controlled during the period.
        The company continues to invest in research and development.
        Management remains confident in the business outlook.
        Customer engagement metrics showed positive trends.
        """ * 3  # Repeat to ensure >500 chars

        text = f"""
        Item 2. Management's Discussion and Analysis

        {mda_content}

        Item 3. Quantitative and Qualitative Disclosures

        Market risk info.
        """

        result = extractor.extract_mda(text, "10-Q")

        assert result is not None
        assert "improvement" in result

    def test_extract_mda_too_short(self, extractor):
        """Test MD&A extraction rejects short content."""
        text = """
        ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

        Short.

        ITEM 8. FINANCIAL STATEMENTS
        """

        result = extractor.extract_mda(text, "10-K")

        # Should be None because content is too short (<500 chars)
        assert result is None


class TestSentimentCollector:
    """Tests for SentimentCollector."""

    @pytest.fixture
    def collector(self):
        """Create SentimentCollector instance."""
        return SentimentCollector()

    def test_collect_filing_texts_empty(self, collector):
        """Empty filings list returns empty results."""
        results = collector.collect_filing_texts("0000320193", [])
        assert results == []

    def test_collect_filing_texts_filter_forms(self, collector):
        """Should filter to only 10-K and 10-Q forms."""
        # Mock the extractor
        collector.extractor = Mock()
        collector.extractor.extract_filing = Mock(return_value=None)

        filings = [
            {"accession": "1", "form": "10-K", "filing_date": "2024-01-15"},
            {"accession": "2", "form": "8-K", "filing_date": "2024-01-10"},  # Should skip
            {"accession": "3", "form": "10-Q", "filing_date": "2024-04-15"},
            {"accession": "4", "form": "DEF 14A", "filing_date": "2024-03-01"},  # Should skip
        ]

        collector.collect_filing_texts("0000320193", filings)

        # Should only call extract_filing for 10-K and 10-Q
        assert collector.extractor.extract_filing.call_count == 2

    @patch('quantdl.collection.fundamental.SECClient')
    def test_get_filings_metadata(self, mock_client_class, collector):
        """Test getting filing metadata from SEC."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_client.fetch_company_facts.return_value = {
            "facts": {
                "us-gaap": {
                    "Revenue": {
                        "units": {
                            "USD": [
                                {
                                    "accn": "0000320193-24-000001",
                                    "filed": "2024-01-15",
                                    "form": "10-K",
                                    "frame": "CY2023"
                                }
                            ]
                        }
                    }
                }
            }
        }

        filings = collector.get_filings_metadata("0000320193")

        assert len(filings) == 1
        assert filings[0]["form"] == "10-K"
        assert filings[0]["fiscal_year"] == 2023
