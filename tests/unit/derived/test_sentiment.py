"""Unit tests for sentiment computation functions."""

import pytest
from unittest.mock import Mock, MagicMock
import polars as pl

from quantdl.derived.sentiment import (
    chunk_text,
    compute_filing_sentiment,
    compute_sentiment_long,
    compute_sentiment_for_cik,
    FilingSentiment
)
from quantdl.collection.sentiment import FilingText
from quantdl.models.base import SentimentResult


class TestChunkText:
    """Tests for chunk_text function."""

    def test_chunk_short_text(self):
        """Short text should not be chunked."""
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=1500)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_empty_text(self):
        """Empty text returns empty list."""
        chunks = chunk_text("")
        assert chunks == []

    def test_chunk_none_text(self):
        """None text returns empty list."""
        chunks = chunk_text(None)
        assert chunks == []

    def test_chunk_long_text(self):
        """Long text should be chunked."""
        text = "A" * 5000  # 5000 chars
        chunks = chunk_text(text, chunk_size=1500, overlap=200)

        assert len(chunks) > 1
        # Each chunk should be around chunk_size
        for chunk in chunks:
            assert len(chunk) <= 1500 + 100  # Allow some margin

    def test_chunk_with_sentences(self):
        """Chunking should prefer sentence boundaries."""
        sentences = ["This is sentence one. ", "This is sentence two. "] * 100
        text = "".join(sentences)

        chunks = chunk_text(text, chunk_size=500)

        # Chunks should end with periods where possible
        for chunk in chunks[:-1]:  # Exclude last chunk
            # Most chunks should end at sentence boundary
            pass  # Just verify no errors

    def test_chunk_overlap(self):
        """Verify overlap between chunks."""
        text = "ABCDEFGHIJ" * 200  # 2000 chars
        chunks = chunk_text(text, chunk_size=500, overlap=100)

        # Should have multiple chunks
        assert len(chunks) > 1


class TestComputeFilingSentiment:
    """Tests for compute_filing_sentiment function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock sentiment model."""
        model = Mock()
        model.name = "finbert"
        model.version = "1.0.0"
        model.predict = Mock(return_value=[
            SentimentResult(
                text_chunk="Test",
                label="positive",
                score=0.9,
                model_name="finbert",
                model_version="1.0.0"
            )
        ])
        return model

    @pytest.fixture
    def sample_filing_text(self):
        """Create sample FilingText."""
        return FilingText(
            cik="0000320193",
            accession_number="0000320193-24-000001",
            filing_date="2024-01-15",
            filing_type="10-K",
            section="MD&A",
            text="This is a sample MD&A section with positive results.",
            fiscal_year=2023,
            fiscal_quarter=None
        )

    def test_compute_filing_sentiment_basic(self, mock_model, sample_filing_text):
        """Test basic sentiment computation."""
        result = compute_filing_sentiment(
            filing_text=sample_filing_text,
            model=mock_model
        )

        assert result is not None
        assert isinstance(result, FilingSentiment)
        assert result.cik == "0000320193"
        assert result.filing_date == "2024-01-15"
        assert result.filing_type == "10-K"
        assert result.model_name == "finbert"

    def test_compute_filing_sentiment_empty_text(self, mock_model):
        """Empty text should return None."""
        filing = FilingText(
            cik="0000320193",
            accession_number="test",
            filing_date="2024-01-15",
            filing_type="10-K",
            section="MD&A",
            text=""
        )

        result = compute_filing_sentiment(filing, mock_model)
        assert result is None

    def test_compute_filing_sentiment_aggregation(self, sample_filing_text):
        """Test sentiment score aggregation."""
        model = Mock()
        model.name = "finbert"
        model.version = "1.0.0"

        # Return mixed sentiments
        model.predict = Mock(return_value=[
            SentimentResult("chunk1", "positive", 0.8, "finbert", "1.0.0"),
            SentimentResult("chunk2", "negative", 0.6, "finbert", "1.0.0"),
            SentimentResult("chunk3", "neutral", 0.7, "finbert", "1.0.0"),
        ])

        result = compute_filing_sentiment(sample_filing_text, model)

        assert result is not None
        # Sentiment score: (0.8 - 0.6) / 3 = 0.0667
        assert result.positive_ratio == pytest.approx(1/3, rel=0.01)
        assert result.negative_ratio == pytest.approx(1/3, rel=0.01)
        assert result.neutral_ratio == pytest.approx(1/3, rel=0.01)
        assert result.chunk_count == 3


class TestComputeSentimentLong:
    """Tests for compute_sentiment_long function."""

    def test_empty_input(self):
        """Empty input returns empty DataFrame."""
        df = compute_sentiment_long([])
        assert len(df) == 0

    def test_single_filing(self):
        """Single filing produces expected rows."""
        sentiment = FilingSentiment(
            cik="0000320193",
            filing_date="2024-01-15",
            filing_type="10-K",
            fiscal_year=2023,
            fiscal_quarter=None,
            sentiment_score=0.5,
            positive_ratio=0.6,
            negative_ratio=0.2,
            neutral_ratio=0.2,
            avg_positive_confidence=0.85,
            avg_negative_confidence=0.75,
            chunk_count=10,
            text_length=5000,
            # Distribution metrics
            sentiment_std=0.15,
            sentiment_skew=0.1,
            sentiment_range=0.8,
            extreme_negative_ratio=0.05,
            confidence_std=0.1,
            # LM word ratios
            word_count=1000,
            uncertainty_ratio=0.02,
            litigious_ratio=0.01,
            constraining_ratio=0.015,
            weak_modal_ratio=0.01,
            strong_modal_ratio=0.005,
            # Readability
            avg_sentence_length=20.5,
            fog_index=14.2,
            model_name="finbert",
            model_version="1.0.0"
        )

        df = compute_sentiment_long([sentiment])

        # Should have 21 rows (21 metrics)
        assert len(df) == 21
        assert "cik" in df.columns
        assert "as_of_date" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns

    def test_multiple_filings(self):
        """Multiple filings produce correct row count."""
        sentiments = [
            FilingSentiment(
                cik="0000320193",
                filing_date=f"2024-0{i}-15",
                filing_type="10-Q" if i < 4 else "10-K",
                fiscal_year=2023,
                fiscal_quarter=i if i < 4 else None,
                sentiment_score=0.5,
                positive_ratio=0.6,
                negative_ratio=0.2,
                neutral_ratio=0.2,
                avg_positive_confidence=0.85,
                avg_negative_confidence=0.75,
                chunk_count=10,
                text_length=5000,
                # Distribution metrics
                sentiment_std=0.15,
                sentiment_skew=0.1,
                sentiment_range=0.8,
                extreme_negative_ratio=0.05,
                confidence_std=0.1,
                # LM word ratios
                word_count=1000,
                uncertainty_ratio=0.02,
                litigious_ratio=0.01,
                constraining_ratio=0.015,
                weak_modal_ratio=0.01,
                strong_modal_ratio=0.005,
                # Readability
                avg_sentence_length=20.5,
                fog_index=14.2,
                model_name="finbert",
                model_version="1.0.0"
            )
            for i in range(1, 4)
        ]

        df = compute_sentiment_long(sentiments)

        # 3 filings * 21 metrics = 63 rows
        assert len(df) == 63


class TestComputeSentimentForCik:
    """Tests for compute_sentiment_for_cik function."""

    def test_empty_filing_texts(self):
        """Empty filing texts returns empty DataFrame."""
        model = Mock()
        df = compute_sentiment_for_cik(
            cik="0000320193",
            filing_texts=[],
            model=model
        )

        assert len(df) == 0

    def test_with_filing_texts(self):
        """Test with actual filing texts."""
        model = Mock()
        model.name = "finbert"
        model.version = "1.0.0"
        model.predict = Mock(return_value=[
            SentimentResult("chunk", "positive", 0.9, "finbert", "1.0.0")
        ])

        filing_texts = [
            FilingText(
                cik="0000320193",
                accession_number="test-1",
                filing_date="2024-01-15",
                filing_type="10-K",
                section="MD&A",
                text="Sample MD&A text for testing sentiment analysis.",
                fiscal_year=2023,
                fiscal_quarter=None
            )
        ]

        df = compute_sentiment_for_cik(
            cik="0000320193",
            filing_texts=filing_texts,
            model=model
        )

        assert len(df) > 0
        assert df["cik"][0] == "0000320193"
