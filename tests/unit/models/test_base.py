"""Unit tests for sentiment model base classes."""

import pytest
from quantdl.models.base import SentimentResult


class TestSentimentResult:
    """Tests for SentimentResult dataclass."""

    def test_sentiment_result_creation(self):
        """Test basic SentimentResult creation."""
        result = SentimentResult(
            text_chunk="Test text",
            label="positive",
            score=0.95,
            model_name="finbert",
            model_version="1.0.0"
        )

        assert result.text_chunk == "Test text"
        assert result.label == "positive"
        assert result.score == 0.95
        assert result.model_name == "finbert"
        assert result.model_version == "1.0.0"

    def test_sentiment_result_to_dict(self):
        """Test SentimentResult.to_dict() method."""
        result = SentimentResult(
            text_chunk="Sample text",
            label="negative",
            score=0.85,
            model_name="finbert",
            model_version="1.0.0"
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["text_chunk"] == "Sample text"
        assert d["label"] == "negative"
        assert d["score"] == 0.85
        assert d["model_name"] == "finbert"
        assert d["model_version"] == "1.0.0"

    def test_sentiment_result_labels(self):
        """Test all valid sentiment labels."""
        for label in ["positive", "negative", "neutral"]:
            result = SentimentResult(
                text_chunk="Test",
                label=label,
                score=0.9,
                model_name="test",
                model_version="1.0"
            )
            assert result.label == label
