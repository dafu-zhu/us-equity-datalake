"""
Base classes for sentiment analysis models.

Provides abstract interface for sentiment models and result dataclass.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SentimentResult:
    """Result from sentiment model prediction."""

    text_chunk: str
    label: str  # "positive", "negative", "neutral"
    score: float  # Confidence score (0-1)
    model_name: str
    model_version: str

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "text_chunk": self.text_chunk,
            "label": self.label,
            "score": self.score,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


class SentimentModel(ABC):
    """Abstract base class for sentiment analysis models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier (e.g., 'finbert')."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version for reproducibility."""
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Maximum input tokens supported by the model."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Called lazily on first predict."""
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[SentimentResult]:
        """
        Run sentiment prediction on a batch of texts.

        :param texts: List of text chunks to analyze
        :return: List of SentimentResult for each input text
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""
        pass

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return False
