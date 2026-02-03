"""
Model abstractions for NLP and ML inference.

This module provides:
- SentimentModel ABC: Base class for sentiment analysis models
- SentimentResult: Dataclass for sentiment predictions
- FinBERTModel: FinBERT implementation using ProsusAI/finbert
"""

from quantdl.models.base import SentimentModel, SentimentResult
from quantdl.models.finbert import FinBERTModel

__all__ = ["SentimentModel", "SentimentResult", "FinBERTModel"]
