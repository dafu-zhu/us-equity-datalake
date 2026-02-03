"""
FinBERT sentiment analysis model implementation.

Uses ProsusAI/finbert for financial domain sentiment analysis.
Optimized for CUDA GPU acceleration with CPU fallback.
"""

import logging
from typing import List, Optional

from quantdl.models.base import SentimentModel, SentimentResult


class FinBERTModel(SentimentModel):
    """
    FinBERT sentiment model using ProsusAI/finbert.

    Features:
    - Lazy loading: Model loaded on first predict() call
    - CUDA support: Auto-detects GPU, falls back to CPU
    - Batch inference: Efficient processing of multiple texts
    - Memory optimized: ~440 MB GPU memory when loaded

    Example:
        >>> model = FinBERTModel()
        >>> results = model.predict(["The company reported strong earnings."])
        >>> print(results[0].label, results[0].score)
        positive 0.95
    """

    MODEL_ID = "ProsusAI/finbert"
    MODEL_VERSION = "1.0.0"  # Track for reproducibility
    MAX_TOKENS = 512

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 8,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize FinBERT model.

        :param device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        :param batch_size: Batch size for inference (8-16 recommended for GPU)
        :param logger: Optional logger instance
        """
        self._device = device
        self._batch_size = batch_size
        self._logger = logger or logging.getLogger(__name__)

        # Lazy loading - these are set in load()
        self._tokenizer = None
        self._model = None
        self._pipeline = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "finbert"

    @property
    def version(self) -> str:
        return self.MODEL_VERSION

    @property
    def max_tokens(self) -> int:
        return self.MAX_TOKENS

    def is_loaded(self) -> bool:
        return self._loaded

    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self._logger.info(f"CUDA available: {device_name}")
                return "cuda"
        except ImportError:
            pass

        self._logger.info("Using CPU for inference")
        return "cpu"

    def load(self) -> None:
        """Load FinBERT model and tokenizer."""
        if self._loaded:
            return

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                pipeline
            )
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for FinBERT. "
                "Install with: uv add transformers torch"
            ) from e

        # Determine device
        device = self._device or self._detect_device()
        device_idx = 0 if device == "cuda" else -1

        self._logger.info(f"Loading FinBERT from {self.MODEL_ID}...")

        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID)

        # Move to device
        if device == "cuda":
            self._model = self._model.cuda()

        # Create pipeline for easy inference
        # Use text-classification task (not sentiment-analysis) for FinBERT
        # top_k=None returns all class scores
        self._pipeline = pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            device=device_idx,
            truncation=True,
            max_length=self.MAX_TOKENS,
            top_k=None
        )

        self._loaded = True
        self._logger.info(f"FinBERT loaded on {device}")

    def predict(self, texts: List[str]) -> List[SentimentResult]:
        """
        Run sentiment prediction on texts.

        :param texts: List of text chunks (each should be <512 tokens)
        :return: List of SentimentResult with label and confidence
        """
        if not texts:
            return []

        # Lazy load on first call
        if not self._loaded:
            self.load()

        results = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]

            # Run inference
            batch_outputs = self._pipeline(batch)

            # Parse results
            for text, output in zip(batch, batch_outputs):
                # output is list of dicts: [{'label': 'positive', 'score': 0.9}, ...]
                # Find highest scoring label
                best = max(output, key=lambda x: x['score'])

                # Normalize label to lowercase
                label = best['label'].lower()

                results.append(SentimentResult(
                    text_chunk=text[:100] + "..." if len(text) > 100 else text,
                    label=label,
                    score=best['score'],
                    model_name=self.name,
                    model_version=self.version
                ))

        return results

    def unload(self) -> None:
        """Release model from memory."""
        if not self._loaded:
            return

        try:
            import torch

            del self._pipeline
            del self._model
            del self._tokenizer

            self._pipeline = None
            self._model = None
            self._tokenizer = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._loaded = False
            self._logger.info("FinBERT unloaded")

        except Exception as e:
            self._logger.warning(f"Error unloading FinBERT: {e}")
