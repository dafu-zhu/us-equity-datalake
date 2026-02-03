"""Unit tests for FinBERT model (mocked)."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestFinBERTModel:
    """Tests for FinBERTModel with mocked transformers."""

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers and torch imports."""
        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'torch': MagicMock(),
        }):
            yield

    def test_finbert_properties(self, mock_transformers):
        """Test FinBERT model properties."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()

        assert model.name == "finbert"
        assert model.version == "1.0.0"
        assert model.max_tokens == 512
        assert model.is_loaded() is False

    def test_finbert_lazy_loading(self, mock_transformers):
        """Test that model is not loaded on init."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()

        assert model._tokenizer is None
        assert model._model is None
        assert model._pipeline is None
        assert model.is_loaded() is False

    @patch('quantdl.models.finbert.FinBERTModel.load')
    def test_predict_triggers_load(self, mock_load, mock_transformers):
        """Test that predict() calls load() if not loaded."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        model._loaded = False

        # Mock the pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = [[
            {'label': 'positive', 'score': 0.9},
            {'label': 'negative', 'score': 0.05},
            {'label': 'neutral', 'score': 0.05}
        ]]
        model._pipeline = mock_pipeline
        model._loaded = True

        results = model.predict(["Test text"])

        assert len(results) == 1
        assert results[0].label == "positive"
        assert results[0].score == 0.9

    def test_predict_empty_input(self, mock_transformers):
        """Test predict with empty input."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        results = model.predict([])

        assert results == []

    def test_batch_size_parameter(self, mock_transformers):
        """Test batch_size parameter."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel(batch_size=16)
        assert model._batch_size == 16

    def test_device_parameter(self, mock_transformers):
        """Test device parameter."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel(device="cpu")
        assert model._device == "cpu"

        model_cuda = FinBERTModel(device="cuda")
        assert model_cuda._device == "cuda"
