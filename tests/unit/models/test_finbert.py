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

    def test_logger_parameter(self, mock_transformers):
        """Test custom logger parameter."""
        from quantdl.models.finbert import FinBERTModel
        import logging

        custom_logger = logging.getLogger("test_logger")
        model = FinBERTModel(logger=custom_logger)
        assert model._logger is custom_logger

    def test_default_logger(self, mock_transformers):
        """Test default logger is created."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        assert model._logger is not None


class TestFinBERTModelLoad:
    """Tests for FinBERT model loading."""

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers and torch imports."""
        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'torch': MagicMock(),
        }):
            yield

    def test_load_already_loaded(self, mock_transformers):
        """Test that load() returns early if already loaded."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        model._loaded = True

        # Should return immediately without doing anything
        model.load()
        assert model._loaded is True

    @patch('quantdl.models.finbert.FinBERTModel._detect_device')
    def test_load_uses_auto_detect_when_no_device(self, mock_detect, mock_transformers):
        """Test that load() auto-detects device when none specified."""
        from quantdl.models.finbert import FinBERTModel

        mock_detect.return_value = "cpu"

        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'torch': MagicMock(),
        }):
            model = FinBERTModel(device=None)

            # Mock the imports within load()
            with patch('quantdl.models.finbert.FinBERTModel.load') as mock_load:
                mock_load.return_value = None
                model.load()

    def test_detect_device_cpu_fallback(self, mock_transformers):
        """Test _detect_device returns cpu when torch unavailable."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()

        # Mock torch.cuda.is_available to return False
        with patch.dict('sys.modules', {'torch': MagicMock()}):
            import sys
            sys.modules['torch'].cuda.is_available.return_value = False
            result = model._detect_device()
            assert result == "cpu"

    def test_detect_device_with_import_error(self, mock_transformers):
        """Test _detect_device handles ImportError gracefully."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()

        # Temporarily remove torch from modules to simulate ImportError
        with patch.dict('sys.modules', {'torch': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                result = model._detect_device()
                assert result == "cpu"


class TestFinBERTModelUnload:
    """Tests for FinBERT model unloading."""

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers and torch imports."""
        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'torch': MagicMock(),
        }):
            yield

    def test_unload_when_not_loaded(self, mock_transformers):
        """Test that unload() returns early if not loaded."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        model._loaded = False

        # Should return without error
        model.unload()
        assert model._loaded is False

    def test_unload_clears_attributes(self, mock_transformers):
        """Test that unload() clears model attributes."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        model._loaded = True
        model._pipeline = Mock()
        model._model = Mock()
        model._tokenizer = Mock()

        with patch.dict('sys.modules', {'torch': MagicMock()}):
            model.unload()

        assert model._pipeline is None
        assert model._model is None
        assert model._tokenizer is None
        assert model._loaded is False

    def test_unload_handles_exception(self, mock_transformers):
        """Test that unload() handles exceptions gracefully."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        model._loaded = True
        model._pipeline = Mock()
        model._model = Mock()
        model._tokenizer = Mock()

        # Mock torch import to raise an exception
        with patch.dict('sys.modules', {'torch': None}):
            with patch('builtins.__import__', side_effect=ImportError("No torch")):
                # Should not raise, just log warning
                model.unload()


class TestFinBERTModelPredict:
    """Tests for FinBERT model prediction."""

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers and torch imports."""
        with patch.dict('sys.modules', {
            'transformers': MagicMock(),
            'torch': MagicMock(),
        }):
            yield

    def test_predict_batching(self, mock_transformers):
        """Test that predict processes in batches."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel(batch_size=2)
        model._loaded = True

        # Mock pipeline to return results
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            [{'label': 'positive', 'score': 0.9}, {'label': 'negative', 'score': 0.1}],
            [{'label': 'negative', 'score': 0.8}, {'label': 'positive', 'score': 0.2}],
        ]
        model._pipeline = mock_pipeline

        texts = ["text1", "text2", "text3", "text4"]
        results = model.predict(texts)

        # Should have called pipeline twice (batch_size=2, 4 texts)
        assert mock_pipeline.call_count == 2
        assert len(results) == 4

    def test_predict_truncates_long_text_in_result(self, mock_transformers):
        """Test that predict truncates long text in SentimentResult."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        model._loaded = True

        mock_pipeline = Mock()
        mock_pipeline.return_value = [[
            {'label': 'positive', 'score': 0.9},
        ]]
        model._pipeline = mock_pipeline

        long_text = "A" * 200  # 200 character text
        results = model.predict([long_text])

        # Result text_chunk should be truncated with "..."
        assert len(results[0].text_chunk) == 103  # 100 chars + "..."
        assert results[0].text_chunk.endswith("...")

    def test_predict_lowercase_label(self, mock_transformers):
        """Test that predict lowercases the label."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        model._loaded = True

        mock_pipeline = Mock()
        mock_pipeline.return_value = [[
            {'label': 'POSITIVE', 'score': 0.9},
            {'label': 'NEGATIVE', 'score': 0.1},
        ]]
        model._pipeline = mock_pipeline

        results = model.predict(["test"])

        assert results[0].label == "positive"

    def test_predict_selects_highest_score(self, mock_transformers):
        """Test that predict selects the label with highest score."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        model._loaded = True

        mock_pipeline = Mock()
        mock_pipeline.return_value = [[
            {'label': 'positive', 'score': 0.3},
            {'label': 'negative', 'score': 0.6},
            {'label': 'neutral', 'score': 0.1},
        ]]
        model._pipeline = mock_pipeline

        results = model.predict(["test"])

        assert results[0].label == "negative"
        assert results[0].score == 0.6

    def test_predict_includes_model_info(self, mock_transformers):
        """Test that predict includes model name and version."""
        from quantdl.models.finbert import FinBERTModel

        model = FinBERTModel()
        model._loaded = True

        mock_pipeline = Mock()
        mock_pipeline.return_value = [[
            {'label': 'positive', 'score': 0.9},
        ]]
        model._pipeline = mock_pipeline

        results = model.predict(["test"])

        assert results[0].model_name == "finbert"
        assert results[0].model_version == "1.0.0"
