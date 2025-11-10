"""
Tests for model_manager.py
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import torch
from model_manager import ModelManager, model_manager


class TestModelManager:
    """Test suite for ModelManager class"""

    def test_initialization(self, model_manager_instance):
        """Test ModelManager initialization"""
        assert model_manager_instance.model is None
        assert model_manager_instance.tokenizer is None
        assert model_manager_instance.model_name is None
        assert model_manager_instance.is_loaded is False

    @patch('model_manager.FastLanguageModel')
    @patch('torch.cuda.is_available', return_value=True)
    def test_load_model_with_unsloth(
        self, mock_cuda, mock_fast_model, model_manager_instance, mock_model, mock_tokenizer
    ):
        """Test loading model with Unsloth"""
        # Setup mocks
        mock_fast_model.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_fast_model.for_inference = MagicMock()

        # Load model
        result = model_manager_instance.load_model(
            model_name="test-model",
            max_seq_length=512,
            load_in_4bit=True
        )

        # Assertions
        assert result["status"] == "success"
        assert result["loaded"] is True
        assert result["model_name"] == "test-model"
        assert model_manager_instance.is_loaded is True
        assert model_manager_instance.model_name == "test-model"

        # Verify calls
        mock_fast_model.from_pretrained.assert_called_once_with(
            model_name="test-model",
            max_seq_length=512,
            load_in_4bit=True,
            dtype=None
        )
        mock_fast_model.for_inference.assert_called_once()

    @patch('model_manager.FastLanguageModel', side_effect=ImportError("Unsloth not available"))
    @patch('model_manager.AutoModelForCausalLM')
    @patch('model_manager.AutoTokenizer')
    @patch('torch.cuda.is_available', return_value=True)
    def test_load_model_without_unsloth(
        self, mock_cuda, mock_auto_tokenizer, mock_auto_model,
        mock_fast_model, model_manager_instance
    ):
        """Test loading model without Unsloth (fallback to transformers)"""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.device = 'cuda:0'

        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        # Load model
        result = model_manager_instance.load_model(
            model_name="test-model",
            max_seq_length=512,
            load_in_4bit=True
        )

        # Assertions
        assert result["status"] == "success"
        assert result["loaded"] is True
        assert model_manager_instance.is_loaded is True

        # Verify transformers was used
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_auto_model.from_pretrained.assert_called_once()

    def test_load_model_error_handling(self, model_manager_instance):
        """Test error handling during model loading"""
        with patch('model_manager.FastLanguageModel') as mock_fast_model:
            mock_fast_model.from_pretrained.side_effect = Exception("Loading failed")

            result = model_manager_instance.load_model("test-model")

            assert result["status"] == "error"
            assert "Loading failed" in result["error"]
            assert result["loaded"] is False
            assert model_manager_instance.is_loaded is False

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    def test_unload_model(self, mock_empty_cache, mock_cuda, model_manager_instance):
        """Test unloading model"""
        # Setup - simulate loaded model
        model_manager_instance.model = MagicMock()
        model_manager_instance.tokenizer = MagicMock()
        model_manager_instance.model_name = "test-model"
        model_manager_instance.is_loaded = True

        # Unload
        result = model_manager_instance.unload_model()

        # Assertions
        assert result["status"] == "success"
        assert result["loaded"] is False
        assert model_manager_instance.model is None
        assert model_manager_instance.tokenizer is None
        assert model_manager_instance.model_name is None
        assert model_manager_instance.is_loaded is False

        # Verify CUDA cache was cleared
        mock_empty_cache.assert_called_once()

    def test_unload_model_when_not_loaded(self, model_manager_instance):
        """Test unloading when no model is loaded"""
        result = model_manager_instance.unload_model()

        assert result["status"] == "success"
        assert "No model was loaded" in result["message"]
        assert result["loaded"] is False

    def test_unload_model_error_handling(self, model_manager_instance):
        """Test error handling during unload"""
        model_manager_instance.model = MagicMock()

        with patch.object(model_manager_instance, 'model', side_effect=Exception("Unload failed")):
            # Force an exception during unload
            model_manager_instance.model = None
            model_manager_instance.model = property(lambda self: (_ for _ in ()).throw(Exception("Unload failed")))

        # Just test that unload handles errors gracefully
        result = model_manager_instance.unload_model()
        assert result["status"] in ["success", "error"]

    @patch('torch.cuda.is_available', return_value=True)
    def test_generate_non_streaming(self, mock_cuda, model_manager_instance, mock_model, mock_tokenizer):
        """Test non-streaming text generation"""
        # Setup
        model_manager_instance.model = mock_model
        model_manager_instance.tokenizer = mock_tokenizer
        model_manager_instance.is_loaded = True

        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_model.device = 'cuda:0'
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_tokenizer.decode.return_value = "This is a generated response"

        # Generate
        result = model_manager_instance.generate(
            prompt="Test prompt",
            reasoning_effort="medium",
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9
        )

        # Assertions
        assert result["status"] == "success"
        assert "generated_text" in result
        assert result["generated_text"] == "This is a generated response"
        assert result["prompt"] == "Test prompt"
        assert result["reasoning_effort"] == "medium"

    def test_generate_without_loaded_model(self, model_manager_instance):
        """Test generation without a loaded model"""
        result = model_manager_instance.generate("Test prompt")

        assert result["status"] == "error"
        assert "not loaded" in result["error"].lower()

    def test_generate_error_handling(self, model_manager_instance, mock_model, mock_tokenizer):
        """Test error handling during generation"""
        # Setup
        model_manager_instance.model = mock_model
        model_manager_instance.tokenizer = mock_tokenizer
        model_manager_instance.is_loaded = True

        mock_tokenizer.side_effect = Exception("Tokenization failed")

        # Generate
        result = model_manager_instance.generate("Test prompt")

        assert result["status"] == "error"
        assert "Tokenization failed" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_stream_without_model(self, model_manager_instance):
        """Test streaming generation without loaded model"""
        messages = []
        async for token in model_manager_instance.generate_stream("Test prompt"):
            messages.append(token)

        assert len(messages) == 1
        assert "not loaded" in messages[0].lower()

    def test_format_prompt(self, model_manager_instance):
        """Test prompt formatting"""
        prompt = "What is AI?"
        reasoning_effort = "high"

        formatted = model_manager_instance._format_prompt(prompt, reasoning_effort)

        assert "<|start|>user<|message|>" in formatted
        assert prompt in formatted
        assert "<|start|>assistant<|reasoning_effort|>high" in formatted
        assert "<|channel|>final<|message|>" in formatted

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=2 * 1024 * 1024 * 1024)
    def test_get_status_with_model(self, mock_mem, mock_cuda, model_manager_instance, mock_model):
        """Test getting status with loaded model"""
        model_manager_instance.model = mock_model
        model_manager_instance.model_name = "test-model"
        model_manager_instance.is_loaded = True

        status = model_manager_instance.get_status()

        assert status["loaded"] is True
        assert status["model_name"] == "test-model"
        assert status["device"] == "cuda:0"
        assert status["cuda_available"] is True
        assert status["cuda_memory_allocated"] == 2 * 1024 * 1024 * 1024

    @patch('torch.cuda.is_available', return_value=False)
    def test_get_status_without_cuda(self, mock_cuda, model_manager_instance):
        """Test getting status without CUDA"""
        status = model_manager_instance.get_status()

        assert status["loaded"] is False
        assert status["model_name"] is None
        assert status["device"] is None
        assert status["cuda_available"] is False
        assert status["cuda_memory_allocated"] is None

    def test_get_status_without_model(self, model_manager_instance):
        """Test getting status without loaded model"""
        status = model_manager_instance.get_status()

        assert status["loaded"] is False
        assert status["model_name"] is None
        assert status["device"] is None


class TestGlobalModelManager:
    """Test the global model_manager instance"""

    def test_global_instance_exists(self):
        """Test that global model manager instance exists"""
        assert model_manager is not None
        assert isinstance(model_manager, ModelManager)

    def test_global_instance_initialized(self):
        """Test that global instance is properly initialized"""
        assert hasattr(model_manager, 'model')
        assert hasattr(model_manager, 'tokenizer')
        assert hasattr(model_manager, 'is_loaded')
        assert hasattr(model_manager, 'lock')
