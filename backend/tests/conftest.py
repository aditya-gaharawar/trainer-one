"""
Pytest configuration and fixtures
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock, AsyncMock
import torch


@pytest.fixture
def mock_torch_cuda():
    """Mock torch.cuda functions"""
    with pytest.mock.patch('torch.cuda.is_available', return_value=True):
        with pytest.mock.patch('torch.cuda.memory_allocated', return_value=1024 * 1024 * 1024):
            with pytest.mock.patch('torch.cuda.empty_cache'):
                yield


@pytest.fixture
def mock_model():
    """Mock language model"""
    model = MagicMock()
    model.device = 'cuda:0'
    model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer"""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    tokenizer.decode = MagicMock(return_value="Generated response")
    return tokenizer


@pytest.fixture
def mock_unsloth():
    """Mock Unsloth FastLanguageModel"""
    with pytest.mock.patch('model_manager.FastLanguageModel') as mock:
        mock.from_pretrained = MagicMock()
        mock.for_inference = MagicMock()
        yield mock


@pytest.fixture
def mock_transformers():
    """Mock transformers AutoModel and AutoTokenizer"""
    with pytest.mock.patch('model_manager.AutoModelForCausalLM') as mock_model:
        with pytest.mock.patch('model_manager.AutoTokenizer') as mock_tokenizer:
            yield mock_model, mock_tokenizer


@pytest.fixture
def model_manager_instance():
    """Create a fresh ModelManager instance"""
    from model_manager import ModelManager
    manager = ModelManager()
    yield manager
    # Cleanup
    manager.model = None
    manager.tokenizer = None
    manager.is_loaded = False


@pytest.fixture
def training_manager_instance():
    """Create a fresh TrainingManager instance"""
    from training_manager import TrainingManager
    manager = TrainingManager()
    yield manager
    # Cleanup
    manager.jobs.clear()
    manager.current_job = None


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    from main import app
    client = TestClient(app)
    return client


@pytest.fixture
def sample_training_config():
    """Sample training configuration"""
    return {
        "model_name": "test-model",
        "dataset_name": "test-dataset",
        "dataset_split": "train[:100]",
        "max_seq_length": 512,
        "load_in_4bit": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "max_steps": 10,
        "learning_rate": 2e-4,
        "optimizer": "adamw_8bit",
        "weight_decay": 0.001,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "train_on_responses_only": True,
    }


@pytest.fixture
def sample_inference_request():
    """Sample inference request"""
    return {
        "prompt": "What is the capital of France?",
        "reasoning_effort": "medium",
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }


@pytest.fixture
def sample_model_config():
    """Sample model configuration"""
    return {
        "model_name": "unsloth/gpt-oss-20b",
        "max_seq_length": 1024,
        "load_in_4bit": True
    }
