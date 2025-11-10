"""
Integration tests for FastAPI endpoints
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import torch


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_root_endpoint(self, test_client):
        """Test root endpoint"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "GPT-OSS Fine-tuning API"
        assert data["status"] == "running"
        assert "version" in data

    def test_health_endpoint(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestModelManagementEndpoints:
    """Test model management API endpoints"""

    @patch('main.model_manager')
    def test_load_model_success(self, mock_manager, test_client, sample_model_config):
        """Test successful model loading"""
        mock_manager.load_model.return_value = {
            "status": "success",
            "model_name": "test-model",
            "loaded": True
        }

        response = test_client.post("/model/load", json=sample_model_config)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["loaded"] is True

        mock_manager.load_model.assert_called_once()

    @patch('main.model_manager')
    def test_load_model_with_custom_params(self, mock_manager, test_client):
        """Test loading model with custom parameters"""
        mock_manager.load_model.return_value = {"status": "success", "loaded": True}

        config = {
            "model_name": "custom-model",
            "max_seq_length": 2048,
            "load_in_4bit": False
        }

        response = test_client.post("/model/load", json=config)

        assert response.status_code == 200
        mock_manager.load_model.assert_called_once_with(
            model_name="custom-model",
            max_seq_length=2048,
            load_in_4bit=False
        )

    @patch('main.model_manager')
    def test_load_model_error(self, mock_manager, test_client, sample_model_config):
        """Test model loading error handling"""
        mock_manager.load_model.side_effect = Exception("Loading failed")

        response = test_client.post("/model/load", json=sample_model_config)

        assert response.status_code == 500

    @patch('main.model_manager')
    def test_unload_model_success(self, mock_manager, test_client):
        """Test successful model unloading"""
        mock_manager.unload_model.return_value = {
            "status": "success",
            "loaded": False
        }

        response = test_client.post("/model/unload")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["loaded"] is False

        mock_manager.unload_model.assert_called_once()

    @patch('main.model_manager')
    def test_unload_model_error(self, mock_manager, test_client):
        """Test model unloading error handling"""
        mock_manager.unload_model.side_effect = Exception("Unload failed")

        response = test_client.post("/model/unload")

        assert response.status_code == 500

    @patch('main.model_manager')
    @patch('torch.cuda.is_available', return_value=True)
    def test_model_status(self, mock_cuda, mock_manager, test_client):
        """Test getting model status"""
        mock_manager.get_status.return_value = {
            "loaded": True,
            "model_name": "test-model",
            "device": "cuda:0",
            "cuda_available": True,
            "cuda_memory_allocated": 1024 * 1024 * 1024
        }

        response = test_client.get("/model/status")

        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] is True
        assert data["model_name"] == "test-model"
        assert data["cuda_available"] is True


class TestInferenceEndpoints:
    """Test inference API endpoints"""

    @patch('main.model_manager')
    def test_inference_success(self, mock_manager, test_client, sample_inference_request):
        """Test successful inference"""
        mock_manager.is_loaded = True
        mock_manager.generate.return_value = {
            "status": "success",
            "generated_text": "This is a generated response",
            "prompt": sample_inference_request["prompt"],
            "reasoning_effort": "medium"
        }

        response = test_client.post("/inference", json=sample_inference_request)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "generated_text" in data
        assert data["generated_text"] == "This is a generated response"

    @patch('main.model_manager')
    def test_inference_without_model(self, mock_manager, test_client, sample_inference_request):
        """Test inference without loaded model"""
        mock_manager.is_loaded = False

        response = test_client.post("/inference", json=sample_inference_request)

        assert response.status_code == 400
        assert "not loaded" in response.json()["detail"].lower()

    @patch('main.model_manager')
    def test_inference_with_custom_params(self, mock_manager, test_client):
        """Test inference with custom parameters"""
        mock_manager.is_loaded = True
        mock_manager.generate.return_value = {"status": "success", "generated_text": "Response"}

        request_data = {
            "prompt": "Custom prompt",
            "reasoning_effort": "high",
            "max_new_tokens": 1024,
            "temperature": 0.9,
            "top_p": 0.95
        }

        response = test_client.post("/inference", json=request_data)

        assert response.status_code == 200

        # Verify parameters were passed
        call_args = mock_manager.generate.call_args[1]
        assert call_args["reasoning_effort"] == "high"
        assert call_args["max_new_tokens"] == 1024
        assert call_args["temperature"] == 0.9
        assert call_args["top_p"] == 0.95

    @patch('main.model_manager')
    def test_inference_error_handling(self, mock_manager, test_client, sample_inference_request):
        """Test inference error handling"""
        mock_manager.is_loaded = True
        mock_manager.generate.return_value = {
            "status": "error",
            "error": "Generation failed"
        }

        response = test_client.post("/inference", json=sample_inference_request)

        assert response.status_code == 500

    def test_inference_validation(self, test_client):
        """Test request validation"""
        # Missing prompt
        response = test_client.post("/inference", json={
            "reasoning_effort": "medium",
            "max_new_tokens": 512
        })

        assert response.status_code == 422  # Validation error


class TestTrainingEndpoints:
    """Test training management API endpoints"""

    @patch('main.training_manager')
    def test_get_default_config(self, mock_manager, test_client):
        """Test getting default training config"""
        mock_config = {
            "model_name": "test-model",
            "max_steps": 30,
            "learning_rate": 2e-4
        }
        mock_manager.get_default_config.return_value = mock_config

        response = test_client.get("/training/config/default")

        assert response.status_code == 200
        data = response.json()
        assert data == mock_config

    @patch('main.training_manager')
    def test_create_training_job(self, mock_manager, test_client, sample_training_config):
        """Test creating a training job"""
        mock_job = MagicMock()
        mock_job.to_dict.return_value = {
            "job_id": "test_job_001",
            "status": "pending",
            "config": sample_training_config
        }
        mock_manager.create_job.return_value = mock_job

        response = test_client.post("/training/create", json=sample_training_config)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "job" in data
        assert data["job"]["job_id"] == "test_job_001"

    @patch('main.training_manager')
    def test_create_training_job_error(self, mock_manager, test_client, sample_training_config):
        """Test error handling when creating job"""
        mock_manager.create_job.side_effect = Exception("Creation failed")

        response = test_client.post("/training/create", json=sample_training_config)

        assert response.status_code == 500

    @patch('main.training_manager')
    def test_start_training_job(self, mock_manager, test_client):
        """Test starting a training job"""
        mock_manager.start_job.return_value = {
            "status": "success",
            "job_id": "test_job_001",
            "message": "Training job started"
        }

        response = test_client.post("/training/start/test_job_001")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["job_id"] == "test_job_001"

    @patch('main.training_manager')
    def test_start_training_job_not_found(self, mock_manager, test_client):
        """Test starting a nonexistent job"""
        mock_manager.start_job.return_value = {
            "status": "error",
            "error": "Job not found"
        }

        response = test_client.post("/training/start/nonexistent")

        assert response.status_code == 400

    @patch('main.training_manager')
    def test_list_training_jobs(self, mock_manager, test_client):
        """Test listing all training jobs"""
        mock_jobs = [
            {"job_id": "job_001", "status": "completed"},
            {"job_id": "job_002", "status": "running"},
        ]
        mock_manager.list_jobs.return_value = mock_jobs

        response = test_client.get("/training/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert len(data["jobs"]) == 2

    @patch('main.training_manager')
    def test_get_training_job(self, mock_manager, test_client):
        """Test getting a specific training job"""
        mock_job = {
            "job_id": "test_job_001",
            "status": "running",
            "progress": 50.0
        }
        mock_manager.get_job_status.return_value = mock_job

        response = test_client.get("/training/job/test_job_001")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test_job_001"
        assert data["status"] == "running"

    @patch('main.training_manager')
    def test_get_training_job_not_found(self, mock_manager, test_client):
        """Test getting a nonexistent job"""
        mock_manager.get_job_status.return_value = None

        response = test_client.get("/training/job/nonexistent")

        assert response.status_code == 404

    @patch('main.training_manager')
    def test_cancel_training_job(self, mock_manager, test_client):
        """Test canceling a training job"""
        mock_manager.cancel_job.return_value = {
            "status": "success",
            "message": "Job cancelled"
        }

        response = test_client.post("/training/cancel/test_job_001")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @patch('main.training_manager')
    def test_cancel_training_job_error(self, mock_manager, test_client):
        """Test error when canceling job"""
        mock_manager.cancel_job.return_value = {
            "status": "error",
            "error": "Job not running"
        }

        response = test_client.post("/training/cancel/test_job_001")

        assert response.status_code == 400


class TestRequestValidation:
    """Test API request validation"""

    def test_load_model_validation(self, test_client):
        """Test model loading request validation"""
        # Invalid model config
        response = test_client.post("/model/load", json={
            "max_seq_length": "invalid"  # Should be int
        })

        assert response.status_code == 422

    def test_inference_validation(self, test_client):
        """Test inference request validation"""
        # Temperature out of range
        response = test_client.post("/inference", json={
            "prompt": "Test",
            "temperature": 3.0  # Max is 2.0
        })

        assert response.status_code == 422

    def test_training_config_validation(self, test_client):
        """Test training config validation"""
        # Invalid batch size
        response = test_client.post("/training/create", json={
            "model_name": "test",
            "batch_size": 0  # Must be > 0
        })

        assert response.status_code == 422


class TestWebSocketInference:
    """Test WebSocket streaming inference"""

    @patch('main.model_manager')
    def test_websocket_connection(self, mock_manager, test_client):
        """Test WebSocket connection for streaming"""
        mock_manager.is_loaded = True

        # Mock async generator
        async def mock_generate_stream(*args, **kwargs):
            yield "Hello"
            yield " "
            yield "World"

        mock_manager.generate_stream = mock_generate_stream

        with test_client.websocket_connect("/inference/stream") as websocket:
            # Send request
            websocket.send_json({
                "prompt": "Test prompt",
                "reasoning_effort": "medium",
                "max_new_tokens": 256
            })

            # Receive start signal
            data = websocket.receive_json()
            assert data["type"] == "start"

            # Receive tokens
            tokens = []
            while True:
                data = websocket.receive_json()
                if data["type"] == "complete":
                    break
                elif data["type"] == "token":
                    tokens.append(data["content"])

            assert tokens == ["Hello", " ", "World"]

    @patch('main.model_manager')
    def test_websocket_without_model(self, mock_manager, test_client):
        """Test WebSocket when model not loaded"""
        mock_manager.is_loaded = False

        with test_client.websocket_connect("/inference/stream") as websocket:
            websocket.send_json({
                "prompt": "Test prompt"
            })

            data = websocket.receive_json()
            assert "error" in data
            assert "not loaded" in data["error"].lower()

    @patch('main.model_manager')
    def test_websocket_missing_prompt(self, mock_manager, test_client):
        """Test WebSocket with missing prompt"""
        mock_manager.is_loaded = True

        with test_client.websocket_connect("/inference/stream") as websocket:
            websocket.send_json({
                "reasoning_effort": "medium"
            })

            data = websocket.receive_json()
            assert "error" in data
            assert "required" in data["error"].lower()


class TestCORSHeaders:
    """Test CORS configuration"""

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present"""
        response = test_client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )

        # CORS middleware should add these headers
        assert "access-control-allow-origin" in response.headers
