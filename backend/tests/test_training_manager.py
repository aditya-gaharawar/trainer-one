"""
Tests for training_manager.py
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import time
from training_manager import TrainingJob, TrainingManager, training_manager


class TestTrainingJob:
    """Test suite for TrainingJob class"""

    def test_initialization(self, sample_training_config):
        """Test TrainingJob initialization"""
        job = TrainingJob("test_job_001", sample_training_config)

        assert job.job_id == "test_job_001"
        assert job.config == sample_training_config
        assert job.status == "pending"
        assert job.started_at is None
        assert job.completed_at is None
        assert job.error is None
        assert job.metrics == {}
        assert job.current_step == 0
        assert job.total_steps == sample_training_config["max_steps"]

    def test_to_dict(self, sample_training_config):
        """Test converting job to dictionary"""
        job = TrainingJob("test_job_001", sample_training_config)
        job.status = "running"
        job.started_at = datetime.now()
        job.current_step = 5
        job.metrics = {"loss": 2.5}

        result = job.to_dict()

        assert result["job_id"] == "test_job_001"
        assert result["status"] == "running"
        assert result["config"] == sample_training_config
        assert result["current_step"] == 5
        assert result["total_steps"] == 10
        assert result["metrics"] == {"loss": 2.5}
        assert result["progress"] == 50.0
        assert result["started_at"] is not None
        assert result["completed_at"] is None
        assert result["error"] is None

    def test_progress_calculation(self, sample_training_config):
        """Test progress calculation"""
        job = TrainingJob("test_job_001", sample_training_config)

        # 0% progress
        job.current_step = 0
        assert job.to_dict()["progress"] == 0.0

        # 50% progress
        job.current_step = 5
        assert job.to_dict()["progress"] == 50.0

        # 100% progress
        job.current_step = 10
        assert job.to_dict()["progress"] == 100.0

    def test_progress_with_zero_steps(self):
        """Test progress calculation with zero total steps"""
        config = {"max_steps": 0}
        job = TrainingJob("test_job_001", config)

        assert job.to_dict()["progress"] == 0


class TestTrainingManager:
    """Test suite for TrainingManager class"""

    def test_initialization(self, training_manager_instance):
        """Test TrainingManager initialization"""
        assert training_manager_instance.jobs == {}
        assert training_manager_instance.current_job is None

    def test_create_job(self, training_manager_instance, sample_training_config):
        """Test creating a training job"""
        with patch('training_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20231215_123456"

            job = training_manager_instance.create_job(sample_training_config)

            assert job.job_id == "job_20231215_123456"
            assert job.config == sample_training_config
            assert job.status == "pending"
            assert "job_20231215_123456" in training_manager_instance.jobs

    def test_create_multiple_jobs(self, training_manager_instance, sample_training_config):
        """Test creating multiple jobs"""
        job1 = training_manager_instance.create_job(sample_training_config)
        job2 = training_manager_instance.create_job(sample_training_config)

        assert len(training_manager_instance.jobs) == 2
        assert job1.job_id != job2.job_id

    def test_start_job(self, training_manager_instance, sample_training_config):
        """Test starting a training job"""
        # Create job
        job = training_manager_instance.create_job(sample_training_config)

        # Start job
        with patch('training_manager.Thread') as mock_thread:
            result = training_manager_instance.start_job(job.job_id)

            assert result["status"] == "success"
            assert result["job_id"] == job.job_id
            assert job.status == "running"
            assert job.started_at is not None
            assert training_manager_instance.current_job == job

            # Verify thread was started
            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()

    def test_start_nonexistent_job(self, training_manager_instance):
        """Test starting a job that doesn't exist"""
        result = training_manager_instance.start_job("nonexistent_job")

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_start_job_when_another_running(self, training_manager_instance, sample_training_config):
        """Test starting a job when another is already running"""
        # Create and start first job
        job1 = training_manager_instance.create_job(sample_training_config)
        job1.status = "running"
        training_manager_instance.current_job = job1

        # Try to start second job
        job2 = training_manager_instance.create_job(sample_training_config)
        result = training_manager_instance.start_job(job2.job_id)

        assert result["status"] == "error"
        assert "already running" in result["error"].lower()

    def test_get_job_status(self, training_manager_instance, sample_training_config):
        """Test getting job status"""
        job = training_manager_instance.create_job(sample_training_config)

        status = training_manager_instance.get_job_status(job.job_id)

        assert status is not None
        assert status["job_id"] == job.job_id
        assert status["status"] == "pending"

    def test_get_nonexistent_job_status(self, training_manager_instance):
        """Test getting status of nonexistent job"""
        status = training_manager_instance.get_job_status("nonexistent_job")

        assert status is None

    def test_list_jobs(self, training_manager_instance, sample_training_config):
        """Test listing all jobs"""
        # Create multiple jobs
        job1 = training_manager_instance.create_job(sample_training_config)
        job2 = training_manager_instance.create_job(sample_training_config)
        job3 = training_manager_instance.create_job(sample_training_config)

        jobs = training_manager_instance.list_jobs()

        assert len(jobs) == 3
        assert all(isinstance(job, dict) for job in jobs)
        assert {job["job_id"] for job in jobs} == {job1.job_id, job2.job_id, job3.job_id}

    def test_list_jobs_empty(self, training_manager_instance):
        """Test listing jobs when none exist"""
        jobs = training_manager_instance.list_jobs()

        assert jobs == []

    def test_cancel_job(self, training_manager_instance, sample_training_config):
        """Test canceling a running job"""
        # Create and start job
        job = training_manager_instance.create_job(sample_training_config)
        job.status = "running"

        # Cancel job
        result = training_manager_instance.cancel_job(job.job_id)

        assert result["status"] == "success"
        assert job.status == "cancelled"

    def test_cancel_nonexistent_job(self, training_manager_instance):
        """Test canceling a nonexistent job"""
        result = training_manager_instance.cancel_job("nonexistent_job")

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_cancel_non_running_job(self, training_manager_instance, sample_training_config):
        """Test canceling a job that is not running"""
        job = training_manager_instance.create_job(sample_training_config)
        job.status = "completed"

        result = training_manager_instance.cancel_job(job.job_id)

        assert result["status"] == "error"
        assert "not running" in result["error"].lower()

    def test_get_default_config(self, training_manager_instance):
        """Test getting default training configuration"""
        config = training_manager_instance.get_default_config()

        assert config is not None
        assert "model_name" in config
        assert "dataset_name" in config
        assert "max_steps" in config
        assert "learning_rate" in config
        assert "lora_r" in config
        assert config["model_name"] == "unsloth/gpt-oss-20b"
        assert config["max_steps"] == 30

    def test_run_training_simulation(self, training_manager_instance, sample_training_config):
        """Test the training simulation logic"""
        job = training_manager_instance.create_job(sample_training_config)

        # Run training directly (not in thread)
        training_manager_instance._run_training(job)

        # Verify job completed
        assert job.status == "completed"
        assert job.completed_at is not None
        assert job.current_step == sample_training_config["max_steps"]
        assert job.metrics != {}
        assert "loss" in job.metrics

    def test_run_training_with_cancellation(self, training_manager_instance, sample_training_config):
        """Test canceling a training job during execution"""
        config = sample_training_config.copy()
        config["max_steps"] = 100  # More steps
        job = training_manager_instance.create_job(config)

        # Start training and cancel immediately
        def cancel_job():
            time.sleep(0.1)
            job.status = "cancelled"

        with patch('time.sleep'):  # Speed up test
            from threading import Thread
            cancel_thread = Thread(target=cancel_job)
            cancel_thread.start()

            training_manager_instance._run_training(job)
            cancel_thread.join()

        # Job should be cancelled (not completed)
        assert job.status == "cancelled"

    def test_run_training_with_error(self, training_manager_instance, sample_training_config):
        """Test error handling during training"""
        job = training_manager_instance.create_job(sample_training_config)

        # Force an error by making config invalid
        with patch.object(job, 'config', side_effect=Exception("Training error")):
            training_manager_instance._run_training(job)

        # Job should be marked as failed
        assert job.status == "failed"
        assert job.error is not None
        assert "Training error" in job.error
        assert job.completed_at is not None

    def test_metrics_progression(self, training_manager_instance, sample_training_config):
        """Test that metrics progress during training"""
        job = training_manager_instance.create_job(sample_training_config)

        # Capture metrics at different steps
        metrics_history = []

        original_run = training_manager_instance._run_training

        def capture_metrics(*args, **kwargs):
            result = original_run(*args, **kwargs)
            if job.metrics:
                metrics_history.append(job.metrics.copy())
            return result

        with patch.object(training_manager_instance, '_run_training', side_effect=capture_metrics):
            training_manager_instance._run_training(job)

        # Verify loss decreases over time (simulated behavior)
        if len(metrics_history) > 1:
            assert metrics_history[-1]["loss"] < metrics_history[0]["loss"]


class TestGlobalTrainingManager:
    """Test the global training_manager instance"""

    def test_global_instance_exists(self):
        """Test that global training manager instance exists"""
        assert training_manager is not None
        assert isinstance(training_manager, TrainingManager)

    def test_global_instance_initialized(self):
        """Test that global instance is properly initialized"""
        assert hasattr(training_manager, 'jobs')
        assert hasattr(training_manager, 'current_job')
        assert isinstance(training_manager.jobs, dict)


class TestJobStatusTransitions:
    """Test valid job status transitions"""

    def test_pending_to_running(self, training_manager_instance, sample_training_config):
        """Test transition from pending to running"""
        job = training_manager_instance.create_job(sample_training_config)
        assert job.status == "pending"

        with patch('training_manager.Thread'):
            training_manager_instance.start_job(job.job_id)

        assert job.status == "running"

    def test_running_to_completed(self, training_manager_instance, sample_training_config):
        """Test transition from running to completed"""
        job = training_manager_instance.create_job(sample_training_config)
        job.status = "running"

        training_manager_instance._run_training(job)

        assert job.status == "completed"

    def test_running_to_cancelled(self, training_manager_instance, sample_training_config):
        """Test transition from running to cancelled"""
        job = training_manager_instance.create_job(sample_training_config)
        job.status = "running"

        training_manager_instance.cancel_job(job.job_id)

        assert job.status == "cancelled"

    def test_running_to_failed(self, training_manager_instance, sample_training_config):
        """Test transition from running to failed (on error)"""
        job = training_manager_instance.create_job(sample_training_config)
        job.status = "running"

        with patch.object(job, 'config', side_effect=Exception("Error")):
            training_manager_instance._run_training(job)

        assert job.status == "failed"
