"""
Training Manager for GPT-OSS Fine-tuning
Handles training job management and configuration
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingJob:
    """Represents a single training job"""

    def __init__(self, job_id: str, config: Dict[str, Any]):
        self.job_id = job_id
        self.config = config
        self.status = "pending"
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.metrics = {}
        self.current_step = 0
        self.total_steps = config.get("max_steps", 30)

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary"""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "config": self.config,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metrics": self.metrics,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        }


class TrainingManager:
    """Manages training jobs and configurations"""

    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.current_job: Optional[TrainingJob] = None

    def create_job(self, config: Dict[str, Any]) -> TrainingJob:
        """
        Create a new training job

        Args:
            config: Training configuration dictionary

        Returns:
            TrainingJob instance
        """
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = TrainingJob(job_id, config)
        self.jobs[job_id] = job

        logger.info(f"Created training job: {job_id}")
        return job

    def start_job(self, job_id: str) -> Dict[str, Any]:
        """
        Start a training job

        Args:
            job_id: ID of the job to start

        Returns:
            Status dictionary
        """
        if job_id not in self.jobs:
            return {"status": "error", "error": "Job not found"}

        job = self.jobs[job_id]

        if self.current_job and self.current_job.status == "running":
            return {
                "status": "error",
                "error": f"Another job is already running: {self.current_job.job_id}"
            }

        # Start training in a separate thread
        job.status = "running"
        job.started_at = datetime.now()
        self.current_job = job

        thread = Thread(target=self._run_training, args=(job,))
        thread.start()

        logger.info(f"Started training job: {job_id}")

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Training job started"
        }

    def _run_training(self, job: TrainingJob):
        """
        Run the actual training (to be implemented with real training logic)

        Args:
            job: TrainingJob instance
        """
        try:
            logger.info(f"Running training job: {job.job_id}")

            # This is a placeholder - in production, this would run the actual training
            # from the notebook code with proper integration

            config = job.config

            # Simulated training steps (replace with actual training logic)
            total_steps = config.get("max_steps", 30)

            for step in range(total_steps):
                # Simulate training step
                job.current_step = step + 1

                # Simulate metrics
                job.metrics = {
                    "loss": 2.5 - (step / total_steps) * 2.0,  # Simulated decreasing loss
                    "learning_rate": config.get("learning_rate", 2e-4),
                    "epoch": step / (total_steps / config.get("num_epochs", 1))
                }

                logger.info(f"Job {job.job_id} - Step {step + 1}/{total_steps}")

                # Check if job should be stopped
                if job.status == "cancelled":
                    logger.info(f"Job {job.job_id} cancelled")
                    return

            # Mark job as completed
            job.status = "completed"
            job.completed_at = datetime.now()
            self.current_job = None

            logger.info(f"Job {job.job_id} completed successfully")

        except Exception as e:
            logger.error(f"Error in training job {job.job_id}: {str(e)}")
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now()
            self.current_job = None

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific job

        Args:
            job_id: Job identifier

        Returns:
            Job status dictionary or None if not found
        """
        if job_id not in self.jobs:
            return None

        return self.jobs[job_id].to_dict()

    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all training jobs

        Returns:
            List of job dictionaries
        """
        return [job.to_dict() for job in self.jobs.values()]

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running job

        Args:
            job_id: Job identifier

        Returns:
            Status dictionary
        """
        if job_id not in self.jobs:
            return {"status": "error", "error": "Job not found"}

        job = self.jobs[job_id]

        if job.status != "running":
            return {"status": "error", "error": f"Job is not running (status: {job.status})"}

        job.status = "cancelled"
        logger.info(f"Job {job_id} marked for cancellation")

        return {
            "status": "success",
            "message": f"Job {job_id} cancelled"
        }

    def get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            "model_name": "unsloth/gpt-oss-20b",
            "dataset_name": "HuggingFaceH4/Multilingual-Thinking",
            "dataset_split": "train[:1000]",
            "max_seq_length": 1024,
            "load_in_4bit": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
            "max_steps": 30,
            "learning_rate": 2e-4,
            "optimizer": "adamw_8bit",
            "weight_decay": 0.001,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "train_on_responses_only": True,
        }


# Global training manager instance
training_manager = TrainingManager()
