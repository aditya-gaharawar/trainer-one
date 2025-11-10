"""
Trainer One - Backend API
Professional AI Model Fine-tuning Platform
Provides REST API endpoints for model inference and training management
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv

from model_manager import model_manager
from training_manager import training_manager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Trainer One API",
    description="Professional AI Model Fine-tuning Platform - Backend API for model inference and training management",
    version="1.0.0",
    contact={
        "name": "Trainer One",
        "url": "https://github.com/aditya-gaharawar/trainer-one",
    },
    license_info={
        "name": "LGPL-3.0",
    }
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class ModelLoadRequest(BaseModel):
    model_name: str = Field(default="unsloth/gpt-oss-20b", description="HuggingFace model name")
    max_seq_length: int = Field(default=1024, description="Maximum sequence length")
    load_in_4bit: bool = Field(default=True, description="Use 4-bit quantization")


class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    reasoning_effort: str = Field(default="medium", description="Reasoning level: low, medium, or high")
    max_new_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stream: bool = Field(default=False, description="Enable streaming response")


class TrainingConfigRequest(BaseModel):
    model_name: str = Field(default="unsloth/gpt-oss-20b")
    dataset_name: str = Field(default="HuggingFaceH4/Multilingual-Thinking")
    dataset_split: str = Field(default="train[:1000]")
    max_seq_length: int = Field(default=1024)
    load_in_4bit: bool = Field(default=True)
    lora_r: int = Field(default=8, description="LoRA rank")
    lora_alpha: int = Field(default=16, description="LoRA alpha")
    lora_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    warmup_steps: int = Field(default=5, ge=0)
    max_steps: int = Field(default=30, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    optimizer: str = Field(default="adamw_8bit")
    weight_decay: float = Field(default=0.001, ge=0)
    lr_scheduler_type: str = Field(default="linear")
    seed: int = Field(default=3407)
    train_on_responses_only: bool = Field(default=True)


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "name": "Trainer One API",
        "description": "Professional AI Model Fine-tuning Platform",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded
    }


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.post("/model/load")
async def load_model(request: ModelLoadRequest):
    """Load a model for inference"""
    try:
        result = model_manager.load_model(
            model_name=request.model_name,
            max_seq_length=request.max_seq_length,
            load_in_4bit=request.load_in_4bit
        )
        return result
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/unload")
async def unload_model():
    """Unload the current model"""
    try:
        result = model_manager.unload_model()
        return result
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/status")
async def model_status():
    """Get current model status"""
    try:
        return model_manager.get_status()
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Inference Endpoints
# ============================================================================

@app.post("/inference")
async def inference(request: InferenceRequest):
    """Generate text from the model (non-streaming)"""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=400,
            detail="Model not loaded. Please load a model first using /model/load"
        )

    try:
        result = model_manager.generate(
            prompt=request.prompt,
            reasoning_effort=request.reasoning_effort,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/inference/stream")
async def inference_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming text generation"""
    await websocket.accept()

    try:
        # Receive request data
        data = await websocket.receive_json()

        prompt = data.get("prompt")
        reasoning_effort = data.get("reasoning_effort", "medium")
        max_new_tokens = data.get("max_new_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)

        if not prompt:
            await websocket.send_json({"error": "Prompt is required"})
            await websocket.close()
            return

        if not model_manager.is_loaded:
            await websocket.send_json({"error": "Model not loaded"})
            await websocket.close()
            return

        # Send start signal
        await websocket.send_json({"type": "start"})

        # Stream generation
        async for token in model_manager.generate_stream(
            prompt=prompt,
            reasoning_effort=reasoning_effort,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        ):
            await websocket.send_json({"type": "token", "content": token})

        # Send completion signal
        await websocket.send_json({"type": "complete"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket: {str(e)}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


# ============================================================================
# Training Management Endpoints
# ============================================================================

@app.get("/training/config/default")
async def get_default_config():
    """Get default training configuration"""
    return training_manager.get_default_config()


@app.post("/training/create")
async def create_training_job(config: TrainingConfigRequest):
    """Create a new training job"""
    try:
        job = training_manager.create_job(config.model_dump())
        return {
            "status": "success",
            "job": job.to_dict()
        }
    except Exception as e:
        logger.error(f"Error creating training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/start/{job_id}")
async def start_training_job(job_id: str):
    """Start a training job"""
    try:
        result = training_manager.start_job(job_id)

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/jobs")
async def list_training_jobs():
    """List all training jobs"""
    try:
        jobs = training_manager.list_jobs()
        return {"jobs": jobs}
    except Exception as e:
        logger.error(f"Error listing training jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/job/{job_id}")
async def get_training_job(job_id: str):
    """Get status of a specific training job"""
    try:
        job = training_manager.get_job_status(job_id)

        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/cancel/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a running training job"""
    try:
        result = training_manager.cancel_job(job_id)

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
