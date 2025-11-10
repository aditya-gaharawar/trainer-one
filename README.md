# GPT-OSS Fine-tuning Platform

A full-stack web application for fine-tuning and interacting with OpenAI's GPT-OSS language models using the Unsloth framework.

## Features

### Backend (FastAPI)
- **Model Management**: Load/unload GPT-OSS models (20B/120B parameters)
- **Inference API**: REST and WebSocket endpoints for text generation
- **Training Management**: Create, start, monitor, and cancel fine-tuning jobs
- **Memory Efficient**: 4-bit quantization and LoRA support
- **Real-time Streaming**: WebSocket support for streaming responses

### Frontend (React + Vite)
- **Chat Interface**: Interactive chat with adjustable reasoning effort
- **Training Dashboard**: Configure and monitor training jobs
- **Model Manager**: Easy model loading and configuration
- **Real-time Updates**: Live training metrics and streaming responses
- **Modern UI**: Built with Tailwind CSS and Lucide icons

## Architecture

```
trainer-one/
├── backend/                    # FastAPI backend
│   ├── main.py                # API endpoints
│   ├── model_manager.py       # Model loading & inference
│   ├── training_manager.py    # Training job management
│   ├── requirements.txt       # Python dependencies
│   └── .env.example          # Environment variables template
│
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── TrainingDashboard.jsx
│   │   │   └── ModelManager.jsx
│   │   ├── services/
│   │   │   └── api.js        # API client
│   │   ├── App.jsx           # Main app component
│   │   └── main.jsx          # Entry point
│   ├── package.json
│   └── vite.config.js
│
└── gpt_oss_(20B)_Fine_tuning.ipynb.txt  # Original notebook
```

## Prerequisites

- **Python**: 3.9 or higher
- **Node.js**: 16 or higher
- **GPU**: CUDA-capable GPU recommended (Tesla T4 or better)
- **RAM**: 16GB minimum (32GB+ recommended)
- **Storage**: 50GB+ for models

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd trainer-one
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) If unsloth is not available, install it
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Copy and configure environment variables
cp .env.example .env
# Edit .env if needed
```

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Copy and configure environment variables
cp .env.example .env
# Edit .env if needed (default: VITE_API_URL=http://localhost:8000)
```

## Running the Application

### Start the Backend

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

The backend will start on `http://localhost:8000`

You can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Start the Frontend

```bash
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

## Usage Guide

### 1. Load a Model

1. Open the application at `http://localhost:3000`
2. In the **Model Manager** panel (left sidebar):
   - Select a model (e.g., `unsloth/gpt-oss-20b`)
   - Configure max sequence length (default: 1024)
   - Enable 4-bit quantization (recommended)
   - Click **Load Model**

**Note**: Model loading can take several minutes depending on your hardware and internet connection.

### 2. Chat with the Model

1. Navigate to the **Chat** tab
2. Configure parameters:
   - **Reasoning Effort**: low, medium, or high (controls thinking depth)
   - **Temperature**: 0.0 (deterministic) to 2.0 (creative)
   - **Max Tokens**: Maximum length of generated response
   - **Streaming**: Enable for real-time token generation
3. Type your message and press **Send** or hit Enter
4. Watch the model respond in real-time (if streaming is enabled)

### 3. Fine-tune the Model

1. Navigate to the **Training** tab
2. Click **Show Config** to customize training parameters:
   - Model name and dataset
   - LoRA configuration (rank, alpha, dropout)
   - Training hyperparameters (learning rate, batch size, steps)
3. Click **Start New Training Job**
4. Monitor progress with:
   - Real-time progress bar
   - Step counter
   - Training metrics (loss, learning rate, epoch)
5. View training history at the bottom of the page

## API Endpoints

### Model Management
- `POST /model/load` - Load a model
- `POST /model/unload` - Unload the current model
- `GET /model/status` - Get model status

### Inference
- `POST /inference` - Generate text (non-streaming)
- `WebSocket /inference/stream` - Generate text (streaming)

### Training
- `GET /training/config/default` - Get default training config
- `POST /training/create` - Create a training job
- `POST /training/start/{job_id}` - Start a training job
- `GET /training/jobs` - List all training jobs
- `GET /training/job/{job_id}` - Get job status
- `POST /training/cancel/{job_id}` - Cancel a training job

## Configuration

### Backend Environment Variables

```bash
MODEL_NAME=unsloth/gpt-oss-20b    # Default model
MAX_SEQ_LENGTH=1024                # Max sequence length
LOAD_IN_4BIT=true                  # Use 4-bit quantization
DEVICE=cuda                         # cuda or cpu
HOST=0.0.0.0                       # Server host
PORT=8000                          # Server port
```

### Frontend Environment Variables

```bash
VITE_API_URL=http://localhost:8000  # Backend API URL
```

## Training Configuration

Default training parameters:

```json
{
  "model_name": "unsloth/gpt-oss-20b",
  "dataset_name": "HuggingFaceH4/Multilingual-Thinking",
  "dataset_split": "train[:1000]",
  "max_seq_length": 1024,
  "load_in_4bit": true,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0,
  "batch_size": 1,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 5,
  "max_steps": 30,
  "learning_rate": 0.0002,
  "optimizer": "adamw_8bit",
  "weight_decay": 0.001,
  "lr_scheduler_type": "linear",
  "seed": 3407,
  "train_on_responses_only": true
}
```

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Unsloth**: Fast LLM fine-tuning framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **TRL**: Transformer Reinforcement Learning
- **BitsAndBytes**: Quantization library

### Frontend
- **React**: UI library
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Icon library
- **Axios**: HTTP client

## Performance Tips

1. **Use 4-bit Quantization**: Reduces memory usage by ~75%
2. **Adjust Sequence Length**: Lower values use less memory
3. **LoRA Training**: Only trains 1% of parameters
4. **Gradient Checkpointing**: Saves 30% VRAM during training
5. **Batch Size**: Use 1 for consumer GPUs
6. **Streaming**: Enable for better UX with long responses

## Troubleshooting

### Model Loading Issues
- Ensure you have enough GPU memory
- Try 4-bit quantization
- Check CUDA installation: `torch.cuda.is_available()`
- Verify internet connection for model downloads

### Training Failures
- Reduce batch size or max sequence length
- Ensure dataset is accessible
- Check GPU memory availability
- Monitor system resources

### WebSocket Connection Issues
- Check firewall settings
- Verify backend is running
- Ensure correct API URL in frontend `.env`

## Development

### Backend Development

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd frontend
npm run dev
```

### Build for Production

```bash
# Frontend
cd frontend
npm run build
npm run preview
```

## Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [OpenAI Harmony Format](https://github.com/openai/harmony)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

## License

This project uses the LGPL-3.0 license (as per the original notebook).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Built with [Unsloth](https://unsloth.ai/) for fast LLM fine-tuning
- Uses OpenAI's GPT-OSS models
- Dataset: [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)
