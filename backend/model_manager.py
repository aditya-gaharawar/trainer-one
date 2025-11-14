"""
Model Manager for GPT-OSS Fine-tuning
Handles model loading, inference, and management
"""

import os
import torch
from typing import Optional, Dict, Any, AsyncIterator
from threading import Lock, Thread
from transformers import TextIteratorStreamer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and inference operations"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.is_loaded = False
        self.lock = Lock()

    def load_model(
        self,
        model_name: str = "unsloth/gpt-oss-20b",
        max_seq_length: int = 1024,
        load_in_4bit: bool = True,
        dtype: Any = None
    ) -> Dict[str, Any]:
        """
        Load the GPT-OSS model for inference

        Args:
            model_name: HuggingFace model identifier
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to use 4-bit quantization
            dtype: Data type for model weights

        Returns:
            Dictionary with loading status and info
        """
        with self.lock:
            try:
                logger.info(f"Loading model: {model_name}")

                # Try to import unsloth, fallback to transformers if not available
                try:
                    from unsloth import FastLanguageModel

                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=model_name,
                        max_seq_length=max_seq_length,
                        load_in_4bit=load_in_4bit,
                        dtype=dtype,
                    )

                    # Enable faster inference
                    FastLanguageModel.for_inference(self.model)

                except ImportError:
                    logger.warning("Unsloth not available, using transformers directly")
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        load_in_4bit=load_in_4bit,
                        device_map="auto",
                        torch_dtype=dtype or torch.float16,
                    )

                self.model_name = model_name
                self.is_loaded = True

                logger.info(f"Model loaded successfully: {model_name}")

                return {
                    "status": "success",
                    "model_name": model_name,
                    "max_seq_length": max_seq_length,
                    "loaded": True
                }

            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "loaded": False
                }

    def unload_model(self) -> Dict[str, Any]:
        """Unload the current model to free memory"""
        with self.lock:
            try:
                if self.model is not None:
                    del self.model
                    del self.tokenizer
                    self.model = None
                    self.tokenizer = None
                    self.model_name = None
                    self.is_loaded = False

                    # Clear CUDA cache if available
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    logger.info("Model unloaded successfully")
                    return {"status": "success", "loaded": False}
                else:
                    return {"status": "success", "message": "No model was loaded", "loaded": False}

            except Exception as e:
                logger.error(f"Error unloading model: {str(e)}")
                return {"status": "error", "error": str(e)}

    async def generate_stream(
        self,
        prompt: str,
        reasoning_effort: str = "medium",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming output

        Args:
            prompt: Input prompt
            reasoning_effort: Reasoning level (low, medium, high)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Yields:
            Generated text tokens
        """
        if not self.is_loaded or self.model is None:
            yield "Error: Model not loaded. Please load a model first."
            return

        try:
            # Format prompt with Harmony format
            formatted_prompt = self._format_prompt(prompt, reasoning_effort)

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True
            ).to(self.model.device)

            # Generate with streaming
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "streamer": streamer,
            }

            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream the output
            for text in streamer:
                yield text

            thread.join()

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            yield f"Error: {str(e)}"

    def generate(
        self,
        prompt: str,
        reasoning_effort: str = "medium",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Generate text (non-streaming)

        Args:
            prompt: Input prompt
            reasoning_effort: Reasoning level (low, medium, high)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Dictionary with generated text and metadata
        """
        if not self.is_loaded or self.model is None:
            return {
                "status": "error",
                "error": "Model not loaded. Please load a model first."
            }

        try:
            # Format prompt with Harmony format
            formatted_prompt = self._format_prompt(prompt, reasoning_effort)

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                )

            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return {
                "status": "success",
                "generated_text": generated_text,
                "prompt": prompt,
                "reasoning_effort": reasoning_effort
            }

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _format_prompt(self, prompt: str, reasoning_effort: str = "medium") -> str:
        """
        Format prompt using OpenAI Harmony format

        Args:
            prompt: User's input prompt
            reasoning_effort: Reasoning level

        Returns:
            Formatted prompt string
        """
        # OpenAI Harmony format with reasoning effort
        formatted = (
            f"<|start|>user<|message|>{prompt}<|end|>"
            f"<|start|>assistant<|reasoning_effort|>{reasoning_effort}<|channel|>final<|message|>"
        )
        return formatted

    def get_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "device": str(self.model.device) if self.model else None,
            "cuda_available": torch.cuda.is_available(),
            "cuda_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
        }


# Global model manager instance
model_manager = ModelManager()
