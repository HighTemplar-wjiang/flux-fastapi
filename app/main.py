import io
import os
import threading
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from diffusers import FluxPipeline


# Global initialization with environment-configurable defaults
MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
DEFAULT_GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", "0.0"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "4"))
DEFAULT_MAX_SEQ_LEN = int(os.getenv("DEFAULT_MAX_SEQ_LEN", "256"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "true").lower() in ("1", "true", "yes", "y")
TORCH_DTYPE_STR = os.getenv("TORCH_DTYPE", "bfloat16").lower()  # bfloat16|float16|float32


def _parse_dtype(dtype_str: str) -> torch.dtype:
	dtype_str = dtype_str.lower()
	if dtype_str in ("bf16", "bfloat16"):
		return torch.bfloat16
	if dtype_str in ("fp16", "float16", "half"):
		return torch.float16
	if dtype_str in ("fp32", "float32"):
		return torch.float32
	# default
	return torch.bfloat16


requested_dtype: torch.dtype = _parse_dtype(TORCH_DTYPE_STR)

app = FastAPI(title="FLUX Image Generation API", version="0.1.0")

_pipeline_lock = threading.Lock()
_pipeline: Optional[FluxPipeline] = None


def _get_pipeline() -> FluxPipeline:
	global _pipeline
	if _pipeline is not None:
		return _pipeline
	with _pipeline_lock:
		if _pipeline is not None:
			return _pipeline

		pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=requested_dtype)
		if ENABLE_CPU_OFFLOAD:
			pipe.enable_model_cpu_offload()
		_pipeline = pipe
		return _pipeline


class GenerateRequest(BaseModel):
	prompt: str = Field(..., description="Text prompt to generate the image")
	guidance_scale: Optional[float] = Field(None, description="Classifier-free guidance scale")
	num_inference_steps: Optional[int] = Field(None, description="Number of diffusion steps")
	max_sequence_length: Optional[int] = Field(None, description="Max sequence length for the text encoder")
	seed: Optional[int] = Field(None, description="Random seed for reproducibility")
	height: Optional[int] = Field(None, description="Image height")
	width: Optional[int] = Field(None, description="Image width")
	# If true, return image/png bytes directly; otherwise still returns image/png
	# Kept for future extensibility (e.g., base64/json), but for now always returns PNG bytes
	stream_png: Optional[bool] = Field(True, description="Return raw PNG bytes")


@app.get("/health")
def health() -> dict:
	return {"status": "ok", "model": MODEL_ID}


@app.post("/generate", response_class=Response)
def generate(request: GenerateRequest):
	if not request.prompt or not request.prompt.strip():
		raise HTTPException(status_code=400, detail="prompt must be provided")

	pipe = _get_pipeline()

	guidance_scale = request.guidance_scale if request.guidance_scale is not None else DEFAULT_GUIDANCE
	num_steps = request.num_inference_steps if request.num_inference_steps is not None else DEFAULT_STEPS
	max_seq_len = request.max_sequence_length if request.max_sequence_length is not None else DEFAULT_MAX_SEQ_LEN
	height = request.height if request.height is not None else DEFAULT_HEIGHT
	width = request.width if request.width is not None else DEFAULT_WIDTH

	# Generator on CPU is commonly used for reproducibility regardless of GPU availability
	generator = None
	if request.seed is not None:
		generator = torch.Generator(device="cpu").manual_seed(int(request.seed))

	try:
		with torch.inference_mode():
			image = pipe(
				request.prompt,
				guidance_scale=guidance_scale,
				num_inference_steps=num_steps,
				max_sequence_length=max_seq_len,
				height=height,
				width=width,
				generator=generator,
			).images[0]
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

	buf = io.BytesIO()
	image.save(buf, format="PNG")
	png_bytes = buf.getvalue()
	return Response(content=png_bytes, media_type="image/png")

