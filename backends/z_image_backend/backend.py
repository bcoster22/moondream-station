import os
import torch
import logging
import base64
import io
import gc
from PIL import Image
from diffusers import AutoPipelineForText2Image
from transformers import BitsAndBytesConfig
from accelerate import init_empty_weights

logger = logging.getLogger(__name__)

# No global pipeline storage
_model_args = {}
_models_dir = None

def init_backend(**kwargs):
    """Initialize the backend with model arguments"""
    global _model_args, _models_dir
    _model_args = kwargs
    
    # Define models directory relative to this backend file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _models_dir = os.path.join(base_dir, "models")
    os.makedirs(_models_dir, exist_ok=True)
    
    logger.info(f"Z-Image Backend initialized (Ephemeral 8-bit Balanced). Models directory: {_models_dir}")

def create_ephemeral_pipeline():
    """Creates a pipeline, used for a single run, then discarded."""
    gc.collect()
    torch.cuda.empty_cache()
    
    model_id = _model_args.get("model_id", "Tongyi-MAI/Z-Image-Turbo")
    logger.info(f"EPHEMERAL LOAD: {model_id} (8-bit + 3GB Limit)")
    
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True # Allow CPU offload for 8bit
        )
        
        # 3GB VRAM Limit
        max_memory = {0: "3GB", "cpu": "16GB"}
        offload_folder = os.path.join(_models_dir, "offload_buffer_8bit")
        os.makedirs(offload_folder, exist_ok=True)
        
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            quantization_config=quantization_config, # 8-bit
            torch_dtype=torch.float16,
            cache_dir=_models_dir,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder
        )
        
        logger.info("Pipeline loaded (8-bit + Offload).")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise e

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate(prompt: str, negative_prompt: str = "", steps: int = 8, width: int = 1024, height: int = 1024, guidance_scale: float = 3.5, **kwargs):
    pipeline = None
    try:
        pipeline = create_ephemeral_pipeline()
        
        logger.info(f"Generating: {prompt}")
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]
        
        img_str = image_to_base64(image)
        logger.info("Generation complete.")
        
        return {"image": f"data:image/png;base64,{img_str}"}

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {"error": str(e)}
        
    finally:
        if pipeline:
            del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Pipeline unloaded.")

def run_generation(prompt: str, **kwargs):
    return generate(prompt, **kwargs)
