import os
import torch
import base64
import io
from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, EulerDiscreteScheduler, PipelineQuantizationConfig, AutoencoderKL
import logging

# Global instance
BACKEND = None

class SDXLBackend:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger("sdxl_backend")
        
        # Default models (can be overridden by manifest args)
        self.model_id = config.get("model_id", "RunDiffusion/Juggernaut-XL-Lightning")
        self.use_4bit = config.get("use_4bit", True)
        self.compile = config.get("compile", False) # Optional torch.compile
        
        # Pipeline storage
        self.pipeline = None
        self._models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))

    def initialize(self):
        """
        Load the SDXL model with 4-bit quantization for 8GB VRAM compatibility.
        """
        try:
            self.logger.info(f"Initializing SDXL Backend with model: {self.model_id}")
            self.logger.info(f"4-bit Quantization: {self.use_4bit}")

            # 4-bit Config using PipelineQuantizationConfig
            quantization_config = None
            if self.use_4bit:
                # We use the generic wrapper required by AutoPipeline
                quantization_config = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.float16,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4"
                    }
                )

            # Load Pipeline
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                cache_dir=os.path.join(self._models_dir, "models")
            )

            # Optimizations for SDXL Lightning
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config, 
                timestep_spacing="trailing"
            )

            # Fix for "Casting a quantized model to new dtype is unsupported"
            # We explicitly load VAE in float32 to avoid quantization issues
            # The VAE is small enough to fit in VRAM alongside 4-bit UNet
            try:
                self.logger.info("Reloading VAE in float16 to fix quantization casting...")
                vae = AutoencoderKL.from_pretrained(
                    self.model_id, 
                    subfolder="vae", 
                    torch_dtype=torch.float16,
                    cache_dir=os.path.join(self._models_dir, "models")
                )
                self.pipeline.vae = vae
            except Exception as e:
                self.logger.warning(f"Could not reload VAE: {e}")

            self.pipeline.enable_model_cpu_offload()

            self.logger.info("SDXL Model loaded successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize SDXL backend: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_image(self, prompt, negative_prompt="", steps=4, guidance_scale=1.5, width=1024, height=1024, num_images=1, **kwargs):
        """
        Generate an image using the loaded pipeline.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")

        self.logger.info(f"Generating image: '{prompt}' (Steps: {steps}, CFG: {guidance_scale})")
            
        try:
            with torch.inference_mode():
                images = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    width=int(width),
                    height=int(height),
                    num_images_per_prompt=int(num_images)
                ).images

            # Convert to base64
            results = []
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                results.append(img_str)

            return results

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            if "out of memory" in str(e).lower():
                 # Attempt cleanup
                 torch.cuda.empty_cache()
            raise e

# Module-level interface
def init_backend(model_id=None, **kwargs):
    global BACKEND
    config = kwargs.copy()
    if model_id:
        config['model_id'] = model_id
    
    BACKEND = SDXLBackend(config)
    return BACKEND.initialize()

def generate(prompt, **kwargs):
    if not BACKEND:
        return {"error": "Backend not initialized"}
    return BACKEND.generate_image(prompt, **kwargs)

def images(**kwargs):
    if not BACKEND:
        return {"error": "Backend not initialized"}
    return BACKEND.generate_image(**kwargs)
