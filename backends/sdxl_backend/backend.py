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
        
        self.model_id = config.get("model_id", "RunDiffusion/Juggernaut-XL-Lightning")
        self.use_4bit = config.get("use_4bit", True)
        self.compile = config.get("compile", False) 
        
        self.pipeline = None
        self.img2img_pipeline = None # Lazy load
        self._models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))

    def initialize(self):
        try:
            self.logger.info(f"Initializing SDXL Backend with model: {self.model_id}")

            quantization_config = None
            if self.use_4bit:
                quantization_config = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.float16,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4"
                    }
                )

            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                cache_dir=os.path.join(self._models_dir, "models")
            )

            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config, 
                timestep_spacing="trailing"
            )

            try:
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
            return False

    def get_img2img(self):
        if self.img2img_pipeline:
            return self.img2img_pipeline
        self.img2img_pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline)
        return self.img2img_pipeline

    def generate_image(self, prompt, negative_prompt="", steps=4, guidance_scale=1.5, width=1024, height=1024, num_images=1, image=None, strength=0.75, **kwargs):
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
            
        try:
            # DEBUG SAVE
            debug_path = "/home/bcoster/moondream_debug_init.png"
            log_path = "/home/bcoster/moondream_params.txt"
            
            if image:
                if isinstance(image, str) and "," in image:
                    image = image.split(",")[1]
                
                init_image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
                init_image = init_image.resize((int(width), int(height)))
                
                # Save input to prove what we received
                init_image.save(debug_path)
                
                with open(log_path, "w") as f:
                    f.write(f"Mode: Img2Img\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Strength: {strength} (Type: {type(strength)})\n")
                    f.write(f"Steps: {steps}\n")
                    f.write(f"Input Saved to: {debug_path}\n")

                with torch.inference_mode():
                    pipe = self.get_img2img()
                    images = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=init_image,
                        strength=float(strength),
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance_scale),
                        num_images_per_prompt=int(num_images)
                    ).images
            else:
                 with open(log_path, "w") as f:
                    f.write(f"Mode: Text2Image (Image arg missing/None)\n")
                 
                 images = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=int(steps),
                    # ...
                 ).images

            results = []
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                results.append(img_str)
            return results

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            with open("/home/bcoster/moondream_error.txt", "w") as f:
                f.write(str(e))
            torch.cuda.empty_cache()
            raise e

def init_backend(model_id=None, **kwargs):
    global BACKEND
    config = kwargs.copy()
    if model_id: config['model_id'] = model_id
    BACKEND = SDXLBackend(config)
    return BACKEND.initialize()

def generate(prompt, **kwargs):
    if not BACKEND: return {"error": "Backend not initialized"}
    return BACKEND.generate_image(prompt, **kwargs)

def images(**kwargs):
    if not BACKEND: return {"error": "Backend not initialized"}
    return BACKEND.generate_image(**kwargs)
