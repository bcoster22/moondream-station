import os
import torch
import logging
import psutil
import gc
from diffusers import AutoPipelineForImage2Image

# Import extracted modules
from .schedulers import create_scheduler, get_available_schedulers as get_schedulers_list, get_available_samplers as get_samplers_list
from .model_loader import find_local_checkpoint, load_pipeline, is_model_downloaded as check_model_downloaded, get_model_file_details as get_model_details
from .image_generator import generate_image as gen_image

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
        self.img2img_pipeline = None  # Lazy load
        self._models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))

    def initialize(self):
        """Initialize SDXL pipeline using model_loader"""
        try:
            self.logger.info(f"Initializing SDXL Backend with model: {self.model_id}")
            
            # Find local checkpoint
            checkpoint_path, is_directory = find_local_checkpoint(self.model_id, self._models_dir)
            
            # Load pipeline
            self.pipeline, success = load_pipeline(
                checkpoint_path, 
                is_directory, 
                self.config, 
                self.device
            )
            
            if success:
                self.logger.info("SDXL Model loaded successfully.")
                return True
            else:
                self.logger.error("Failed to load SDXL pipeline")
                return False

        except Exception as e:
            self.logger.error(f"Failed to initialize SDXL backend: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_img2img(self):
        """Lazy-load img2img pipeline from text2img pipeline"""
        if self.img2img_pipeline:
            return self.img2img_pipeline
        
        self.logger.info("Creating Img2Img pipeline from Text2Image...")
        try:
            self.img2img_pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline)
        except Exception as e:
            self.logger.error(f"Failed to create img2img pipe: {e}")
            raise e
        return self.img2img_pipeline

    def set_scheduler(self, scheduler_name):
        """Set scheduler using schedulers module"""
        if not self.pipeline:
            return
        
        self.pipeline.scheduler = create_scheduler(scheduler_name, self.pipeline.scheduler.config)

    def get_available_schedulers(self):
        """Return list of available schedulers from schedulers module"""
        return get_schedulers_list()
    
    def get_available_samplers(self):
        """Return list of available samplers from schedulers module"""
        return get_samplers_list()

    def generate_image(self, prompt, negative_prompt="", steps=4, guidance_scale=1.5, 
                      width=1024, height=1024, num_images=1, image=None, strength=0.75, 
                      scheduler="euler", **kwargs):
        """Generate image using image_generator module"""
        return gen_image(
            pipeline=self.pipeline,
            img2img_pipeline_getter=self.get_img2img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images=num_images,
            image=image,
            strength=strength,
            scheduler_setter=self.set_scheduler,
            scheduler=scheduler,
            **kwargs
        )


# Module-level API functions
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


def get_available_schedulers():
    """Get list of available schedulers with metadata"""
    if not BACKEND:
        # Return default list even if backend not initialized
        return [
            {"id": "dpm_pp_2m_karras", "name": "DPM++ 2M Karras", "description": "Best quality, works great with 20-35 steps", "recommended": True, "optimal_steps_min": 20, "optimal_steps_max": 35},
            {"id": "dpm_pp_2m", "name": "DPM++ 2M", "description": "Good quality without Karras sigmas", "recommended": False, "optimal_steps_min": 20, "optimal_steps_max": 40},
            {"id": "euler_a", "name": "Euler Ancestral", "description": "More creative variation between steps", "recommended": False, "optimal_steps_min": 15, "optimal_steps_max": 30},
            {"id": "euler", "name": "Euler", "description": "Fast and reliable, good for testing", "recommended": False, "optimal_steps_min": 8, "optimal_steps_max": 25}
        ]
    return BACKEND.get_available_schedulers()


def get_available_samplers():
    """Get list of available samplers with metadata"""
    if not BACKEND:
        return [
            {"id": "dpmpp_2m_sde_gpu", "name": "DPM++ 2M SDE", "description": "Best for texture", "recommended": True},
            {"id": "dpmpp_2m", "name": "DPM++ 2M", "description": "Best for structure", "recommended": True},
            {"id": "euler", "name": "Euler", "description": "Simple and stable", "recommended": False}
        ]
    return BACKEND.get_available_samplers()


def unload_backend():
    """Unload backend and free memory"""
    global BACKEND
    
    print("[SDXL-Backend] Unloading backend...")
    
    if BACKEND:
        try:
            if hasattr(BACKEND, 'pipeline') and BACKEND.pipeline:
                del BACKEND.pipeline
        except Exception as e:
            print(f"[SDXL-Backend] Error deleting pipeline: {e}")
            
        try:
            if hasattr(BACKEND, 'img2img_pipeline') and BACKEND.img2img_pipeline:
                del BACKEND.img2img_pipeline
        except Exception as e:
            print(f"[SDXL-Backend] Error deleting img2img_pipeline: {e}")
            
        BACKEND = None
        
    # Force Garbage Collection (Crucial for System RAM)
    for _ in range(3):
        gc.collect()
    
    # Force VRAM Clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    print(f"[SDXL-Backend] Backend unloaded. RAM: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    print("[SDXL-Backend] Backend unloaded and memory cleared.")


def is_model_downloaded(model_id):
    """Check if a model is downloaded locally (uses model_loader)"""
    return check_model_downloaded(model_id)


def get_model_file_details(model_id):
    """Get the file path and size of a model (uses model_loader)"""
    return get_model_details(model_id)
