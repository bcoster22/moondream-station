import os
import torch
import base64
import io
from PIL import Image
from diffusers import (
    AutoPipelineForText2Image, 
    AutoPipelineForImage2Image, 
    EulerDiscreteScheduler, 
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PipelineQuantizationConfig, 
    AutoencoderKL
)
import logging
import psutil

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

            # Map HuggingFace model IDs to local checkpoint files
            checkpoints_dir = os.path.join(self._models_dir, "checkpoints")
            model_checkpoint_map = {
                "RunDiffusion/Juggernaut-XL-Lightning": "juggernaut-xl-lightning.safetensors",
                "SG161222/RealVisXL_V5.0": "realvisxl-v5.safetensors",
                "cyberdelia/CyberRealisticXL": "cyberrealistic-xl-v80.safetensors",
                "Cyberdelia/CyberRealistic_XL": "cyberrealistic-xl-v80.safetensors",
                "cyberdelia/CyberRealisticPony": "cyberrealistic-pony.safetensors",
                "Lykon/dreamshaper-xl-1-0": "dreamshaper-xl.safetensors",
                "dataautogpt3/ProteusV0.4": "proteus-xl.safetensors",
                "cagliostrolab/animagine-xl-3.1": "animagine-xl.safetensors",
                "imagepipeline/NightVisionXL": "nightvision-xl.safetensors",
                "Disra/NightVisionXL": "nightvision-xl.safetensors",
                "stablediffusionapi/epicrealism-xl-v5": "epicrealism-xl-purefix.safetensors",
                "stablediffusionapi/epicella-xl": "epicella-xl.safetensors",
                "stablediffusionapi/zavychromaxl-v80": "zavychroma-xl.safetensors",
                "Leosam/HelloWorld_XL": "helloworld-xl.safetensors",
                "Copax/Copax_TimeLessXL": "copax-timeless-xl.safetensors",
                "stablediffusionapi/albedobase-xl-v13": "albedobase-xl.safetensors" 
            }
            
            # Check if we have a local checkpoint file
            # Robust Finder: Look recursively for the file
            checkpoint_file = None
            if self.model_id.endswith(".safetensors"):
                # Handle cases like "checkpoints/my-model.safetensors"
                checkpoint_file = os.path.basename(self.model_id)
            else:
                checkpoint_file = model_checkpoint_map.get(self.model_id)
            
            self.logger.info(f"Checkpoint lookup: model_id={self.model_id}, mapped_file={checkpoint_file}")
            
            checkpoint_path = None
            
            # Roots to search
            diffusers_dir = os.path.join(self._models_dir, "diffusers")
            sdxl_models_dir = os.path.join(self._models_dir, "sdxl-models")
            
            search_roots = [checkpoints_dir, diffusers_dir, sdxl_models_dir]
            self.logger.info(f"Searching for model assets in: {search_roots}")

            if checkpoint_file:
                # STRATEGY A: Find specific .safetensors file
                p1 = os.path.join(checkpoints_dir, checkpoint_file)
                if os.path.exists(p1):
                    checkpoint_path = p1
                    self.logger.info(f"Found checkpoint file (direct): {checkpoint_path}")
                else:
                    # Recursive walk
                    for root_dir in search_roots:
                        if not os.path.exists(root_dir): continue
                        for root, dirs, files in os.walk(root_dir):
                            if checkpoint_file in files:
                                checkpoint_path = os.path.join(root, checkpoint_file)
                                self.logger.info(f"Found checkpoint file (recursive): {checkpoint_path}")
                                break
                        if checkpoint_path: break
            
            # STRATEGY B: Fuzzy match for Diffusers directory
            # We determine this later if checkpoint_path is None OR if single file load fails.
            
            # Determine loading method
            load_success = False
            
            # Attempt 1: Single File
            if checkpoint_path and os.path.isfile(checkpoint_path):
                try:
                    from diffusers import StableDiffusionXLPipeline
                    
                    self.logger.info(f"Loading from single checkpoint file: {checkpoint_path}")
                    self.pipeline = StableDiffusionXLPipeline.from_single_file(
                        checkpoint_path,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        safety_checker=None,
                        feature_extractor=None
                    )
                    load_success = True
                except Exception as e:
                    self.logger.warning(f"Failed to load from single checkpoint: {e}. Falling back to directory search.")
                    checkpoint_path = None # Reset to force directory search
            
                def normalize(s): return s.lower().replace("-", "").replace("_", "")
                
                for root_dir in search_roots:
                    if not os.path.exists(root_dir): continue
                    for root, dirs, files in os.walk(root_dir):
                        if "blobs" in root: continue
                        
                        folder_name = os.path.basename(root)
                        # Robust Fuzzy Matching
                        # 1. Direct containment (case-insensitive)
                        # 2. Normalized containment (ignoring - and _)
                        match = False
                        
                        n_name = normalize(name)
                        n_folder = normalize(folder_name)
                        
                        if name in folder_name.lower(): match = True
                        if folder_name.lower() in name: match = True
                        if n_name in n_folder: match = True
                        if len(n_folder) > 3 and n_folder in n_name: match = True
                        
                        if match: 
                            if "model_index.json" in files:
                                checkpoint_path = root
                                self.logger.info(f"Found Diffusers model root at: {checkpoint_path}")
                                break
                    if checkpoint_path and os.path.isdir(checkpoint_path): break

                if checkpoint_path and os.path.isdir(checkpoint_path):
                    self.logger.info(f"Loading from Diffusers directory: {checkpoint_path}")
                    
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
                        checkpoint_path,
                        torch_dtype=torch.float16,
                        quantization_config=quantization_config,
                        use_safetensors=True,
                        local_files_only=True
                    )
                    load_success = True

            # Attempt 3: HuggingFace Download (Last Resort)
            if not load_success:
                self.logger.info(f"Loading from HuggingFace: {self.model_id}")
                
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
                    use_safetensors=True
                )
                load_success = True

            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config, 
                timestep_spacing="trailing"
            )

            try:
                vae = AutoencoderKL.from_pretrained(
                    self.model_id, 
                    subfolder="vae", 
                    torch_dtype=torch.float16
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

    def get_img2img(self):
        if self.img2img_pipeline:
            return self.img2img_pipeline
        
        self.logger.info("Creating Img2Img pipeline from Text2Image...")
        # AutoPipeline 'from_pipe' shares components (model offload should persist)
        try:
            self.img2img_pipeline = AutoPipelineForImage2Image.from_pipe(self.pipeline)
        except Exception as e:
            self.logger.error(f"Failed to create img2img pipe: {e}")
            raise e
        return self.img2img_pipeline

    def set_scheduler(self, scheduler_name):
        """Set scheduler for God Tier presets"""
        if not self.pipeline: return
        
        try:
            config = self.pipeline.scheduler.config
            
            # DPM++ 2M Variants (Most Common)
            if scheduler_name == "dpm_pp_2m_karras" or scheduler_name == "karras":
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    config, use_karras_sigmas=True
                )
            elif scheduler_name == "dpm_pp_2m_sde_karras" or scheduler_name == "dpmpp_2m_sde_gpu":
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    config, 
                    use_karras_sigmas=True,
                    algorithm_type="sde-dpmsolver++"
                )
            elif scheduler_name == "dpm_pp_2m":
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    config, use_karras_sigmas=False
                )
            
            # God Tier Specialized Schedulers
            elif scheduler_name == "beta":
                # Beta scheduler - smoother gradients for soft portraits
                self.pipeline.scheduler = DDIMScheduler.from_config(
                    config,
                    beta_schedule="scaled_linear",  # Beta schedule variant
                    clip_sample=False
                )
            elif scheduler_name == "exponential":
                # Exponential - aggressive end curve for dark/moody
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    config,
                    use_karras_sigmas=True,
                    final_sigmas_type="zero"  # Exponential-like behavior
                )
            elif scheduler_name == "sgm_uniform":
                # SGM Uniform - for Lightning models
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    config,
                    timestep_spacing="trailing",
                    use_karras_sigmas=False
                )
            elif scheduler_name == "simple" or scheduler_name == "normal":
                # Simple/Normal - basic linear schedule
                self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    config,
                    timestep_spacing="linspace"
                )
            elif scheduler_name == "align_your_steps" or scheduler_name == "ays":
                # AYS - Smart step scheduler for efficiency
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    config,
                    use_karras_sigmas=True,
                    algorithm_type="dpmsolver++"
                )
            
            # Euler Variants
            elif scheduler_name == "euler_a" or scheduler_name == "euler_ancestral":
                self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(config)
            elif scheduler_name == "euler":
                self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    config, timestep_spacing="trailing"
                )
            
            # Additional Samplers
            elif scheduler_name == "heun":
                self.pipeline.scheduler = HeunDiscreteScheduler.from_config(config)
            elif scheduler_name == "dpm2":
                self.pipeline.scheduler = KDPM2DiscreteScheduler.from_config(config)
            elif scheduler_name == "lms":
                self.pipeline.scheduler = LMSDiscreteScheduler.from_config(config)
                
            else:
                # Default to Euler
                self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    config, timestep_spacing="trailing"
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to set scheduler {scheduler_name}: {e}")

    def get_available_schedulers(self):
        """Return comprehensive list of available schedulers with metadata"""
        return [
            {
                "id": "dpm_pp_2m_karras",
                "name": "DPM++ 2M Karras",
                "aliases": ["karras", "dpmpp_2m_karras"],
                "description": "Best quality, works great with 20-35 steps",
                "recommended": True,
                "optimal_steps_min": 20,
                "optimal_steps_max": 35,
                "best_for": "structure",
                "category": "DPM"
            },
            {
                "id": "dpm_pp_2m_sde_karras",
                "name": "DPM++ 2M SDE Karras",
                "aliases": ["dpmpp_2m_sde_gpu", "dpmpp_2m_sde_karras"],
                "description": "Adds grain/texture detail, perfect for skin & fabric",
                "recommended": True,
                "optimal_steps_min": 30,
                "optimal_steps_max": 45,
                "best_for": "texture",
                "category": "DPM"
            },
            {
                "id": "beta",
                "name": "Beta (DDIM)",
                "aliases": ["beta_schedule"],
                "description": "Smoothest gradients for soft portraits",
                "recommended": False,
                "optimal_steps_min": 35,
                "optimal_steps_max": 50,
                "best_for": "portraits",
                "category": "Specialized"
            },
            {
                "id": "exponential",
                "name": "Exponential",
                "aliases": ["exp"],
                "description": "Aggressive end-curve for dark/moody scenes",
                "recommended": False,
                "optimal_steps_min": 35,
                "optimal_steps_max": 45,
                "best_for": "dark",
                "category": "Specialized"
            },
            {
                "id": "align_your_steps",
                "name": "Align Your Steps (AYS)",
                "aliases": ["ays"],
                "description": "Front-loaded efficiency - 30-step quality in 12-15 steps",
                "recommended": False,
                "optimal_steps_min": 12,
                "optimal_steps_max": 20,
                "best_for": "speed",
                "category": "Efficiency"
            },
            {
                "id": "sgm_uniform",
                "name": "SGM Uniform",
                "aliases": ["uniform"],
                "description": "For Lightning models - ultra-fast",
                "recommended": False,
                "optimal_steps_min": 6,
                "optimal_steps_max": 10,
                "best_for": "lightning",
                "category": "Efficiency"
            },
            {
                "id": "simple",
                "name": "Simple",
                "aliases": ["normal", "linear"],
                "description": "Basic linear schedule - predictable",
                "recommended": False,
                "optimal_steps_min": 20,
                "optimal_steps_max": 50,
                "best_for": "general",
                "category": "Basic"
            },
            {
                "id": "dpm_pp_2m",
                "name": "DPM++ 2M",
                "aliases": ["dpmpp_2m"],
                "description": "Good quality without Karras sigmas",
                "recommended": False,
                "optimal_steps_min": 20,
                "optimal_steps_max": 40,
                "best_for": "general",
                "category": "DPM"
            },
            {
                "id": "euler_a",
                "name": "Euler Ancestral",
                "aliases": ["euler_ancestral", "euler_a"],
                "description": "Creative variation, adds noise throughout",
                "recommended": False,
                "optimal_steps_min": 15,
                "optimal_steps_max": 30,
                "best_for": "creative",
                "category": "Euler"
            },
            {
                "id": "euler",
                "name": "Euler",
                "aliases": ["euler_discrete"],
                "description": "Fast and reliable, good for testing",
                "recommended": False,
                "optimal_steps_min": 8,
                "optimal_steps_max": 25,
                "best_for": "testing",
                "category": "Euler"
            },
            {
                "id": "heun",
                "name": "Heun",
                "aliases": [],
                "description": "2nd order solver - more accurate",
                "recommended": False,
                "optimal_steps_min": 15,
                "optimal_steps_max": 30,
                "best_for": "quality",
                "category": "Advanced"
            },
            {
                "id": "dpm2",
                "name": "DPM2",
                "aliases": ["kdpm2"],
                "description": "Alternative 2nd order solver",
                "recommended": False,
                "optimal_steps_min": 20,
                "optimal_steps_max": 35,
                "best_for": "quality",
                "category": "Advanced"
            },
            {
                "id": "lms",
                "name": "LMS",
                "aliases": [],
                "description": "Linear multi-step - stable",
                "recommended": False,
                "optimal_steps_min": 20,
                "optimal_steps_max": 40,
                "best_for": "stable",
                "category": "Basic"
            }
        ]
    
    def get_available_samplers(self):
        """Return list of available samplers (note: samplers are tied to schedulers in diffusers)"""
        return [
            {
                "id": "dpmpp_2m_sde_gpu",
                "name": "DPM++ 2M SDE",
                "description": "Best for skin/fabric texture",
                "scheduler_required": "dpm_pp_2m_sde_karras",
                "recommended": True,
                "category": "Texture"
            },
            {
                "id": "dpmpp_2m",
                "name": "DPM++ 2M",
                "description": "Clean structure, no texture noise",
                "scheduler_required": "dpm_pp_2m_karras",
                "recommended": True,
                "category": "Structure"
            },
            {
                "id": "dpmpp_3m_sde_gpu",
                "name": "DPM++ 3M SDE",
                "description": "Highest detail resolution (slow)",
                "scheduler_required": "dpm_pp_2m_sde_karras",
                "recommended": False,
                "category": "Detail"
            },
            {
                "id": "dpmpp_sde_gpu",
                "name": "DPM++ SDE",
                "description": "For Lightning models",
                "scheduler_required": "sgm_uniform",
                "recommended": False,
                "category": "Speed"
            },
            {
                "id": "euler",
                "name": "Euler",
                "description": "Simple and stable",
                "scheduler_required": "euler",
                "recommended": False,
                "category": "Basic"
            },
            {
                "id": "euler_ancestral",
                "name": "Euler Ancestral",
                "description": "Creative with variation",
                "scheduler_required": "euler_a",
                "recommended": False,
                "category": "Creative"
            },
            {
                "id": "restart",
                "name": "Restart",
                "description": "Self-correcting for anatomy",
                "scheduler_required": "karras",
                "recommended": False,
                "category": "Specialist"
            }
        ]

    def generate_image(self, prompt, negative_prompt="", steps=4, guidance_scale=1.5, width=1024, height=1024, num_images=1, image=None, strength=0.75, scheduler="euler", **kwargs):
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")
            
        try:
            # Configure Scheduler
            self.set_scheduler(scheduler)

            with torch.inference_mode():
                if image:
                    # Img2Img
                    self.logger.info(f"Generating Img2Img: '{prompt}' (Steps: {steps}, Strength: {strength})")
                    pipe = self.get_img2img()
                    
                    # Clean input
                    if isinstance(image, str) and "," in image:
                        image = image.split(",")[1]
                    
                    init_image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
                    init_image = init_image.resize((int(width), int(height)))
                    
                    self.logger.info(f"Input Image Size: {init_image.size}")

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
                    # Text2Image
                    self.logger.info(f"Generating Text2Image: '{prompt}'")
                    images = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance_scale),
                        width=int(width),
                        height=int(height),
                        num_images_per_prompt=int(num_images)
                    ).images

            results = []
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                results.append(img_str)

            # Measure Peak VRAM
            peak_vram = 0
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            
            return {
                "images": results,
                "stats": {
                    "peak_vram_mb": peak_vram / (1024 * 1024)
                }
            }

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
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
    global BACKEND
    import gc
    import torch
    
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
    # We run this multiple times to catch circular references and unreachables
    for _ in range(3):
        gc.collect()
    
    # Force VRAM API Clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    print(f"[SDXL-Backend] Backend unloaded. RAM: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
    print("[SDXL-Backend] Backend unloaded and memory cleared.")

def is_model_downloaded(model_id):
    """
    Check if a model is downloaded locally.
    Reuses the robust logic from SDXLBackend.initialize but returns boolean.
    """
    models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    sdxl_models_dir = os.path.join(models_dir, "sdxl-models")
    
    # Map HuggingFace model IDs to local checkpoint files (Need to keep this in sync with SDXLBackend)
    model_checkpoint_map = {
        "RunDiffusion/Juggernaut-XL-Lightning": "juggernaut-xl-lightning.safetensors",
        "SG161222/RealVisXL_V5.0": "realvisxl-v5.safetensors",
        "cyberdelia/CyberRealisticXL": "cyberrealistic-xl-v80.safetensors",
        "Cyberdelia/CyberRealistic_XL": "cyberrealistic-xl-v80.safetensors",
        "cyberdelia/CyberRealisticPony": "cyberrealistic-pony.safetensors",
        "Lykon/dreamshaper-xl-1-0": "dreamshaper-xl.safetensors",
        "dataautogpt3/ProteusV0.4": "proteus-xl.safetensors",
        "cagliostrolab/animagine-xl-3.1": "animagine-xl.safetensors",
        "imagepipeline/NightVisionXL": "nightvision-xl.safetensors",
        "Disra/NightVisionXL": "nightvision-xl.safetensors",
        "stablediffusionapi/epicrealism-xl-v5": "epicrealism-xl-purefix.safetensors",
        "stablediffusionapi/epicella-xl": "epicella-xl.safetensors",
        "stablediffusionapi/zavychromaxl-v80": "zavychroma-xl.safetensors",
        "Leosam/HelloWorld_XL": "helloworld-xl.safetensors",
        "Copax/Copax_TimeLessXL": "copax-timeless-xl.safetensors",
        "stablediffusionapi/albedobase-xl-v13": "albedobase-xl.safetensors"
    }

    checkpoint_file = model_checkpoint_map.get(model_id)
    if not checkpoint_file:
        return False
        
    # 1. Direct check
    if os.path.exists(os.path.join(checkpoints_dir, checkpoint_file)):
        return True
        
    # 2. Recursive check
    def normalize(s): return s.lower().replace("-", "").replace("_", "")
    
    search_roots = [checkpoints_dir, sdxl_models_dir]
    for root_dir in search_roots:
        if not os.path.exists(root_dir): continue
        for root, dirs, files in os.walk(root_dir):
            if checkpoint_file in files:
                return True
            if "diffusion_pytorch_model.safetensors" in files or "model_index.json" in files:
                # Check parent folder match with robust fuzzy logic
                folder_name = os.path.basename(root)
                n_id = normalize(model_id.split("/")[-1])
                n_folder = normalize(folder_name)
                
                check_match = False
                if model_id.split("/")[-1] in folder_name.lower(): check_match = True
                if folder_name.lower() in model_id.split("/")[-1]: check_match = True
                if n_id in n_folder: check_match = True
                if len(n_folder) > 3 and n_folder in n_id: check_match = True
                
                if check_match:
                    return True
    
    return False

def get_model_file_details(model_id):
    """
    Get the file path and size (in bytes) of a model.
    Returns: (path, size_bytes) or (None, 0) if not found.
    """
    models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    sdxl_models_dir = os.path.join(models_dir, "sdxl-models")
    
    # Map HuggingFace model IDs to local checkpoint files (Need to keep this in sync with SDXLBackend)
    model_checkpoint_map = {
        "RunDiffusion/Juggernaut-XL-Lightning": "juggernaut-xl-lightning.safetensors",
        "SG161222/RealVisXL_V5.0": "realvisxl-v5.safetensors",
        "cyberdelia/CyberRealisticXL": "cyberrealistic-xl-v80.safetensors",
        "Cyberdelia/CyberRealistic_XL": "cyberrealistic-xl-v80.safetensors",
        "cyberdelia/CyberRealisticPony": "cyberrealistic-pony.safetensors",
        "Lykon/dreamshaper-xl-1-0": "dreamshaper-xl.safetensors",
        "dataautogpt3/ProteusV0.4": "proteus-xl.safetensors",
        "cagliostrolab/animagine-xl-3.1": "animagine-xl.safetensors",
        "imagepipeline/NightVisionXL": "nightvision-xl.safetensors",
        "Disra/NightVisionXL": "nightvision-xl.safetensors",
        "stablediffusionapi/epicrealism-xl-v5": "epicrealism-xl-purefix.safetensors",
        "stablediffusionapi/epicella-xl": "epicella-xl.safetensors",
        "stablediffusionapi/zavychromaxl-v80": "zavychroma-xl.safetensors",
        "Leosam/HelloWorld_XL": "helloworld-xl.safetensors",
        "Copax/Copax_TimeLessXL": "copax-timeless-xl.safetensors",
        "stablediffusionapi/albedobase-xl-v13": "albedobase-xl.safetensors"
    }

    checkpoint_file = model_checkpoint_map.get(model_id)
    
    # 1. Direct Checkpoint File Check
    if checkpoint_file:
        p1 = os.path.join(checkpoints_dir, checkpoint_file)
        if os.path.exists(p1):
            return p1, os.path.getsize(p1)
            
    # 2. Recursive Search for File or Folder
    def normalize(s): return s.lower().replace("-", "").replace("_", "")

    search_roots = [checkpoints_dir, sdxl_models_dir]
    for root_dir in search_roots:
        if not os.path.exists(root_dir): continue
        for root, dirs, files in os.walk(root_dir):
            if checkpoint_file and checkpoint_file in files:
                p = os.path.join(root, checkpoint_file)
                return p, os.path.getsize(p)
                
            if "model_index.json" in files or "diffusion_pytorch_model.safetensors" in files:
                 folder_name = os.path.basename(root)
                 n_id = normalize(model_id.split("/")[-1])
                 n_folder = normalize(folder_name)
                 
                 check_match = False
                 if model_id.split("/")[-1] in folder_name.lower(): check_match = True
                 if folder_name.lower() in model_id.split("/")[-1]: check_match = True
                 if n_id in n_folder: check_match = True
                 if len(n_folder) > 3 and n_folder in n_id: check_match = True

                 if check_match:
                     if "diffusion_pytorch_model.safetensors" in files:
                         p = os.path.join(root, "diffusion_pytorch_model.safetensors")
                         return p, os.path.getsize(p)
                     return root, 0
                     
    return None, 0
