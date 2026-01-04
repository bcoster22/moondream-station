"""
Model Discovery and Loading for SDXL Backend
Handles local checkpoint search, fuzzy matching, and pipeline initialization.
"""

import os
import torch
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    PipelineQuantizationConfig,
    AutoencoderKL
)
import logging

logger = logging.getLogger("sdxl_backend.model_loader")


def get_model_checkpoint_map():
    """
    Mapping of HuggingFace model IDs to local checkpoint filenames.
    Single source of truth for model paths.
    """
    return {
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


def _normalize(s):
    """Normalize string for fuzzy matching (lowercase, remove hyphens/underscores)"""
    return s.lower().replace("-", "").replace("_", "")


def find_local_checkpoint(model_id, models_dir):
    """
    Find local checkpoint file or directory for a model ID.
    Returns: (path, is_directory) or (None, False) if not found.
    
    Search strategies:
    1. Direct checkpoint file lookup
    2. Recursive checkpoint file search
    3. Fuzzy match for Diffusers directory
    """
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    diffusers_dir = os.path.join(models_dir, "diffusers")
    sdxl_models_dir = os.path.join(models_dir, "sdxl-models")
    
    search_roots = [checkpoints_dir, diffusers_dir, sdxl_models_dir]
    
    # Get checkpoint filename from map
    model_checkpoint_map = get_model_checkpoint_map()
    checkpoint_file = None
    
    if model_id.endswith(".safetensors"):
        checkpoint_file = os.path.basename(model_id)
    else:
        checkpoint_file = model_checkpoint_map.get(model_id)
    
    logger.info(f"Checkpoint lookup: model_id={model_id}, mapped_file={checkpoint_file}")
    logger.info(f"Searching in: {search_roots}")
    
    # Strategy A: Find specific .safetensors file
    if checkpoint_file:
        # Direct check
        p1 = os.path.join(checkpoints_dir, checkpoint_file)
        if os.path.exists(p1):
            logger.info(f"Found checkpoint file (direct): {p1}")
            return p1, False
        
        # Recursive search
        for root_dir in search_roots:
            if not os.path.exists(root_dir):
                continue
            for root, dirs, files in os.walk(root_dir):
                if checkpoint_file in files:
                    path = os.path.join(root, checkpoint_file)
                    logger.info(f"Found checkpoint file (recursive): {path}")
                    return path, False
    
    # Strategy B: Fuzzy match for Diffusers directory
    # Extract name for fuzzy matching
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    
    for root_dir in search_roots:
        if not os.path.exists(root_dir):
            continue
        for root, dirs, files in os.walk(root_dir):
            if "blobs" in root:
                continue
            
            folder_name = os.path.basename(root)
            
            # Robust Fuzzy Matching
            n_name = _normalize(name)
            n_folder = _normalize(folder_name)
            
            match = False
            if name in folder_name.lower(): match = True
            if folder_name.lower() in name: match = True
            if n_name in n_folder: match = True
            if len(n_folder) > 3 and n_folder in n_name: match = True
            
            if match and "model_index.json" in files:
                logger.info(f"Found Diffusers model root at: {root}")
                return root, True
    
    logger.warning(f"No local checkpoint found for {model_id}")
    return None, False


def load_pipeline(checkpoint_path, is_directory, config, device="cuda"):
    """
    Load SDXL pipeline from local checkpoint or directory.
    Returns: (pipeline, success)
    """
    use_4bit = config.get("use_4bit", True)
    compile_mode = config.get("compile", False)
    model_id = config.get("model_id", "RunDiffusion/Juggernaut-XL-Lightning")
    
    from diffusers import EulerDiscreteScheduler
    
    try:
        # Load from single checkpoint file
        if checkpoint_path and not is_directory:
            logger.info(f"Loading from single checkpoint file: {checkpoint_path}")
            pipeline = StableDiffusionXLPipeline.from_single_file(
                checkpoint_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                feature_extractor=None
            )
        
        # Load from Diffusers directory
        elif checkpoint_path and is_directory:
            logger.info(f"Loading from Diffusers directory: {checkpoint_path}")
            
            quantization_config = None
            if use_4bit:
                quantization_config = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.float16,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4"
                    }
                )
            
            pipeline = AutoPipelineForText2Image.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                use_safetensors=True,
                local_files_only=True
            )
        
        # Fallback: Download from HuggingFace
        else:
            logger.info(f"Loading from HuggingFace: {model_id}")
            
            quantization_config = None
            if use_4bit:
                quantization_config = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.float16,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4"
                    }
                )
            
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                use_safetensors=True
            )
        
        # Set default scheduler
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing"
        )
        
        # Try to reload VAE with explicit float16 (sometimes helps with artifacts)
        # But if it fails (e.g. missing VAE folder), just ignore it and use the pipeline's VAE
        try:
           if is_directory: # Only relevant for Diffusers format
                logger.info("Attempting to reload VAE in float16...")
                vae = AutoencoderKL.from_pretrained(
                    checkpoint_path,
                    subfolder="vae",
                    torch_dtype=torch.float16,
                    local_files_only=True
                )
                pipeline.vae = vae
                logger.info("VAE reloaded successfully.")
        except Exception as e:
            logger.warning(f"Could not reload VAE (using pipeline default): {e}")
        
        # Enable CPU offload
        pipeline.enable_model_cpu_offload()
        
        logger.info("SDXL Pipeline loaded successfully")
        return pipeline, True
        
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def is_model_downloaded(model_id):
    """
    Check if a model is downloaded locally.
    Returns: bool
    """
    models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))
    checkpoint_path, _ = find_local_checkpoint(model_id, models_dir)
    return checkpoint_path is not None


def get_model_file_details(model_id):
    """
    Get the file path and size (in bytes) of a model.
    Returns: (path, size_bytes) or (None, 0) if not found.
    """
    models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))
    checkpoint_path, is_directory = find_local_checkpoint(model_id, models_dir)
    
    if not checkpoint_path:
        return None, 0
    
    if is_directory:
        # For Diffusers directories, look for main model file
        model_file = os.path.join(checkpoint_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(model_file):
            return model_file, os.path.getsize(model_file)
        # Fallback: return directory path with total size
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(checkpoint_path)
            for filename in filenames
        )
        return checkpoint_path, total_size
    else:
        # Single file
        return checkpoint_path, os.path.getsize(checkpoint_path)
