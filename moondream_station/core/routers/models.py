from fastapi import APIRouter, Request, Response, HTTPException
import os
import glob
from pathlib import Path
from ..hardware_monitor import hw_monitor, model_memory_tracker
from ..sdxl_wrapper import sdxl_backend_new

router = APIRouter()

def discover_models_from_directories(config):
    """
    Auto-discover models from all model directories.
    Scans: analysis/, vision/, models/, diffusers/, checkpoints/
    Returns a list of discovered models with metadata.
    """
    discovered = []
    # Use config models dir for consistency with generation logic
    models_dir = config.get("models_dir")
    if not models_dir:
            models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))
    
    print(f"DEBUG: Discovering models in {models_dir}")

    
    # Helper function to categorize models by directory and name
    def categorize_model(path, name):
        name_lower = name.lower()
        path_lower = path.lower()
        
        # Check directory first
        if "/analysis/" in path_lower or "wd14" in name_lower or "tagger" in name_lower or "swinv2" in name_lower:
            return "analysis"
        elif "/vision/" in path_lower or "moondream" in name_lower or "florence" in name_lower or "joycaption" in name_lower or "joy-caption" in name_lower:
            return "vision"
        else:
            # Default to generation for diffusers/checkpoints
            return "generation"

    # 1. Scan analysis/ directory for WD14 and other taggers
    analysis_dir = os.path.join(models_dir, "analysis")
    if os.path.exists(analysis_dir):
        for item in os.listdir(analysis_dir):
            item_path = os.path.join(analysis_dir, item)
            if os.path.isdir(item_path):
                has_config = any(os.path.exists(os.path.join(item_path, f)) 
                                for f in ["config.json", "model_index.json", "preprocessor_config.json"])
                if has_config:
                    total_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(item_path)
                        for filename in filenames
                    )
                    
                    discovered.append({
                        "id": f"analysis/{item}",
                        "name": item.replace("-", " ").replace("_", " ").title(),
                        "description": "Image tagging and analysis model",
                        "version": "Custom",
                        "type": "tagging",
                        "format": "transformers",
                        "source": "custom",
                        "is_downloaded": True,
                        "size_bytes": total_size,
                        "file_path": item_path,
                        "has_warning": False,
                        "last_known_vram_mb": 2000
                    })
    
    # 2. Scan vision/ directory for Moondream and other vision models
    vision_dir = os.path.join(models_dir, "vision")
    if os.path.exists(vision_dir):
        for item in os.listdir(vision_dir):
            item_path = os.path.join(vision_dir, item)
            if os.path.isdir(item_path):
                has_config = any(os.path.exists(os.path.join(item_path, f)) 
                                for f in ["config.json", "model_index.json", "preprocessor_config.json"])
                if has_config:
                    total_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(item_path)
                        for filename in filenames
                    )
                    
                    m_type = "captioning" if "caption" in item.lower() else "vision"

                    discovered.append({
                        "id": f"vision/{item}",
                        "name": item.replace("-", " ").replace("_", " ").title(),
                        "description": "Vision-language model for image understanding",
                        "version": "Custom",
                        "type": m_type,
                        "format": "transformers",
                        "source": "custom",
                        "is_downloaded": True,
                        "size_bytes": total_size,
                        "file_path": item_path,
                        "has_warning": False,
                        "last_known_vram_mb": 4000
                    })
    
    # 3. Scan models/models--* for HuggingFace cached models
    hf_models_dir = os.path.join(models_dir, "models")
    if os.path.exists(hf_models_dir):
        for item in os.listdir(hf_models_dir):
            if item.startswith("models--") and not item.startswith("."):
                item_path = os.path.join(hf_models_dir, item)
                if os.path.isdir(item_path):
                    model_name = item.replace("models--", "").replace("--", "/")
                    display_name = model_name.split("/")[-1].replace("-", " ").title()
                    
                    snapshots_dir = os.path.join(item_path, "snapshots")
                    if os.path.exists(snapshots_dir):
                        snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                        if snapshots:
                            real_path = os.path.join(snapshots_dir, snapshots[0])
                            total_size = sum(
                                os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(real_path)
                                for filename in filenames
                            )
                            
                            m_type = categorize_model(item_path, model_name)
                            # Alignment fix
                            if m_type == "analysis": m_type = "tagging"
                            # We can refine vision vs captioning here too if needed, but categorize_model is basic.
                            
                            discovered.append({
                                "id": model_name,
                                "name": display_name,
                                "description": f"HuggingFace model: {model_name}",
                                "version": "HF Cache",
                                "type": m_type,
                                "format": "transformers",
                                "source": "huggingface",
                                "is_downloaded": True,
                                "size_bytes": total_size,
                                "file_path": real_path,
                                "has_warning": False,
                                "last_known_vram_mb": 4000
                            })

    # 4. Scan diffusers/ directory for Diffusers models (recursive)
    diffusers_dir = os.path.join(models_dir, "diffusers")
    if os.path.exists(diffusers_dir):
        # Limit recursion depth to avoid scanning too deep (e.g. only 3 levels)
        max_depth = 3
        initial_depth = diffusers_dir.rstrip(os.path.sep).count(os.path.sep)
        
        for root, dirs, files in os.walk(diffusers_dir):
            current_depth = root.count(os.path.sep)
            if current_depth - initial_depth > max_depth:
                continue
                
            if "model_index.json" in files:
                model_path = root
                # If inside snapshots/HASH, we want a cleaner ID
                # e.g. diffusers/albedobase-xl/snapshots/HASH -> diffusers/albedobase-xl
                
                # Try to deduce a clean ID from the path relative to diffusers_dir
                rel_path = os.path.relpath(model_path, diffusers_dir)
                parts = rel_path.split(os.sep)
                
                # Simplify ID: if it contains 'snapshots', take the part before it
                if "snapshots" in parts:
                    idx = parts.index("snapshots")
                    if idx > 0:
                        clean_id = "/".join(parts[:idx])
                    else:
                        clean_id = rel_path
                else:
                    clean_id = rel_path
                    
                model_id = f"diffusers/{clean_id}"
                
                # Avoid duplicates if multiple snapshots exist (pick one, usually walk visits in order)
                if any(m["id"] == model_id for m in discovered):
                    continue

                model_name = clean_id.split("/")[-1].replace("-", " ").title()
                
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
                
                discovered.append({
                    "id": model_id,
                    "name": model_name,
                    "description": "Diffusers Model (Folder)",
                    "version": "Local",
                    "type": "generation",
                    "format": "diffusers",
                    "source": "local",
                    "is_downloaded": True,
                    "size_bytes": total_size,
                    "file_path": model_path,
                    "has_warning": False,
                    "last_known_vram_mb": 8000
                })

    # 5. Scan checkpoints/ directory for Single File models (recursive)
    checkpoints_dir = os.path.join(models_dir, "checkpoints")
    if os.path.exists(checkpoints_dir):
        for root, _, files in os.walk(checkpoints_dir):
            for file in files:
                if file.lower().endswith(('.safetensors', '.ckpt', '.pt', '.bin')):
                    model_path = os.path.join(root, file)
                    relative_path = os.path.relpath(model_path, checkpoints_dir)
                    model_name = os.path.splitext(file)[0].replace("-", " ").title()
                    
                    file_size = os.path.getsize(model_path)
                    # Heuristic: Ignore small files
                    if file_size < 100 * 1024 * 1024:
                        continue

                    discovered.append({
                        "id": f"checkpoints/{relative_path}",
                        "name": f"{model_name} (Checkpoint)",
                        "description": "Local Checkpoint Single File",
                        "version": "Local",
                        "type": "generation",
                        "format": "single_file",
                        "source": "local",
                        "is_downloaded": True,
                        "size_bytes": file_size,
                        "file_path": model_path,
                        "has_warning": False,
                        "last_known_vram_mb": 6000
                    })
    
    return discovered

@router.get("")
async def list_models(request: Request, response: Response = None):
    try:
        manifest_manager = request.app.state.manifest_manager
        config = request.app.state.config
        
        all_models = []
        
        # 1. Manifest Models (Vision/Analysis)
        manifest_models_dict = manifest_manager.get_models()
        for manifest_key, m_raw in manifest_models_dict.items():
            m_data = {}
            if hasattr(m_raw, "model_dump"): m_data = m_raw.model_dump()
            elif hasattr(m_raw, "dict"): m_data = m_raw.dict()
            elif hasattr(m_raw, "__dict__"): m_data = m_raw.__dict__
            elif isinstance(m_raw, dict): m_data = m_raw
            
            # Use manifest key as ID if no explicit ID found
            mid = m_data.get("id") or m_data.get("model_id") or m_data.get("args", {}).get("model_id") or manifest_key
            
            m = {
                "id": mid,
                "name": m_data.get("name", mid),
                "description": m_data.get("description", ""),
                "version": m_data.get("version", "1.0.0"),
                "is_downloaded": True,
                "last_known_vram_mb": model_memory_tracker.get_last_known_vram(mid)
            }
            
            # Auto-categorize by model ID
            mid_lower = mid.lower()
            if "moondream" in mid_lower or "florence" in mid_lower or "joycaption" in mid_lower:
                m["type"] = "vision"
            elif "wd14" in mid_lower or "tagger" in mid_lower:
                m["type"] = "analysis"
            elif "sdxl" in mid_lower or "juggernaut" in mid_lower or "animagine" in mid_lower or "dreamshaper" in mid_lower or "proteus" in mid_lower or "realvis" in mid_lower or "cyberrealistic" in mid_lower:
                m["type"] = "generation"
            else:
                m["type"] = "analysis"
            
            all_models.append(m)
        
        # 2. Auto-discovered Custom Models (all models from filesystem)
        discovered = discover_models_from_directories(config)
        all_models.extend(discovered)
        
        # ADD VRAM HEADERS for frontend polling
        if response:
            try:
                gpus = hw_monitor.get_gpus()
                if gpus and len(gpus) > 0:
                    vram_used = gpus[0].get("memory_used", 0)
                    vram_total = gpus[0].get("memory_total", 0)
                    response.headers["X-VRAM-Used"] = str(vram_used)
                    response.headers["X-VRAM-Total"] = str(vram_total)
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        return {"models": all_models}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refresh")
async def refresh_models(request: Request, response: Response = None):
    """
    Refresh the model list by re-scanning directories.
    Returns the updated model list.
    """
    try:
        return await list_models(request, response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/switch")
async def switch_model(request: Request):
    body = await request.json()
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=400, detail="model is required")
    
    # dependencies
    config = request.app.state.config
    inference_service = request.app.state.inference_service
    manifest_manager = request.app.state.manifest_manager
        
    success = inference_service.start(model_id)
    if success:
        # Capture previous model BEFORE updating config (Critical for unload logic)
        previous_model = config.get("current_model")
        
        config.set("current_model", model_id)
        
        # Unload previous model from tracker
        try:
            if previous_model and previous_model != model_id:
                model_memory_tracker.track_model_unload(previous_model)
                print(f"Unloaded previous model from tracker: {previous_model}")
        except Exception as e:
            print(f"Warning: Failed to unload previous model: {e}")
        
        # Track model load and get stats
        vram_mb = 0
        ram_mb = 0
        try:
            model_info = manifest_manager.get_models().get(model_id)
            if model_info:
                model_memory_tracker.track_model_load(model_id, model_info.name)
                # Get the stats we just tracked
                if model_id in model_memory_tracker.loaded_models:
                    stats = model_memory_tracker.loaded_models[model_id]
                    vram_mb = stats.get("vram_mb", 0)
                    ram_mb = stats.get("ram_mb", 0)
        except Exception as e:
            print(f"Warning: Failed to track model load: {e}")
        
        return {
            "status": "success", 
            "model": model_id,
            "vram_mb": vram_mb,
            "ram_mb": ram_mb
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to switch model")

@router.get("/recommend")
async def recommend_models():
    """Get recommended models for quick setup"""
    return {
        "recommended": [
            {"id": "moondream-2", "name": "Moondream 2", "type": "vision", "reason": "Fast, efficient vision model"},
            {"id": "sdxl-base", "name": "SDXL Base", "type": "generation", "reason": "High-quality image generation"}
        ]
    }
