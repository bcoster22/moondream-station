from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import JSONResponse
import os
import time
import json
from threading import Thread

from ..hardware_monitor import hw_monitor, model_memory_tracker
from ..sdxl_wrapper import sdxl_backend_new
from .models import discover_models_from_directories

router = APIRouter()

@router.get("/samplers")
async def list_samplers():
    """List available samplers"""
    return {"samplers": ["euler", "euler_a", "dpm", "dpm++", "ddim", "pndm"]}

@router.get("/schedulers")
async def list_schedulers():
    """List available schedulers"""
    return {"schedulers": ["euler", "euler_ancestral", "dpm_solver", "dpm_solver++", "ddim", "pndm", "lms"]}

@router.post("/images/generations")
@router.post("/generate")
async def generate_image(request: Request):
    """
    Generate generic image using SDXL Backend with Smart VRAM Management
    """
    if not sdxl_backend_new:
            return JSONResponse(content={"error": "SDXL Backend not available"}, status_code=500)
    
    # Dependencies
    inference_service = request.app.state.inference_service
    config = request.app.state.config
    
    # Helper to unload all models (Replicated from rest_server)
    def unload_all_models_helper():
        print("Unloading all models...")
        inference_service.stop()
        sdxl_backend_new.unload_backend()
        model_memory_tracker.update_memory_usage()

    # Get VRAM Mode from Header (high, balanced, low)
    vram_mode = request.headers.get("X-VRAM-Mode", "balanced") 

    try:
        data = await request.json()
        prompt = data.get("prompt")
        if not prompt:
                return JSONResponse(content={"error": "Prompt is required"}, status_code=400)

        # Smart Switching: Unload Moondream if needed
        if vram_mode in ["balanced", "low"]:
            if inference_service.is_running():
                print(f"[VRAM] Unloading Moondream for SDXL generation ({vram_mode} mode)...")
                inference_service.unload_model()

        # Resolve Model Path to Local File
        target_model_id = data.get("model", "sdxl-realism")
        resolved_model_path = target_model_id # Default to ID if all else fails
        
        try:
            found = False
            # Use discover_models_from_directories instead of self.scan_models
            available_models = discover_models_from_directories(config)
            
            # Strategy 1: Exact Match
            for m in available_models:
                if m.get("id") == target_model_id and m.get("file_path"):
                    resolved_model_path = m["file_path"]
                    found = True
                    print(f"[SDXL] Resolved {target_model_id} (Exact) to: {resolved_model_path}")
                    break
            
            # Strategy 2: Fuzzy Match (if ID is part of the name or file path)
            if not found:
                for m in available_models:
                    if (target_model_id in m.get("id", "") or target_model_id in m.get("name", "")) and m.get("file_path"):
                        resolved_model_path = m["file_path"]
                        found = True
                        print(f"[SDXL] Resolved {target_model_id} (Fuzzy) to: {resolved_model_path}")
                        break

            # Strategy 3: Manual HuggingFace Cache Construction (Offline Fail-safe)
            if not found and "/" in target_model_id:
                # Construct expected HF cache path: models--User--Repo
                # e.g. hf/Lykon/dreamshaper-xl-lightning -> models--Lykon--dreamshaper-xl-lightning
                parts = target_model_id.replace("hf/", "").split("/")
                if len(parts) >= 2:
                    user_name = parts[0]
                    repo_name = parts[1]
                    hf_folder = f"models--{user_name}--{repo_name}"
                    
                    # Determine models directory from config
                    models_dir = config.get("models_dir")
                    # The 'models' subdirectory inside the main models dir stores HF cache
                    hf_models_dir = os.path.join(models_dir, "models")
                    
                    expected_path = os.path.join(hf_models_dir, hf_folder, "snapshots")
                    
                    if os.path.exists(expected_path):
                        # Find latest snapshot (hash folder)
                        snapshots = [os.path.join(expected_path, d) for d in os.listdir(expected_path) if os.path.isdir(os.path.join(expected_path, d))]
                        if snapshots:
                            # Sort by modification time to get latest? Or just pick first.
                            latest_snapshot = max(snapshots, key=os.path.getmtime)
                            resolved_model_path = latest_snapshot
                            found = True
                            print(f"[SDXL] Resolved {target_model_id} (Manual HF) to: {resolved_model_path}")
                        else:
                            print(f"[SDXL] Found {hf_folder} but no snapshots.")
                    else:
                        print(f"[SDXL] Manual path verify failed: {expected_path} does not exist")

            # Strategy 4: Manual Checkpoint Resolution (Offline Fail-safe)
            if not found and target_model_id.startswith("checkpoint/"):
                # Extract filename part
                filename_part = target_model_id.replace("checkpoint/", "")
                # Use config models dir
                models_dir = os.path.join(config.get("models_dir"), "checkpoints")
                
                # Try common extensions
                for ext in [".safetensors", ".ckpt", ".bin", ".pt"]:
                    possible_path = os.path.join(models_dir, f"{filename_part}{ext}")
                    if os.path.exists(possible_path):
                        resolved_model_path = possible_path
                        found = True
                        print(f"[SDXL] Resolved {target_model_id} (Manual Checkpoint) to: {resolved_model_path}")
                        break
                
                if not found:
                        print(f"[SDXL] Manual checkpoint verify failed for {filename_part} in {models_dir}")

            if not found:
                print(f"[SDXL] Warning: Could not resolve {target_model_id}. Attempts failed.")
                # Debug: Print available
                ids = [m.get("id") for m in available_models] if 'available_models' in locals() else []
                print(f"[SDXL] Available IDs: {ids}")

        except Exception as e:
            print(f"[SDXL] Error during resolution: {e}")


        # Init Backend with Resolved Path
        success = sdxl_backend_new.init_backend(
            model_id=resolved_model_path,
            use_4bit=True
        )
        if not success:
                return JSONResponse(content={"error": "Failed to initialize SDXL backend"}, status_code=500)

        # Generation Logic with Retry
        width = data.get("width", 1024)
        height = data.get("height", 1024)
        steps = data.get("steps", 8)
        image = data.get("image") 
        strength = data.get("strength", 0.75)
        scheduler = data.get("scheduler", "euler")

        try:
            result = sdxl_backend_new.generate(
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                image=image,
                strength=strength,
                scheduler=scheduler
            )
        except Exception as gen_err:
            if "out of memory" in str(gen_err).lower():
                print("[VRAM] OOM detected during generation. Triggering Emergency Unload and Retry...")
                unload_all_models_helper()
                # Retry once
                success = sdxl_backend_new.init_backend(
                    model_id=data.get("model", "sdxl-realism"),
                    use_4bit=True
                )
                result = sdxl_backend_new.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    image=image,
                    strength=strength,
                    scheduler=scheduler
                )
            else:
                raise gen_err

        # Extract images and stats
        generated_images = []
        stats = {}
        
        if isinstance(result, dict) and "images" in result:
            generated_images = result["images"]
            stats = result.get("stats", {})
        else:
            generated_images = result
        
        # Low VRAM Cleanup
        if vram_mode == "low":
            print("[VRAM] Low mode: Unloading SDXL after generation.")
            sdxl_backend_new.unload_backend()

        # Prepare Headers
        headers = {}
        try:
            gpus = hw_monitor.get_gpus()
            if gpus:
                headers["X-VRAM-Used"] = str(gpus[0]["memory_used"]) 
                headers["X-VRAM-Total"] = str(gpus[0]["memory_total"])
        except: pass

        content = {
            "created": int(time.time()), 
            "data": [{"b64_json": img} for img in generated_images], 
            "images": generated_images, 
            "image": generated_images[0] if generated_images else None,
            "stats": stats
        }
        
        return JSONResponse(content=content, headers=headers)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
