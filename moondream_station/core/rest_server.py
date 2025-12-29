import asyncio
import sys
import os
# Import SDXL backend from local backends directory
try:
    # Add backends directory to path
    backends_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backends')
    if backends_dir not in sys.path:
        sys.path.insert(0, backends_dir)
    
    # Import the SDXL backend
    from sdxl_backend.backend import SDXLBackend
    
    # Create a wrapper class that matches the expected interface
    class SDXLBackendWrapper:
        def __init__(self):
            self._instance = None
            self._current_model = None
        
        def init_backend(self, model_id="sdxl-realism", use_4bit=True):
            """Initialize SDXL backend with specified model"""
            try:
                # Map model IDs to HuggingFace paths
                model_mapping = {
                    "sdxl-realism": "RunDiffusion/Juggernaut-XL-Lightning",
                    "sdxl-anime": "cagliostrolab/animagine-xl-3.1",
                    "sdxl-surreal": "Lykon/dreamshaper-xl-lightning"
                }
                hf_model = model_mapping.get(model_id, model_id)
                
                # Create or reinitialize if model changed
                if self._instance is None or self._current_model != hf_model:
                    config = {
                        "model_id": hf_model,
                        "use_4bit": use_4bit,
                        "compile": False
                    }
                    self._instance = SDXLBackend(config)
                    self._current_model = hf_model
                
                # Initialize if not already loaded
                if self._instance.pipeline is None:
                    print(f"[SDXL] Initializing backend with model: {hf_model}")
                    self._instance.initialize()
                
                return True
            except Exception as e:
                print(f"[SDXL] Error initializing backend: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        def generate(self, prompt, width=1024, height=1024, steps=8, image=None, strength=0.75):
            """Generate image using SDXL backend"""
            if self._instance is None or self._instance.pipeline is None:
                raise RuntimeError("SDXL backend not initialized")
            
            try:
                # Call the backend's generate method
                if hasattr(self._instance, 'generate'):
                    return self._instance.generate(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        init_image_b64=image,
                        strength=strength
                    )
                else:
                    # Fallback to direct pipeline call
                    import base64
                    import io
                    from PIL import Image as PILImage
                    
                    result = self._instance.pipeline(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps
                    ).images
                    
                    # Convert to base64
                    output_images = []
                    for img in result:
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        output_images.append(img_b64)
                    
                    return output_images
            except Exception as e:
                print(f"[SDXL] Error generating image: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        def unload_backend(self):
            """Unload SDXL backend to free VRAM"""
            if self._instance is not None:
                self._instance.pipeline = None
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[SDXL] Backend unloaded and VRAM freed")
    
    # Create global instance
    sdxl_backend_new = SDXLBackendWrapper()
    print("[SDXL] Backend loaded successfully from local backends directory")
    
except ImportError as e:
    print(f"Warning: Could not import SDXL backend: {e}")
    print("SDXL generation will not be available.")
    sdxl_backend_new = None

import json
import time
import uvicorn
import psutil
import torch
import os
import sys
import subprocess
import urllib.request
from threading import Thread
try:
    import pynvml
except ImportError:
    pynvml = None

class HardwareMonitor:
    def __init__(self):
        self.nvidia_available = False
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.nvidia_available = True
            except Exception:
                pass

    def get_environment_status(self):
        import os
        
        # Detect execution type
        execution_type = "System"
        if os.path.exists("/.dockerenv"):
            execution_type = "Docker"
        elif os.environ.get("VIRTUAL_ENV"):
            execution_type = "Venv"

        status = {
            "platform": "CPU",
            "accelerator_available": False,
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, 'cuda', 'Unknown'),
            "hip_version": getattr(torch.version, 'hip', None),
            "execution_type": execution_type
        }
        
        if torch.cuda.is_available():
            status["platform"] = "CUDA"
            status["accelerator_available"] = True
        elif hasattr(torch.version, 'hip') and torch.version.hip:
            status["platform"] = "ROCm"
            status["accelerator_available"] = True
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
             status["platform"] = "XPU"
             status["accelerator_available"] = True
        elif self.nvidia_available:
            # Fallback: Driver is working, but Torch might not see it
            status["platform"] = "NVIDIA Driver"
            status["accelerator_available"] = True
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode()
                status["cuda_version"] = f"Driver {driver}"
            except:
                pass
        
        return status

    def get_gpus(self):
        gpus = []
        if self.nvidia_available:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    gpus.append({
                        "id": i,
                        "name": name,
                        "load": utilization.gpu,
                        "memory_used": int(memory.used / 1024 / 1024), # MB
                        "memory_total": int(memory.total / 1024 / 1024), # MB
                        "temperature": temp,
                        "type": "NVIDIA"
                    })
            except Exception as e:
                print(f"Nvidia monitoring error: {e}")
        return gpus

# Global monitor instance
hw_monitor = HardwareMonitor()

class ModelMemoryTracker:
    """Track memory usage per loaded model"""
    def __init__(self):
        self.loaded_models = {}  # model_id -> {name, vram_mb, ram_mb, loaded_at}
        self.last_known_vram = {}  # model_id -> vram_mb (persists after unload)
        self.base_vram = 0
        self.base_ram = 0
        
    def record_baseline(self):
        """Record baseline memory before any models loaded"""
        try:
            # Use nvidia-smi to get ACTUAL VRAM usage (includes X Windows, etc.)
            if pynvml:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    # Baseline = Total - Free (what's currently used by system)
                    self.base_vram = (memory.total - memory.free) / (1024 * 1024)  # MB
                    print(f"Baseline VRAM: {self.base_vram:.0f}MB (system + X Windows)")
                except Exception as e:
                    print(f"Failed to get baseline VRAM via pynvml: {e}")
                    # Fallback to torch
                    if torch.cuda.is_available():
                        self.base_vram = torch.cuda.memory_allocated(0) / (1024 * 1024)
            elif torch.cuda.is_available():
                self.base_vram = torch.cuda.memory_allocated(0) / (1024 * 1024)
                
            self.base_ram = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        except Exception as e:
            print(f"Failed to record baseline: {e}")
    
    def track_model_load(self, model_id: str, model_name: str):
        """Track a model being loaded"""
        import time
        try:
            vram_mb = 0
            ram_mb = 0
            
            # Use nvidia-smi for ACTUAL VRAM usage
            if pynvml:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    # Current usage = Total - Free
                    current_vram = (memory.total - memory.free) / (1024 * 1024)
                    # Model VRAM = Current - Baseline
                    vram_mb = current_vram - self.base_vram
                    print(f"VRAM: Total={memory.total/1024/1024:.0f}MB, Free={memory.free/1024/1024:.0f}MB, Used={current_vram:.0f}MB, Baseline={self.base_vram:.0f}MB, Model={vram_mb:.0f}MB")
                except Exception as e:
                    print(f"Failed to get VRAM via pynvml: {e}")
                    # Fallback to torch
                    if torch.cuda.is_available():
                        vram_mb = (torch.cuda.memory_allocated(0) / (1024 * 1024)) - self.base_vram
            elif torch.cuda.is_available():
                vram_mb = (torch.cuda.memory_allocated(0) / (1024 * 1024)) - self.base_vram
            
            ram_mb = (psutil.Process().memory_info().rss / (1024 * 1024)) - self.base_ram
            
            self.loaded_models[model_id] = {
                "id": model_id,
                "name": model_name,
                "vram_mb": int(vram_mb),
                "ram_mb": int(ram_mb),
                "loaded_at": int(time.time())
            }
            self.last_known_vram[model_id] = int(vram_mb)
            print(f"Tracked model load: {model_id} - VRAM: {vram_mb:.0f}MB, RAM: {ram_mb:.0f}MB")
        except Exception as e:
            print(f"Failed to track model load: {e}")
    
    def track_model_unload(self, model_id: str):
        """Track a model being unloaded"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            print(f"Tracked model unload: {model_id}")
    
    def update_memory_usage(self):
        """Update memory usage for all loaded models"""
        try:
            if not self.loaded_models:
                return
            
            current_vram = 0
            current_ram = 0
            
            # Use nvidia-smi for ACTUAL VRAM
            if pynvml:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_vram = (memory.total - memory.free) / (1024 * 1024)
                except:
                    if torch.cuda.is_available():
                        current_vram = torch.cuda.memory_allocated(0) / (1024 * 1024)
            elif torch.cuda.is_available():
                current_vram = torch.cuda.memory_allocated(0) / (1024 * 1024)
            
            current_ram = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Distribute memory across loaded models proportionally
            num_models = len(self.loaded_models)
            if num_models > 0:
                vram_per_model = (current_vram - self.base_vram) / num_models
                ram_per_model = (current_ram - self.base_ram) / num_models
                
                for model_id in self.loaded_models:
                    self.loaded_models[model_id]["vram_mb"] = int(vram_per_model)
                    self.loaded_models[model_id]["ram_mb"] = int(ram_per_model)
                    self.last_known_vram[model_id] = int(vram_per_model)
        except Exception as e:
            print(f"Failed to update memory usage: {e}")
    
    def get_loaded_models(self):
        """Get list of loaded models with memory info"""
        self.update_memory_usage()
        return list(self.loaded_models.values())
    
    def get_last_known_vram(self, model_id: str) -> int:
        """Get last known VRAM usage for a model (even if unloaded)"""
        return self.last_known_vram.get(model_id, 0)

# Global tracker instance
model_memory_tracker = ModelMemoryTracker()
model_memory_tracker.record_baseline()


from threading import Thread
from typing import Any, Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .inference_service import InferenceService


class RestServer:
    def __init__(self, config, manifest_manager, session_state=None, analytics=None):
        # ... (init code remains same)
        self.config = config
        self.manifest_manager = manifest_manager
        self.session_state = session_state
        self.analytics = analytics
        self.inference_service = InferenceService(config, manifest_manager)
        self.app = FastAPI(title="Moondream Station Inference Server", version="1.0.0")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.server = None
        self.server_thread = None
        self._setup_routes()

    def _sse_event_generator(self, raw_generator):
        # ... (remains same)
        token_count = 0
        start_time = time.time()

        for token in raw_generator:
            token_count += 1
            yield f"data: {json.dumps({'chunk': token})}\n\n"

        # Send final stats
        duration = time.time() - start_time
        if duration > 0 and token_count > 0:
            tokens_per_sec = round(token_count / duration, 1)
            stats = {
                "tokens": token_count,
                "duration": round(duration, 2),
                "tokens_per_sec": tokens_per_sec,
            }
            yield f"data: {json.dumps({'stats': stats})}\n\n"

        yield f"data: {json.dumps({'completed': True})}\n\n"

    def _discover_models_from_directories(self):
        """
        Auto-discover models from checkpoints/ and diffusers/ directories.
        Returns a list of discovered models with metadata.
        """
        import glob
        from pathlib import Path
        
        discovered = []
        models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))
        
        # Scan checkpoints directory for single-file models
        checkpoints_dir = os.path.join(models_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            # Support .safetensors, .ckpt, .bin
            for ext in ["*.safetensors", "*.ckpt", "*.bin"]:
                for filepath in glob.glob(os.path.join(checkpoints_dir, ext)):
                    filename = os.path.basename(filepath)
                    name_without_ext = os.path.splitext(filename)[0]
                    file_format = os.path.splitext(filename)[1].replace(".", "")
                    file_size = os.path.getsize(filepath)
                    
                    # Create a model ID from the filename
                    model_id = f"custom/{name_without_ext}"
                    
                    # Check if this is already in curated list (don't duplicate)
                    # This prevents showing the same model twice
                    is_curated = False
                    try:
                        from moondream_station.config import SDXL_MODELS
                        is_curated = any(
                            info.get("hf_id", "").split("/")[-1].lower().replace("-", "").replace("_", "") 
                            in name_without_ext.lower().replace("-", "").replace("_", "")
                            for info in SDXL_MODELS.values()
                        )
                    except:
                        pass
                    
                    if not is_curated:
                        discovered.append({
                            "id": model_id,
                            "name": name_without_ext,
                            "description": f"Custom {file_format.upper()} model",
                            "version": "Custom",
                            "type": "generation",
                            "format": file_format,
                            "source": "custom",
                            "is_downloaded": True,
                            "size_bytes": file_size,
                            "file_path": filepath,
                            "has_warning": file_format in ["ckpt", "bin"],  # Pickle-based formats
                            "last_known_vram_mb": 6000
                        })
        
        # Scan diffusers directory for folder-based models
        diffusers_dir = os.path.join(models_dir, "diffusers")
        if os.path.exists(diffusers_dir):
            for item in os.listdir(diffusers_dir):
                item_path = os.path.join(diffusers_dir, item)
                if os.path.isdir(item_path):
                    model_index = os.path.join(item_path, "model_index.json")
                    if os.path.exists(model_index):
                        # Calculate directory size
                        total_size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk(item_path)
                            for filename in filenames
                        )
                        
                        model_id = f"custom/{item}"
                        
                        # Check if this is already in curated list
                        is_curated = False
                        try:
                            from moondream_station.config import SDXL_MODELS
                            is_curated = any(
                                info.get("hf_id", "").split("/")[-1].lower().replace("-", "").replace("_", "") 
                                in item.lower().replace("-", "").replace("_", "")
                                for info in SDXL_MODELS.values()
                            )
                        except:
                            pass
                        
                        if not is_curated:
                            discovered.append({
                                "id": model_id,
                                "name": item,
                                "description": "Custom Diffusers model",
                                "version": "Custom",
                                "type": "generation",
                                "format": "diffusers",
                                "source": "custom",
                                "is_downloaded": True,
                                "size_bytes": total_size,
                                "file_path": item_path,
                                "has_warning": False,
                                "last_known_vram_mb": 6000
                            })
        
        return discovered
    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            return {"status": "ok", "server": "moondream-station"}

        @self.app.get("/metrics")
        async def metrics():
            """Return system metrics for monitoring"""
            try:
                cpu = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory().percent
                gpus = hw_monitor.get_gpus()
                env = hw_monitor.get_environment_status()
                
                # Determine primary device
                device = "CPU"
                if gpus:
                    device = gpus[0]["name"]
                
                # Get loaded models with real memory usage
                loaded_models = model_memory_tracker.get_loaded_models()
                
                return {
                    "cpu": cpu,
                    "memory": memory,
                    "device": device,
                    "gpus": gpus,
                    "environment": env,
                    "loaded_models": loaded_models
                }
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                return {"cpu": 0, "memory": 0, "device": "Unknown", "gpus": [], "loaded_models": []}
                print(f"Error collecting metrics: {e}")
                return {"cpu": 0, "memory": 0, "device": "Unknown", "gpus": []}

        @self.app.post("/v1/system/gpu-reset")
        async def reset_gpu(gpu_id: int = 0):
            """
            Reset the specified GPU.
            Requires passwordless sudo for nvidia-smi.
            """
            import subprocess
            
            try:
                # Check if nvidia-smi is available and we have permission
                # -n: non-interactive (fails if password needed)
                # Use the new nuclear reset script
                script_path = "/home/bcoster/.moondream-station/moondream-station/nuclear_gpu_reset.sh"
                cmd = ["sudo", "-n", script_path]
                
                process = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=30
                )
                
                if process.returncode == 0:
                    return {"status": "success", "message": f"GPU {gpu_id} reset successfully."}
                
                # Handle permission denied (sudo failed)
                if "password is required" in process.stderr:
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=403, 
                        detail="Permission denied. Passwordless sudo not configured for nvidia-smi."
                    )
                
                # Handle other errors (e.g. GPU busy)
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=500, 
                    detail=f"Reset failed. Stderr: {process.stderr.strip()}. Stdout: {process.stdout.strip()}"
                )
                
            except subprocess.TimeoutExpired:
                from fastapi import HTTPException
                raise HTTPException(status_code=504, detail="Command timed out.")
            except Exception as e:
                # Re-raise HTTP exceptions
                if hasattr(e, "status_code"):
                    raise e
                from fastapi import HTTPException
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v1/system/unload")
        async def unload_model():
            """Force unload the current model from memory"""
            try:
                # Unload Moondream
                if hasattr(self, "inference_service") and self.inference_service:
                    self.inference_service.unload_model()
                
                # Unload SDXL
                if sdxl_backend_new:
                     sdxl_backend_new.unload_backend()

                return {"status": "success", "message": "All models unloaded and VRAM cleared"}
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"status": "error", "message": str(e)}

        
        @self.app.get("/v1/system/prime-profile")
        async def get_prime_profile():
            """Get the current NVIDIA Prime profile (nvidia/on-demand/intel)"""
            import subprocess
            try:
                # prime-select query does not require sudo
                process = subprocess.run(
                    ["prime-select", "query"], 
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                if process.returncode == 0:
                    return {"profile": process.stdout.strip()}
                return {"profile": "unknown", "error": process.stderr.strip()}
            except Exception as e:
                return {"profile": "unknown", "error": str(e)}

        @self.app.post("/v1/system/prime-profile")
        async def set_prime_profile(profile: str):
            """Set the NVIDIA Prime profile (requires sudo/root via script)"""
            import subprocess
            from fastapi import HTTPException
            
            valid_profiles = ["nvidia", "on-demand", "intel"]
            if profile not in valid_profiles:
                raise HTTPException(status_code=400, detail="Invalid profile")
                
            try:
                # Use the script helper
                script_path = "/home/bcoster/.moondream-station/moondream-station/switch_prime_profile.sh"
                cmd = ["sudo", "-n", script_path, profile]
                
                process = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=30
                )
                
                if process.returncode == 0:
                    return {"status": "success", "message": f"Switched to {profile}. Please logout/reboot."}
                
                if "password is required" in process.stderr:
                    raise HTTPException(
                        status_code=403, 
                        detail="Permission denied. Sudo requires password."
                    )
                    
                raise HTTPException(status_code=500, detail=f"Failed: {process.stderr}")
                
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=504, detail="Command timed out.")
            except Exception as e:
                if hasattr(e, "status_code"): raise e
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/v1/system/zombie-killer")
        async def get_zombie_killer_status():
            """Get auto-free zombie VRAM status"""
            enabled = self.config.get("zombie_killer_enabled", False)
            interval = self.config.get("zombie_killer_interval", 60)
            return {"enabled": enabled, "interval": interval}

        @self.app.post("/v1/system/zombie-killer")
        async def toggle_zombie_killer(request: Request):
            """Toggle auto-free zombie VRAM feature"""
            try:
                data = await request.json()
                if "enabled" in data:
                    self.config.set("zombie_killer_enabled", bool(data["enabled"]))
                if "interval" in data:
                    self.config.set("zombie_killer_interval", max(30, int(data["interval"])))
                
                enabled = self.config.get("zombie_killer_enabled", False)
                interval = self.config.get("zombie_killer_interval", 60)
                
                print(f"[Zombie Killer] Auto-free VRAM: {'ENABLED' if enabled else 'DISABLED'} (interval: {interval}s)")
                return {"enabled": enabled, "interval": interval}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v1/vision/batch-caption")
        async def batch_caption(request: Request):
            """
            Batch caption generic images (e.g. WD14 Tagger). 
            Accepts: { "images": ["b64...", ...], "model": "..." }
            """
            if not self.inference_service.is_running():
                return JSONResponse(content={"error": "Inference service not running"}, status_code=503)

            try:
                data = await request.json()
                images_b64 = data.get("images", [])
                model_id = data.get("model", self.config.get("current_model"))
                
                if not images_b64 or not isinstance(images_b64, list):
                    return JSONResponse(content={"error": "images must be a list of base64 strings"}, status_code=400)

                # Ensure correct model is loaded
                if self.config.get("current_model") != model_id:
                     print(f"[Batch] Switching to {model_id}...")
                     if not self.inference_service.start(model_id):
                         return JSONResponse(content={"error": f"Failed to load model {model_id}"}, status_code=500)
                     self.config.set("current_model", model_id)

                # Decode all images
                pil_images = []
                import base64
                import io
                from PIL import Image
                
                for b64 in images_b64:
                    if b64.startswith("data:image"):
                        _, encoded = b64.split(",", 1)
                    else:
                        encoded = b64
                    # Don't decode yet if the worker is in another process? 
                    # Actually, we pass objects to simple worker pool which runs in a thread executor, so passing PIL is fine.
                    raw_bytes = base64.b64decode(encoded)
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    pil_images.append(img)

                # Execute Batch
                start_time = time.time()
                captions = await self.inference_service.execute_function("caption", image=pil_images)
                
                duration = time.time() - start_time

                return {
                    "captions": captions,
                    "count": len(captions) if isinstance(captions, list) else 1,
                    "duration": round(duration, 3)
                }

                return {
                    "captions": captions,
                    "count": len(captions) if isinstance(captions, list) else 1,
                    "duration": round(duration, 3)
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        @self.app.post("/diagnostics/vram-test")
        async def vram_test(request: Request):
            """
            Test VRAM usage with different batch sizes for WD14 tagging.
            Request: { "batch_sizes": [4, 8, 16, 32, 64] }
            Response: [{ "batchSize": 4, "vramPercent": 45.2, "duration": 1.23 }, ...]
            """
            try:
                data = await request.json()
                batch_sizes = data.get("batch_sizes", [4, 8, 16, 32])
                
                # Ensure WD14 model is loaded
                current_model = self.config.get("current_model")
                if not current_model or "wd14" not in current_model.lower():
                    # Switch to WD14
                    if not self.inference_service.start("wd14-vit-v2"):
                        return JSONResponse(content={"error": "Failed to load WD14 model"}, status_code=500)
                    self.config.set("current_model", "wd14-vit-v2")
                
                results = []
                
                # Create dummy test images (simple colored squares)
                from PIL import Image
                import io
                import base64
                
                for batch_size in batch_sizes:
                    # Generate dummy images
                    dummy_images = []
                    for i in range(batch_size):
                        img = Image.new('RGB', (224, 224), color=(128 + i*2, 64, 192))
                        dummy_images.append(img)
                    
                    # Get VRAM before
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            vram_before = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
                    except:
                        vram_before = 0
                    
                    # Process batch
                    import time
                    start_time = time.time()
                    
                    try:
                        result = await self.inference_service.execute_function(
                            "caption", 
                            image=dummy_images,
                            timeout=30.0
                        )
                        duration = time.time() - start_time
                        
                        # Get VRAM after
                        try:
                            import torch
                            if torch.cuda.is_available():
                                vram_after = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
                                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # Total GB
                                vram_percent = (vram_after / vram_total) * 100
                        except:
                            vram_percent = 0
                        
                        results.append({
                            "batchSize": batch_size,
                            "vramPercent": round(vram_percent, 1),
                            "vramGB": round(vram_after, 2),
                            "duration": round(duration, 3),
                            "success": True
                        })
                        
                    except Exception as e:
                        results.append({
                            "batchSize": batch_size,
                            "vramPercent": 100,
                            "vramGB": 0,
                            "duration": 0,
                            "success": False,
                            "error": str(e)
                        })
                        # If we hit OOM, stop testing larger batches
                        if "out of memory" in str(e).lower():
                            break
                
                return {"results": results}
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.post("/v1/generate")
        async def generate_image(request: Request):
            """
            Generate generic image using SDXL Backend with Smart VRAM Management
            """
            if not sdxl_backend_new:
                 return JSONResponse(content={"error": "SDXL Backend not available"}, status_code=500)

            # Get VRAM Mode from Header (high, balanced, low)
            vram_mode = request.headers.get("X-VRAM-Mode", "balanced") 

            try:
                data = await request.json()
                prompt = data.get("prompt")
                if not prompt:
                     return JSONResponse(content={"error": "Prompt is required"}, status_code=400)

                # Smart Switching: Unload Moondream if needed
                if vram_mode in ["balanced", "low"]:
                    if hasattr(self, "inference_service") and self.inference_service.is_running():
                        print(f"[VRAM] Unloading Moondream for SDXL generation ({vram_mode} mode)...")
                        self.inference_service.unload_model()

                # Init Backend
                success = sdxl_backend_new.init_backend(
                    model_id=data.get("model", "sdxl-realism"),
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
                        self.unload_all_models()
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

                return {
                    "created": int(time.time()), 
                    "data": [{"b64_json": img} for img in generated_images], 
                    "images": generated_images, 
                    "image": generated_images[0] if generated_images else None,
                    "stats": stats
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                return JSONResponse(content={"error": str(e)}, status_code=500)


        @self.app.get("/v1/models")
        async def list_models():
            try:
                all_models = []
                
                # 1. Manifest Models (Vision/Analysis)
                manifest_models_dict = self.manifest_manager.get_models()
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
                
                # 2. Curated SDXL Models (Generation)
                try:
                    from moondream_station.config import SDXL_MODELS
                    for model_id, model_info in SDXL_MODELS.items():
                        # Detect if downloaded and format
                        is_downloaded = False
                        size_bytes = 0
                        detected_format = None
                        
                        try:
                            if sdxl_backend_new:
                                is_downloaded = sdxl_backend_new.is_model_downloaded(model_info.get("hf_id"))
                                if hasattr(sdxl_backend_new, "get_model_file_details"):
                                    file_path, size_bytes = sdxl_backend_new.get_model_file_details(model_info.get("hf_id"))
                                    if file_path:
                                        if os.path.isdir(file_path) and os.path.exists(os.path.join(file_path, "model_index.json")):
                                            detected_format = "diffusers"
                                        elif os.path.isfile(file_path):
                                            ext = os.path.splitext(file_path)[1].lower()
                                            detected_format = ext.replace(".", "")
                        except: pass
                        
                        # Add format badge to display name
                        display_name = model_info["name"]
                        if detected_format:
                            display_name = f"{display_name} [{detected_format.upper()}]"
                        
                        all_models.append({
                            "id": model_id,
                            "name": display_name,
                            "description": model_info["description"],
                            "version": "SDXL",
                            "last_known_vram_mb": model_memory_tracker.get_last_known_vram(model_id) or 6000,
                            "type": "generation",
                            "source": "curated",
                            "is_downloaded": is_downloaded,
                            "size_bytes": size_bytes,
                            "format": detected_format,
                            "has_warning": False
                        })
                except: pass
                
                # 3. Auto-discovered Custom Models
                discovered = self._discover_models_from_directories()
                all_models.extend(discovered)
                
                return {"models": all_models}
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v1/models/refresh")
        async def refresh_models():
            """
            Refresh the model list by re-scanning directories.
            Returns the updated model list.
            """
            try:
                return await list_models()
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/v1/stats")
        async def get_stats():
            stats = self.inference_service.get_stats()
            # Add requests processed from session state
            if self.session_state:
                stats["requests_processed"] = self.session_state.state["requests_processed"]
            else:
                stats["requests_processed"] = 0
            return stats

        @self.app.post("/v1/models/switch")
        async def switch_model(request: Request):
            body = await request.json()
            model_id = body.get("model")
            if not model_id:
                raise HTTPException(status_code=400, detail="model is required")
            
            if model_id not in self.manifest_manager.get_models():
                raise HTTPException(status_code=404, detail="Model not found")
                
            success = self.inference_service.start(model_id)
            if success:
                # Capture previous model BEFORE updating config (Critical for unload logic)
                previous_model = self.config.get("current_model")
                
                self.config.set("current_model", model_id)
                
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
                    model_info = self.manifest_manager.get_models().get(model_id)
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

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await self._handle_chat_completion(request)

        @self.app.api_route(
            "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
        )
        async def dynamic_route(request: Request, path: str):
            return await self._handle_dynamic_request(request, path)



    async def _handle_chat_completion(self, request: Request):
        # NOTE: Removed initial is_running check to allow auto-start

        try:
            body = await request.json()
            messages = body.get("messages", [])
            stream = body.get("stream", False)
            requested_model = body.get("model")
            
            # --- AUTO-START / SWITCH LOGIC ---
            if not self.inference_service.is_running():
                target_model = requested_model if requested_model else "moondream-2"
                print(f"DEBUG: Service stopped. Auto-starting {target_model}...")
                
                # Check previous for unloading (edge case where service stopped but tracker has state)
                previous_model = self.config.get("current_model")
                
                if self.inference_service.start(target_model):
                     self.config.set("current_model", target_model)
                     
                     # TRACKER UPDATE
                     try:
                         if previous_model and previous_model != target_model:
                             model_memory_tracker.track_model_unload(previous_model)
                         
                         model_info = self.manifest_manager.get_models().get(target_model)
                         if model_info:
                             model_memory_tracker.track_model_load(target_model, model_info.name)
                     except Exception as e:
                         print(f"Warning: Failed to track auto-start: {e}")
                else:
                     raise HTTPException(status_code=500, detail="Failed to start model")

            current_model = self.config.get("current_model")

            # Auto-switch model if requested and different
            if requested_model and requested_model != current_model:
                if requested_model in self.manifest_manager.get_models():
                    print(f"Auto-switching to requested model: {requested_model}")
                    
                    # Capture previous for unload
                    previous_before_switch = current_model
                    
                    if self.inference_service.start(requested_model):
                        self.config.set("current_model", requested_model)
                        current_model = requested_model
                        
                        # TRACKER UPDATE
                        try:
                            if previous_before_switch:
                                model_memory_tracker.track_model_unload(previous_before_switch)
                                print(f"[Tracker] Auto-switch unloaded: {previous_before_switch}")
                                
                            model_info = self.manifest_manager.get_models().get(requested_model)
                            if model_info:
                                model_memory_tracker.track_model_load(requested_model, model_info.name)
                                print(f"[Tracker] Auto-switch loaded: {requested_model}")
                        except Exception as e:
                            print(f"Warning: Failed to track auto-switch: {e}")
                    else:
                        raise HTTPException(status_code=500, detail=f"Failed to switch to model {requested_model}")
                else:
                    # If model not found, we might just continue with current if compatible, 
                    # but typically we error or fallback.
                    pass

            model = current_model

            if not messages:
                raise HTTPException(status_code=400, detail="Messages required")

            # Extract last user message
            last_user_msg = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_msg = msg
                    break

            if not last_user_msg:
                raise HTTPException(status_code=400, detail="No user message found")

            content = last_user_msg.get("content", "")
            
            # Parse content for text and image
            prompt_text = ""
            image_url = None

            if isinstance(content, str):
                prompt_text = content
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        prompt_text += item.get("text", "")
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url")

            # Determine function to call
            function_name = "caption"
            kwargs = {"stream": stream}

            if image_url:
                kwargs["image_url"] = image_url
                if prompt_text and prompt_text.lower().strip() not in ["describe this", "caption this", ""]:
                    function_name = "query"
                    kwargs["question"] = prompt_text
                else:
                    # Generic prompt or no prompt -> caption
                    function_name = "caption"
            else:
                # Text only
                raise HTTPException(status_code=400, detail="Image required for Moondream")

            # --- SMART VRAM MANAGEMENT START ---
            vram_mode = request.headers.get("X-VRAM-Mode", "balanced")
            if vram_mode in ["balanced", "low"]:
                # Ensure SDXL is unloaded to free space for Moondream
                if 'sdxl_backend_new' in sys.modules:
                    try:
                        # Only unload if it looks like it might be loaded (backend instance exists)
                        # But our unload_backend check is safe
                        print(f"DEBUG: Smart Switching (Mode: {vram_mode}) - Unloading SDXL before analysis...")
                        sys.modules['sdxl_backend_new'].unload_backend()
                    except Exception as e:
                        print(f"WARNING: Failed to unload SDXL: {e}")
            # --- SMART VRAM MANAGEMENT END ---

            # Execute with OOM Retry
            start_time = time.time()
            try:
                result = await self.inference_service.execute_function(
                    function_name, None, **kwargs
                )
            except Exception as e:
                # Check for OOM
                error_str = str(e).lower()
                if "out of memory" in error_str or "cuda" in error_str or "alloc" in error_str:
                    print(f"CRITICAL: OOM detected during Moondream analysis. Logic: Auto-Recovery. Error: {e}")
                    
                    # 1. Unload everything
                    self.unload_all_models()
                    
                    # 2. Restart Moondream Service
                    print(f"Re-initializing Moondream service for model: {model}")
                    if self.inference_service.start(model):
                        print("Service restarted. Retrying operation...")
                        # 3. Retry execution
                        result = await self.inference_service.execute_function(
                            function_name, None, **kwargs
                        )
                    else:
                        raise HTTPException(status_code=500, detail="OOM Recovery failed: Could not restart service.")
                else:
                    raise e


            # Handle Streaming
            if stream:
                return StreamingResponse(
                    self._sse_chat_generator(result, model), 
                    media_type="text/event-stream"
                )

            # Handle Standard Response
            response_text = ""
            if isinstance(result, dict):
                if "caption" in result:
                    response_text = result["caption"]
                elif "answer" in result:
                    response_text = result["answer"]
                else:
                    # Fallback to JSON string
                    response_text = json.dumps(result)
            else:
                response_text = str(result)

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(response_text.split())
                }
            }

        except Exception as e:
            if self.analytics:
                self.analytics.track_error(type(e).__name__, str(e), "api_chat_completions")
            print(f"Error in chat_completions: {e}")
            raise HTTPException(status_code=500, detail=str(e))


    def _sse_chat_generator(self, raw_generator, model):
        """Convert generator to OpenAI-compatible SSE format"""
        yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
        
        for token in raw_generator:
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    async def _handle_dynamic_request(self, request: Request, path: str):
        # --- PATCH: System Restart Endpoint ---
        if "system/restart" in path:
            print("[System] Received system/restart request.")
            
            def _restart_process():
                print("[System] Restarting server process in 1s...")
                time.sleep(1)
                python = sys.executable
                os.execl(python, python, *sys.argv)
            
            Thread(target=_restart_process, daemon=True).start()
            return JSONResponse({"status": "restarting"})
        # --------------------------------------

        if not self.inference_service.is_running():
            raise HTTPException(status_code=503, detail="Inference service not running")

        function_name = self._extract_function_name(path)
        kwargs = await self._extract_request_data(request)

        # Auto-switch model if requested and different
        requested_model = kwargs.get("model")
        current_model = self.config.get("current_model")

        if requested_model and requested_model != current_model:
            if requested_model in self.manifest_manager.get_models():
                print(f"Auto-switching to requested model: {requested_model}")
                # Capture previous model for tracking unloading BEFORE switch
                previous_model_auto = self.config.get("current_model")
                
                if self.inference_service.start(requested_model):
                    self.config.set("current_model", requested_model)
                    current_model = requested_model
                    
                    # SYSTEM INTEGRATION: Track model switch in memory tracker
                    try:
                        # 1. Unload previous
                        if previous_model_auto and previous_model_auto != requested_model:
                            model_memory_tracker.track_model_unload(previous_model_auto)
                            print(f"[Tracker] Unloaded previous model on auto-switch: {previous_model_auto}")
                            
                        # 2. Load new
                        model_info = self.manifest_manager.get_models().get(requested_model)
                        if model_info:
                            model_memory_tracker.track_model_load(requested_model, model_info.name)
                            print(f"[Tracker] Tracked new model on auto-switch: {requested_model}")
                    except Exception as e:
                        print(f"[Tracker] Warning: Failed to track auto-switch: {e}")
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to switch to model {requested_model}")
            else:
                raise HTTPException(status_code=404, detail=f"Model {requested_model} not found")

        timeout = kwargs.pop("timeout", None)
        if timeout:
            try:
                timeout = float(timeout)
            except (ValueError, TypeError):
                timeout = None

        # Check if streaming is requested
        stream = kwargs.get("stream", False)

        start_time = time.time()
        try:
            result = await self.inference_service.execute_function(
                function_name, timeout, **kwargs
            )

            # --- PATCH: OOM Check for non-raised errors ---
            if isinstance(result, dict) and result.get("error"):
                err_msg = str(result["error"])
                if "CUDA out of memory" in err_msg or "OutOfMemoryError" in err_msg:
                    # Force raise to trigger the OOM handler in the except block
                    raise Exception(f"CUDA out of memory: {err_msg}")
            # ---------------------------------------------

            # Record the request in session state
            if self.session_state:
                self.session_state.record_request(f"/{path}")

            success = not (isinstance(result, dict) and result.get("error"))

            # --- PATCH: Auto-Unload Logic ---
            # Check for header or env var to unload model after inference
            should_unload = request.headers.get("X-Auto-Unload") == "true" or \
                            os.environ.get("MOONDREAM_AUTO_UNLOAD") == "true"
            
            if should_unload:
                print(f"[System] Auto-unloading model (Reason: Auto-Unload requested)")
                # Run unload in a separate thread to avoid blocking the return
                Thread(target=self.inference_service.unload_model, daemon=True).start()
            # -------------------------------
        except Exception as e:
            # --- PATCH: OOM Logging & Auto-Restart ---
            error_str = str(e)
            if "CUDA out of memory" in error_str or "OutOfMemoryError" in type(e).__name__:
                print(f"\n{'='*40}")
                print(f"CRITICAL ERROR: GPU OOM Detected!")
                print(f"Error: {error_str}")
                print(f"{'='*40}\n")
                
                gpu_stats = "N/A"
                mem_stats = "N/A"

                # Run nvidia-smi
                print("[Diagnostics] Running nvidia-smi...")
                try:
                    proc_res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
                    gpu_stats = proc_res.stdout
                    print(gpu_stats)
                except Exception as log_err:
                    print(f"Failed to run nvidia-smi: {log_err}")
                    gpu_stats = f"Failed: {log_err}"

                # Capture Python process memory
                try:
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    mem_stats = f"RSS={mem_info.rss / 1024 / 1024:.2f} MB, VMS={mem_info.vms / 1024 / 1024:.2f} MB"
                    print(f"[Diagnostics] Python Process Memory: {mem_stats}")
                except Exception as log_err:
                    print(f"Failed to get process memory: {log_err}")
                    mem_stats = f"Failed: {log_err}"

                # SEND TO APP LOG SERVER
                try:
                    # Extract Task Details
                    task_info = f"Function: {function_name}\nModel: {kwargs.get('model', 'unknown')}"
                    if 'prompt' in kwargs:
                        p = str(kwargs['prompt'])
                        task_info += f"\nPrompt: {p[:100]}..." if len(p) > 100 else f"\nPrompt: {p}"
                    if 'width' in kwargs and 'height' in kwargs:
                        task_info += f"\nResolution: {kwargs.get('width')}x{kwargs.get('height')}"
                    
                    log_payload = {
                        "level": "CRITICAL",
                        "context": "MoondreamBackend",
                        "message": f"OOM Crash in {function_name}! Auto-Restart Initiated.",
                        "stack": f"Error: {error_str}\n\nTask Details:\n{task_info}\n\nGPU Stats:\n{gpu_stats}\n\nProcess Mem: {mem_stats}" 
                    }
                    
                    req = urllib.request.Request(
                        "http://localhost:3001/log",
                        data=json.dumps(log_payload).encode('utf-8'),
                        headers={'Content-Type': 'application/json'}
                    )
                    urllib.request.urlopen(req, timeout=2)
                    print("[Diagnostics] Sent OOM report to App Log Server.")
                except Exception as log_send_err:
                    print(f"[Diagnostics] Failed to send log to app server: {log_send_err}")

                print("\n[System] Initiating EMERGENCY RESTART to recover from OOM...")
                print(f"{'='*40}\n")
                
                time.sleep(2)
                
                # Re-execute the current process
                python = sys.executable
                os.execl(python, python, *sys.argv)
            # -----------------------------------------

            if self.analytics:
                self.analytics.track_error(
                    type(e).__name__,
                    str(e),
                    f"api_{function_name}"
                )
            raise

        # Handle streaming response
        if stream and isinstance(result, dict) and not result.get("error"):
            # Look for any generator in result (any capability can stream)
            generator_key = None
            generator = None

            for key, value in result.items():
                if hasattr(value, "__iter__") and hasattr(value, "__next__"):
                    generator_key = key
                    generator = value
                    break

            if generator:
                event_generator = self._sse_event_generator(generator)
                return StreamingResponse(
                    event_generator, media_type="text/event-stream"
                )

        # Add token stats and analytics for non-streaming responses
        if isinstance(result, dict) and not result.get("error"):
            token_count = 0
            # Count tokens from any string result
            for key, value in result.items():
                if isinstance(value, str):
                    token_count += len(value.split())

            duration = time.time() - start_time
            if duration > 0 and token_count > 0:
                result["_stats"] = {
                    "tokens": token_count,
                    "duration": round(duration, 2),
                    "tokens_per_sec": round(token_count / duration, 1),
                }

            if self.analytics:
                self.analytics.track_api_call(
                    function_name,
                    duration,
                    tokens=token_count,
                    success=success,
                    model=self.config.get("current_model")
                )

        return JSONResponse(result)

    def _extract_function_name(self, path: str) -> str:
        path_parts = [p for p in path.split("/") if p]
        name = "index"
        
        if path_parts and path_parts[0] == "v1" and len(path_parts) > 1:
            name = path_parts[1]
        elif path_parts:
            name = path_parts[-1]
            
        # Alias mapping
        if name == "answer":
            return "query"
            
        return name

    async def _extract_request_data(self, request: Request) -> Dict[str, Any]:
        kwargs = {}

        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            try:
                body = await request.json()
                kwargs.update(body)
            except json.JSONDecodeError:
                pass
        elif "application/x-www-form-urlencoded" in content_type:
            form = await request.form()
            kwargs.update(dict(form))
        elif "multipart/form-data" in content_type:
            form = await request.form()
            for key, value in form.items():
                kwargs[key] = value

        kwargs.update(dict(request.query_params))

        kwargs["_headers"] = dict(request.headers)
        kwargs["_method"] = request.method

        return kwargs

    def unload_all_models(self):
        """Unloads both Moondream and SDXL to free maximum VRAM"""
        print("[System] Emergency Unload Triggered")
        try:
            if hasattr(self, "inference_service") and self.inference_service:
                self.inference_service.unload_model()
        except: pass
        
        try:
            if sdxl_backend_new:
                sdxl_backend_new.unload_backend()
        except: pass
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()


    def start(self, host: str = "127.0.0.1", port: int = 2020) -> bool:
        if self.server_thread and self.server_thread.is_alive():
            return False

        current_model = self.config.get("current_model")
        if not current_model:
            return False

        if not self.inference_service.start(current_model):
            return False

        try:
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info",  # Suppress more logs
                access_log=False,
            )
            self.server = uvicorn.Server(config)

            self.server_thread = Thread(target=self._run_server, daemon=True)
            self.server_thread.start()

            time.sleep(1)

            return self.is_running()
        except Exception:
            return False

    def _run_server(self):
        try:
            asyncio.run(self.server.serve())
        except (Exception, asyncio.CancelledError):
            # Suppress normal shutdown errors
            pass

    def stop(self) -> bool:
        """Stop the REST server properly"""
        if self.server:
            # Signal server to stop
            self.server.should_exit = True

            # Force shutdown the server
            # if hasattr(self.server, "force_exit"):
            #     self.server.force_exit = True

        # Wait for server thread to finish
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=3)

            # If thread is still alive, something went wrong
            if self.server_thread.is_alive():
                import logging

                logging.warning("Server thread did not shut down cleanly")

        # Stop inference service
        if hasattr(self, "inference_service") and self.inference_service:
            try:
                # Run the async stop in a sync context
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create task if loop is running
                        asyncio.create_task(self.inference_service.stop())
                    else:
                        # Run directly if loop is not running
                        loop.run_until_complete(self.inference_service.stop())
                except RuntimeError:
                    # No event loop, run in new loop
                    asyncio.run(self.inference_service.stop())
            except Exception:
                pass

        # Clean up references
        self.server = None
        self.server_thread = None

        return True

    def is_running(self) -> bool:
        return (
            self.server_thread
            and self.server_thread.is_alive()
            and self.server
            and not self.server.should_exit
        )
