import os
import sys

# Import SDXL backend from local backends directory
# Add backends directory to path
backends_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backends')
if backends_dir not in sys.path:
    sys.path.insert(0, backends_dir)

try:
    from sdxl_backend.backend import SDXLBackend
except ImportError:
    SDXLBackend = None

class SDXLBackendWrapper:
    def __init__(self):
        self._instance = None
        self._current_model = None
    
    def init_backend(self, model_id="sdxl-realism", use_4bit=True):
        """Initialize SDXL backend with specified model"""
        if SDXLBackend is None:
            print("Error: SDXLBackend module not found.")
            return False

        try:
            # Map model IDs to HuggingFace paths
            model_mapping = {
                "sdxl-realism": "RunDiffusion/Juggernaut-XL-Lightning",
                "sdxl-anime": "cagliostrolab/animagine-xl-3.1",
                "sdxl-surreal": "Lykon/dreamshaper-xl-lightning"
            }
            # Local Model Resolution Logic
            if model_id.startswith("diffusers/"):
                # Resolve to local diffusers directory
                models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))
                model_path = os.path.join(models_dir, model_id)
                if os.path.exists(model_path):
                    print(f"[SDXL] Resolved local Diffusers model: {model_path}")
                    hf_model = model_path
                else:
                    print(f"[SDXL] Warning: Local model {model_id} not found at {model_path}")
                    hf_model = model_id # Fallback
            
            elif model_id.startswith("checkpoint/"):
                    # Resolve to local checkpoint file
                models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))
                # Need to find the actual file since ID lacks extension
                import glob
                # logic: models/checkpoints/{name}.*
                ckpt_name = model_id.replace("checkpoint/", "")
                search_pattern = os.path.join(models_dir, "checkpoints", f"{ckpt_name}.*")
                matches = glob.glob(search_pattern)
                
                found_ckpt = None
                for m in matches:
                    if m.endswith((".safetensors", ".ckpt", ".bin", ".pt")):
                        found_ckpt = m
                        break
                
                if found_ckpt:
                        print(f"[SDXL] Resolved local Checkpoint: {found_ckpt}")
                        hf_model = found_ckpt
                else:
                        print(f"[SDXL] Warning: Local checkpoint {model_id} not found")
                        hf_model = model_id
            
            elif model_id.startswith("hf/"):
                    # HuggingFace Cache ID (e.g. hf/stabilityai/sdxl-turbo) -> hf/ is just a UI tag, real ID is rest
                    # But actually, backend expects repo ID for HF. 
                    # However, if we want to ensure offline, we might want to resolve to snapshot path if possible.
                    # For now, stripping 'hf/' prefix is enough to let diffusers find it in cache by repo ID.
                    hf_model = model_id.replace("hf/", "")
                    print(f"[SDXL] Resolved HF Cache Model ID: {hf_model}")

            else:
                hf_model = model_mapping.get(model_id, model_id)
            
            # Create or reinitialize if model changed
            if self._instance is None or self._current_model != hf_model:
                # Unload previous instance if it exists to free VRAM
                if self._instance is not None:
                    print(f"[SDXL] Unloading previous model {self._current_model} before loading {hf_model}...")
                    try:
                        # Manually trigger garbage collection and cache clearing
                        self._instance.pipeline = None
                        import torch
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as cleanup_err:
                        print(f"[SDXL] Warning during cleanup: {cleanup_err}")

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
    
    def generate(self, prompt, width=1024, height=1024, steps=8, image=None, strength=0.75, scheduler="euler"):
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
                    strength=strength,
                    scheduler=scheduler
                )
            else:
                # Fallback to direct pipeline call
                import base64
                import io
                
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

# Global instance
try:
    if SDXLBackend:
        sdxl_backend_new = SDXLBackendWrapper()
        print("[SDXL] Backend loaded successfully from local backends directory")
    else:
        raise ImportError("SDXLBackend class not found")
except Exception as e:
    print(f"Warning: Could not initialize SDXL backend wrapper: {e}")
    sdxl_backend_new = None
