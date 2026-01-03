import asyncio
import sys
import os
from .sdxl_wrapper import sdxl_backend_new

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

from .hardware_monitor import hw_monitor, model_memory_tracker, OtelMonitor


from threading import Thread
from typing import Any, Dict
from fastapi import FastAPI, Request, Response, HTTPException
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
            expose_headers=["X-VRAM-Used", "X-VRAM-Total"],  # CRITICAL: Expose VRAM headers for frontend polling
        )
        self.server = None
        self.server_thread = None
        self.zombie_killer_task = None  # Background task for auto-free VRAM
        
        # Attach services to app state for Routers
        self.app.state.inference_service = self.inference_service
        self.app.state.config = self.config
        self.app.state.manifest_manager = self.manifest_manager
        self.app.state.session_state = self.session_state
        self.app.state.analytics = self.analytics
        
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
                    "loaded_models": loaded_models,
                    "cpu_details": {
                        "usage": cpu,
                        "cores": psutil.cpu_count(logical=True)
                    },
                    "memory_details": {
                        "used_gb": psutil.virtual_memory().used / (1024**3),
                        "total_gb": psutil.virtual_memory().total / (1024**3),
                        "percent": memory
                    }
                }
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                return {"cpu": 0, "memory": 0, "device": "Unknown", "gpus": [], "loaded_models": []}

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
                return JSONResponse(content={"error": str(e)}, status_code=500)
        # Mount System Router
        from .routers.system import router as system_router
        self.app.include_router(system_router, prefix="/v1/system", tags=["System"])

        # Legacy direct routes mapping for compatibility if needed (or ensure frontend uses /v1/system prefix)
        # Note: All extracted routes in system.py use relative paths, so mounting at /v1/system works perfectly.
        # e.g. /prime-profile becomes /v1/system/prime-profile as expected.


        # Mount Tools Router
        from .routers.tools import router as tools_router
        self.app.include_router(tools_router, prefix="/v1/tools", tags=["Tools"])



        # Mount Generation Router
        from .routers.generation import router as generation_router
        self.app.include_router(generation_router, prefix="/v1", tags=["Generation"])

        # Mount Models Router
        from .routers.models import router as models_router
        self.app.include_router(models_router, prefix="/v1/models", tags=["Models"])

        # Mount Vision Router  
        from .routers.vision import router as vision_router
        self.app.include_router(vision_router, prefix="/v1", tags=["Vision"])

        # Mount Diagnostics Router
        from .routers.diagnostics import router as diagnostics_router
        self.app.include_router(diagnostics_router, prefix="/diagnostics", tags=["Diagnostics"])

        @self.app.api_route(
            "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
        )
        async def dynamic_route(request: Request, path: str):
            return await self._handle_dynamic_request(request, path)
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
            # Allow dynamic models (e.g. HF IDs) to be passed through even if not in manifest
            # if requested_model in self.manifest_manager.get_models():
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
                    else:
                        # Dynamic model - try to track with basic info
                        model_memory_tracker.track_model_load(requested_model, requested_model)
                except Exception as e:
                    print(f"[Tracker] Warning: Failed to track auto-switch: {e}")
            else:
                # If start() returns False, it really logic failed
                raise HTTPException(status_code=500, detail=f"Failed to switch to model {requested_model}")
            # else:
            #    raise HTTPException(status_code=404, detail=f"Model {requested_model} not found")

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
            vram_mode = self.config.get("vram_mode", "balanced")
            should_unload = request.headers.get("X-Auto-Unload") == "true" or \
                            os.environ.get("MOONDREAM_AUTO_UNLOAD") == "true" or \
                            vram_mode == "low"
            
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

            headers = {}
            try:
                gpus = hw_monitor.get_gpus()
                if gpus:
                    headers["X-VRAM-Used"] = str(gpus[0]["memory_used"])
                    headers["X-VRAM-Total"] = str(gpus[0]["memory_total"])
            except: pass

            return JSONResponse(result, headers=headers)

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

    async def _zombie_killer_worker(self):
        """Background task that monitors for zombie VRAM and auto-frees it"""
        import asyncio
        otel = OtelMonitor()
        
        while True:
            try:
                # Wait for configured interval
                interval = self.config.get("zombie_killer_interval", 60)
                await asyncio.sleep(interval)
                
                # Check if feature is enabled
                if not self.config.get("zombie_killer_enabled", False):
                    continue
                
                # Get OTEL metrics
                metrics = otel.get_metrics()
                
                # Check for ghost memory
                if metrics.get("ghost_memory", {}).get("detected", False):
                    ghost_vram = metrics["ghost_memory"].get("ghost_vram_mb", 0)
                    print(f"[Zombie Killer] Detected {ghost_vram:.1f}MB zombie VRAM, auto-freeing...")
                    
                    # Unload model to free VRAM
                    if self.inference_service.is_running():
                        self.inference_service.stop()
                        print("[Zombie Killer] Model unloaded successfully")
                    else:
                        print("[Zombie Killer] No model currently loaded")
                        
            except Exception as e:
                print(f"[Zombie Killer] Error: {e}")

    def start_zombie_killer(self):
        """Start the zombie killer background task (called when server starts)"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            if self.zombie_killer_task is None:
                self.zombie_killer_task = loop.create_task(self._zombie_killer_worker())
                print("[Zombie Killer] Background monitoring started")
        except RuntimeError:
            print("[Zombie Killer] Warning: No event loop running, task will start with server")
