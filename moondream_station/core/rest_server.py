import asyncio
import json
import time
import uvicorn
import psutil
import torch
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
                
                return {
                    "cpu": cpu,
                    "memory": memory,
                    "device": device,
                    "gpus": gpus,
                    "environment": env
                }
            except Exception as e:
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
                cmd = ["sudo", "-n", "nvidia-smi", "--gpu-reset", "-i", str(gpu_id)]
                
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
                    detail=f"Reset failed: {process.stderr.strip()}"
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

        @self.app.get("/v1/models")
        async def list_models():
            try:
                models = self.manifest_manager.get_models()
                return {
                    "models": [
                        {
                            "id": model_id,
                            "name": model_info.name,
                            "description": model_info.description,
                            "version": getattr(model_info, "version", "unknown"),
                        }
                        for model_id, model_info in models.items()
                    ]
                }
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
                self.config.set("current_model", model_id)
                return {"status": "success", "model": model_id}
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
        if not self.inference_service.is_running():
            raise HTTPException(status_code=503, detail="Inference service not running")

        try:
            body = await request.json()
            messages = body.get("messages", [])
            stream = body.get("stream", False)
            requested_model = body.get("model")
            current_model = self.config.get("current_model")

            # Auto-switch model if requested and different
            if requested_model and requested_model != current_model:
                if requested_model in self.manifest_manager.get_models():
                    print(f"Auto-switching to requested model: {requested_model}")
                    if self.inference_service.start(requested_model):
                        self.config.set("current_model", requested_model)
                        current_model = requested_model
                    else:
                        raise HTTPException(status_code=500, detail=f"Failed to switch to model {requested_model}")
                else:
                    raise HTTPException(status_code=404, detail=f"Model {requested_model} not found")

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
                # Text only - not supported by VLM usually, but maybe query without image?
                # For now, error or try query
                raise HTTPException(status_code=400, detail="Image required for Moondream")

            # Execute
            start_time = time.time()
            result = await self.inference_service.execute_function(
                function_name, None, **kwargs
            )

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
                if self.inference_service.start(requested_model):
                    self.config.set("current_model", requested_model)
                    current_model = requested_model
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

            # Record the request in session state
            if self.session_state:
                self.session_state.record_request(f"/{path}")

            success = not (isinstance(result, dict) and result.get("error"))
        except Exception as e:
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
