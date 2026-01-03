from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import time
import base64
import io
from PIL import Image

from ..hardware_monitor import hw_monitor, model_memory_tracker
from ..sdxl_wrapper import sdxl_backend_new

router = APIRouter()

def _sse_chat_generator(raw_generator, model):
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

async def _handle_chat_completion(request: Request, response: Response = None):
    # Dependencies
    inference_service = request.app.state.inference_service
    config = request.app.state.config
    manifest_manager = request.app.state.manifest_manager
    analytics = getattr(request.app.state, "analytics", None) # Optional

    # Defensively initialize variables used in error handling/cleanup
    vram_mode = "balanced"
    function_name = "unknown"
    kwargs = {}

    try:
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        requested_model = body.get("model")
        
        # --- AUTO-START / SWITCH LOGIC ---
        if not inference_service.is_running():
            target_model = requested_model if requested_model else "moondream-2"
            print(f"DEBUG: Service stopped. Auto-starting {target_model}...")
            
            # Check previous for unloading (edge case where service stopped but tracker has state)
            previous_model = config.get("current_model")
            
            if inference_service.start(target_model):
                    config.set("current_model", target_model)
                    
                    # TRACKER UPDATE
                    try:
                        if previous_model and previous_model != target_model:
                            model_memory_tracker.track_model_unload(previous_model)
                        
                        model_info = manifest_manager.get_models().get(target_model)
                        if model_info:
                            model_memory_tracker.track_model_load(target_model, model_info.name)
                    except Exception as e:
                        print(f"Warning: Failed to track auto-start: {e}")
            else:
                    raise HTTPException(status_code=500, detail="Failed to start model")

        current_model = config.get("current_model")

        # Auto-switch model if requested and different
        if requested_model and requested_model != current_model:
            if requested_model in manifest_manager.get_models():
                print(f"Auto-switching to requested model: {requested_model}")
                
                # Capture previous for unload
                previous_before_switch = current_model
                
                if inference_service.start(requested_model):
                    config.set("current_model", requested_model)
                    current_model = requested_model
                    
                    # TRACKER UPDATE
                    try:
                        if previous_before_switch:
                            model_memory_tracker.track_model_unload(previous_before_switch)
                            print(f"[Tracker] Auto-switch unloaded: {previous_before_switch}")
                            
                        model_info = manifest_manager.get_models().get(requested_model)
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
            if sdxl_backend_new:
                try:
                    print(f"DEBUG: Smart Switching (Mode: {vram_mode}) - Unloading SDXL before analysis...")
                    sdxl_backend_new.unload_backend()
                except Exception as e:
                    print(f"WARNING: Failed to unload SDXL: {e}")
        # --- SMART VRAM MANAGEMENT END ---

        # Execute with OOM Retry
        start_time = time.time()
        try:
            result = await inference_service.execute_function(
                function_name, None, **kwargs
            )
        except Exception as e:
            # Check for OOM
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str or "alloc" in error_str:
                print(f"CRITICAL: OOM detected during Moondream analysis. Logic: Auto-Recovery. Error: {e}")
                
                # 1. Unload everything (Replica of unload_all_models)
                print("Unloading all models (OOM Recovery)...")
                inference_service.stop()
                sdxl_backend_new.unload_backend()
                model_memory_tracker.update_memory_usage()
                
                # 2. Restart Moondream Service
                print(f"Re-initializing Moondream service for model: {model}")
                if inference_service.start(model):
                    print("Service restarted. Retrying operation...")
                    # 3. Retry execution
                    result = await inference_service.execute_function(
                        function_name, None, **kwargs
                    )
                else:
                    raise HTTPException(status_code=500, detail="OOM Recovery failed: Could not restart service.")
            else:
                raise e


        # Handle Streaming
        if stream:
            return StreamingResponse(
                _sse_chat_generator(result, model), 
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

        # --- ADD RESPONSE HEADERS ---
        if response:
            try:
                # Set standard headers
                response.headers["X-Inference-Time"] = str((time.time() - start_time) * 1000)
                
                # Set VRAM headers
                gpus = hw_monitor.get_gpus()
                if gpus and len(gpus) > 0:
                    vram_used = gpus[0].get("memory_used", 0)
                    vram_total = gpus[0].get("memory_total", 0)
                    response.headers["X-VRAM-Used"] = str(vram_used)
                    response.headers["X-VRAM-Total"] = str(vram_total)
            except Exception as e:
                print(f"Warning: Failed to set response headers: {e}")

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
        if analytics:
            analytics.track_error(type(e).__name__, str(e), "api_chat_completions")
        print(f"Error in chat_completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch-caption") # It was POST in rest_server.py
@router.post("/batch-caption")
async def batch_caption(request: Request):
    """
    Batch caption generic images (e.g. WD14 Tagger). 
    Accepts: { "images": ["b64...", ...], "model": "..." }
    """
    inference_service = request.app.state.inference_service
    config = request.app.state.config
    
    if not inference_service.is_running():
        return JSONResponse(content={"error": "Inference service not running"}, status_code=503)

    try:
        data = await request.json()
        images_b64 = data.get("images", [])
        model_id = data.get("model", config.get("current_model"))
        
        if not images_b64 or not isinstance(images_b64, list):
            return JSONResponse(content={"error": "images must be a list of base64 strings"}, status_code=400)

        # Ensure correct model is loaded
        if config.get("current_model") != model_id:
                print(f"[Batch] Switching to {model_id}...")
                if not inference_service.start(model_id):
                    return JSONResponse(content={"error": f"Failed to load model {model_id}"}, status_code=500)
                config.set("current_model", model_id)

        # Decode all images
        pil_images = []
        
        for b64_str in images_b64:
            if b64_str.startswith("data:image"):
                _, encoded = b64_str.split(",", 1)
            else:
                encoded = b64_str
            
            raw_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            pil_images.append(img)

        # Execute Batch
        start_time = time.time()
        captions = await inference_service.execute_function("caption", image=pil_images)
        
        duration = time.time() - start_time

        return {
            "captions": captions,
            "count": len(captions) if isinstance(captions, list) else 1,
            "duration": round(duration, 3)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/stats")
async def get_stats(request: Request):
    """Get inference service stats"""
    inference_service = request.app.state.inference_service
    
    stats = inference_service.get_stats()
    # Add requests processed from session state (if available) -> This belongs to RestServer logic usually, 
    # but we can try to access it if attached, or just return service stats.
    # RestServer attached session_state to self, maybe we can attach to app state?
    # self.app.state.session_state = self.session_state
    
    if hasattr(request.app.state, "session_state") and request.app.state.session_state:
         stats["requests_processed"] = request.app.state.session_state.state["requests_processed"]
    else:
        # Fallback or if session state is missing
        stats["requests_processed"] = stats.get("requests_processed", 0)
        
    return stats

@router.post("/chat/completions")
async def chat_completions(request: Request, response: Response):
    return await _handle_chat_completion(request, response)
