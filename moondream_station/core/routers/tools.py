from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import os
import json
import subprocess
from .models import discover_models_from_directories

router = APIRouter()

@router.post("/convert")
async def convert_model(request: Request):
    """
    Convert a Diffusers model to a single .safetensors file.
    Streams output logs.
    """
    try:
        data = await request.json()
        model_id = data.get("model_id")
        fp16 = data.get("fp16", True)

        if not model_id:
                return JSONResponse(content={"error": "model_id is required"}, status_code=400)

        # Resolve model path using shared discovery logic
        config = request.app.state.config
        discovered = discover_models_from_directories(config)
        target_model = next((m for m in discovered if m["id"] == model_id), None)
        
        if not target_model:
                return JSONResponse(content={"error": f"Model {model_id} not found"}, status_code=404)
        
        model_path = target_model["file_path"]
        
        # Determine script path (Relative to this file: ../../scripts/...)
        # Current file: .../moondream_station/core/routers/tools.py
        # Root: .../moondream_station/
        # Scripts: .../moondream_station/scripts/
        
        # We need to go up 3 levels from here to get to package root?
        # core/routers/tools.py -> core/routers -> core -> moondream_station -> scripts?
        # No, structure is moondream_station/core/.
        # So: dir(tools.py) -> routers -> core -> moondream_station.
        # Scripts is typicaly at root of repo or inside package?
        # The original code used: os.path.dirname(os.path.dirname(os.path.dirname(__file__))) from rest_server.py
        # rest_server.py is in core/.
        # core/ -> moondream_station/ -> root?
        # Let's trust the relative path logic but adjust for being one level deeper (routers/).
        
        # original rest_server (in core): dirname(dirname(dirname(__file__))) -> core -> moondream_station -> src?
        # Let's verify where 'scripts' are.
        # Assuming moondream_station package structure.
        
        current_dir = os.path.dirname(os.path.abspath(__file__)) # core/routers
        core_dir = os.path.dirname(current_dir) # core
        pkg_dir = os.path.dirname(core_dir) # moondream_station (package root)
        
        # Check if scripts is in package root or one level up?
        # I'll check file system if I can, or guess.
        # Original was: os.path.dirname(os.path.dirname(os.path.dirname(__file__))) from rest_server.py
        # rest_server is in core/. 
        # dirname(rest_server) = core
        # dirname(core) = moondream_station
        # dirname(moondream_station) = src or root?
        # Then "scripts" folder.
        
        # From routers/tools.py:
        # dirname(tools.py) = routers
        # dirname(routers) = core
        # dirname(core) = moondream_station
        # dirname(moondream_station) = src/root
        
        # So I need 4 dirnames to match original depth from rest_server (which used 3).
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        script_path = os.path.join(base_path, "scripts", "convert_diffusers_to_safetensors.py")
        
        if not os.path.exists(script_path):
             # Try one level deeper just in case
             base_path_alt = os.path.dirname(base_path)
             script_path_alt = os.path.join(base_path_alt, "scripts", "convert_diffusers_to_safetensors.py")
             if os.path.exists(script_path_alt):
                 script_path = script_path_alt
             else:
                 # Fallback to blindly assuming standard location relative to CWD if running from root
                 script_path = os.path.abspath("scripts/convert_diffusers_to_safetensors.py")

        if not os.path.exists(script_path):
                return JSONResponse(content={"error": f"Conversion script not found at {script_path}"}, status_code=500)

        # Determine output path
        model_dirname = os.path.dirname(model_path)
        model_basename = os.path.basename(model_path)
        output_filename = f"{model_basename}.safetensors"
        output_path = os.path.join(model_dirname, output_filename)

        # Build command
        cmd = [
            "python3", 
            "-u", # Unbuffered
            script_path,
            "--model_path", model_path,
            "--output_path", output_path
        ]
        
        if fp16:
            cmd.append("--fp16")

        print(f"[Conversion] Running: {' '.join(cmd)}")

        async def generate_output():
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            yield f"data: {json.dumps({'message': f'Starting conversion for {model_id}...'})}\n\n"
            yield f"data: {json.dumps({'message': f'Input: {model_path}'})}\n\n"
            yield f"data: {json.dumps({'message': f'Output: {output_path}'})}\n\n"

            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'message': line.strip()})}\n\n"
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code == 0:
                yield f"data: {json.dumps({'message': 'Conversion completed successfully!', 'type': 'success'})}\n\n"
                yield f"data: {json.dumps({'completed': True, 'success': True})}\n\n"
            else:
                yield f"data: {json.dumps({'message': f'Conversion failed with exit code {return_code}', 'type': 'error'})}\n\n"
                yield f"data: {json.dumps({'completed': True, 'success': False})}\n\n"

        return StreamingResponse(generate_output(), media_type="text/event-stream")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
