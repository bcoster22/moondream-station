from fastapi import APIRouter, Request, HTTPException
import os
import subprocess

from ..hardware_monitor import hw_monitor, model_memory_tracker

router = APIRouter()

@router.get("/scan")
async def run_diagnostics():
    """Run all system diagnostics checks"""
    from moondream_station.core.system_diagnostics import SystemDiagnostician
    config_root = os.path.expanduser("~/.moondream-station")
    diagnostician = SystemDiagnostician(config_root)
    return {"checks": diagnostician.run_all_checks(
        nvidia_available=hw_monitor.nvidia_available,
        memory_tracker=model_memory_tracker
    )}

@router.post("/fix/{fix_id}")
async def fix_diagnostic(fix_id: str):
    """Execute a fix for a specific diagnostic issue"""
    from moondream_station.core.system_diagnostics import SystemDiagnostician
    config_root = os.path.expanduser("~/.moondream-station")
    diagnostician = SystemDiagnostician(config_root)
    result = diagnostician.apply_fix(fix_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@router.get("/backend-health")
async def backend_health(request: Request):
    """Check if inference service is available and initialized"""
    # Requires access to inference service via app.state or dependency injection
    # In rest_server.py constructor: self.app.state.inference_service = self.inference_service
    
    # Check if we have access to inference service, otherwise we need to rely on it being attached to app state
    if hasattr(request.app.state, "inference_service"):
         inference_service = request.app.state.inference_service
         return {
            "backend_imported": True,
            "inference_service_running": inference_service.is_running(),
            "status": "ready" if inference_service.is_running() else "stopped"
        }
    else:
        return {"status": "error", "message": "Inference service not linked to app state"}

@router.post("/setup-autofix")
async def setup_autofix(request: Request):
    """
    Run the Auto Fix setup script with sudo password.
    This configures passwordless sudo for the fix wrapper script.
    """
    import subprocess
    import os
    
    try:
        body = await request.json()
        password = body.get("password", "")
        
        if not password:
            raise HTTPException(status_code=400, detail="Password required")
        
        # Path to setup script (dynamic, portable)
        # Assumes this router file is in moondream_station/core/routers/
        # Setup script is in moondream_station/ (root of package installed?) or maybe in site-packages/moondream_station
        # Let's find relative to this file: ../../../setup_gpu_reset.sh is risky if structure changes.
        # But wait, original code was: os.path.dirname(os.path.dirname(os.path.abspath(__file__))) which is moondream_station/core/.. = moondream_station/
        # If we are in routers/, then ../../.. is moondream_station/.
        
        # Safer to find where the package is installed or where backend is running.
        # Let's assume standard layout.
        
        # Original: /home/bcoster/.moondream-station/moondream-station/moondream_station/core/rest_server.py
        # New: /home/bcoster/.moondream-station/moondream-station/moondream_station/core/routers/diagnostics.py
        
        # The script `setup_gpu_reset.sh` is likely in the repo root or package root.
        # Original logic: script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) -> moondream_station
        
        # Let's try to find it relative to `core` module.
        core_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # moondream_station/core/routers -> moondream_station
        script_path = os.path.join(core_dir, "..", "setup_gpu_reset.sh") # moondream_station/../setup_gpu_reset.sh
        
        # Actually, let's look at the original location again.
        # Original `__file__` (rest_server.py) -> moondream_station/core/rest_server.py
        # `os.path.dirname` -> moondream_station/core
        # `os.path.dirname` -> moondream_station
        # So it was in `moondream_station/`.
        
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script_path = os.path.join(script_dir, "setup_gpu_reset.sh")

        if not os.path.exists(script_path):
             # Fallback to current working directory if not found (development mode)
             cwd_script = os.path.join(os.getcwd(), "setup_gpu_reset.sh")
             if os.path.exists(cwd_script):
                 script_path = cwd_script
             else:
                 raise HTTPException(status_code=500, detail=f"Setup script not found at {script_path}")
        
        # Execute with sudo -S (read password from stdin)
        process = subprocess.Popen(
            ["sudo", "-S", "bash", script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=password + "\n", timeout=30)
        
        if process.returncode == 0:
            return {"success": True, "message": "Auto Fix setup completed successfully"}
        else:
            if "Sorry, try again" in stderr or "authentication failure" in stderr.lower():
                raise HTTPException(status_code=401, detail="Incorrect password")
            raise HTTPException(status_code=500, detail=f"Setup failed: {stderr}")
            
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Setup script timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
