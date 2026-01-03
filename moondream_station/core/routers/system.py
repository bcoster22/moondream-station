from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import subprocess
import shutil
import time
import requests
import re
import sys
import importlib.metadata
from packaging import version
import torch

from ..hardware_monitor import hw_monitor, model_memory_tracker

router = APIRouter()

@router.get("/prime-profile")
async def get_prime_profile():
    """Get the current NVIDIA Prime profile (nvidia/on-demand/intel)"""
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

@router.post("/prime-profile")
async def set_prime_profile(profile: str):
    """Set the NVIDIA Prime profile (requires sudo/root via script)"""
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

@router.get("/zombie-killer")
async def get_zombie_killer_status(request: Request):
    """Get auto-free zombie VRAM status"""
    config = request.app.state.config
    enabled = config.get("zombie_killer_enabled", False)
    interval = config.get("zombie_killer_interval", 60)
    return {"enabled": enabled, "interval": interval}

@router.post("/zombie-killer")
async def toggle_zombie_killer(request: Request):
    """Toggle auto-free zombie VRAM feature"""
    try:
        data = await request.json()
        config = request.app.state.config
        
        if "enabled" in data:
            config.set("zombie_killer_enabled", bool(data["enabled"]))
        if "interval" in data:
            config.set("zombie_killer_interval", max(30, int(data["interval"])))
        
        enabled = config.get("zombie_killer_enabled", False)
        interval = config.get("zombie_killer_interval", 60)
        
        print(f"[Zombie Killer] Auto-free VRAM: {'ENABLED' if enabled else 'DISABLED'} (interval: {interval}s)")
        return {"enabled": enabled, "interval": interval}
    except Exception as e:
        return {"error": str(e)}

@router.get("/dev-mode")
async def get_dev_mode(request: Request):
    """Get Dev Mode status"""
    config = request.app.state.config
    enabled = config.get("dev_mode", False)
    return {"enabled": enabled}

@router.post("/dev-mode")
async def set_dev_mode(request: Request):
    """Toggle Dev Mode"""
    try:
        data = await request.json()
        config = request.app.state.config
        
        if "enabled" in data:
            config.set("dev_mode", bool(data["enabled"]))
        
        enabled = config.get("dev_mode", False)
        return {"enabled": enabled}
    except Exception as e:
        return {"error": str(e)}

@router.get("/backend-version")
async def get_backend_versions():
    """
    Get versions of key backend libraries and check for critical updates.
    """
    libs = ['torch', 'torchvision', 'torchaudio', 'diffusers', 'transformers', 'accelerate', 'moondream']
    versions = {}
    for lib in libs:
        try:
            version_str = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            version_str = "Not Installed"
        
        versions[lib] = version_str
        
    # Check for critical vulnerability CVE-2025-32434
    current_torch = versions.get('torch', '0.0.0').split('+')[0]
    has_critical_update = False
    critical_message = ""
    
    try:
        if version.parse(current_torch) < version.parse("2.6.0"):
            has_critical_update = True
            critical_message = "CRITICAL: Torch version < 2.6.0 detected. Vulnerability CVE-2025-32434 present. Upgrade immediately."
    except Exception:
        pass

    return {
        "versions": versions,
        "has_critical_update": has_critical_update,
        "critical_message": critical_message,
        "python_version": sys.version.split(' ')[0],
        "platform": sys.platform
    }

@router.post("/upgrade-backend")
async def upgrade_backend():
    """
    Upgrade backend dependencies to fix security vulnerabilities.
    Specifically targets torch 2.6.0+ and compatible libraries.
    """
    # Command to upgrade torch, torchvision, torchaudio to 2.6.0+ for CUDA 12.4 (compatible with 12.6)
    pip_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch>=2.6.0", "torchvision>=0.21.0", "torchaudio>=2.6.0", 
        "diffusers>=0.36.0", "transformers>=4.48.0", "accelerate>=1.3.0",
        "--index-url", "https://download.pytorch.org/whl/cu124", 
        "--upgrade", "--no-cache-dir"
    ]
    
    print(f"[System] Upgrading backend: {' '.join(pip_cmd)}")
    
    try:
        process = subprocess.run(
            pip_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for big downloads
        )
        
        if process.returncode == 0:
            return {"status": "success", "message": "Backend upgraded successfully. Please restart the backend server."}
        else:
            return JSONResponse(status_code=500, content={
                "status": "error", 
                "message": "Upgrade failed", 
                "details": process.stderr[-1000:]  # Last 1000 chars
            })
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504, content={"status": "error", "message": "Upgrade timed out (download taking too long). Check server logs."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@router.get("/verify-backend")
async def verify_backend():
    """
    Run comprehensive system health checks.
    """
    checks = []
    all_passed = True
    
    def add_result(name, passed, message):
        nonlocal all_passed
        if not passed: all_passed = False
        checks.append({"name": name, "passed": passed, "message": message})

    # 1. Framework Versions
    try:
        torch_ver = importlib.metadata.version('torch')
        if version.parse(torch_ver) >= version.parse("2.6.0"):
            add_result("PyTorch Version", True, f"v{torch_ver} (Secure)")
        else:
            add_result("PyTorch Version", False, f"v{torch_ver} (Insecure - CVE-2025-32434)")
    except Exception:
        add_result("PyTorch Version", False, "Not Installed")

    # 2. CUDA Availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        add_result("CUDA GPU", True, f"Available: {device_name}")
    else:
        add_result("CUDA GPU", False, "Not available (Running on CPU)")

    # 3. VRAM Check
    try:
        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            total_mem = torch.cuda.mem_get_info()[1] / 1024**3
            add_result("VRAM", True, f"{free_mem:.1f}GB free / {total_mem:.1f}GB total")
        else:
            add_result("VRAM", True, "N/A (CPU Mode)")
    except Exception as e:
        add_result("VRAM", False, f"Error checking VRAM: {str(e)}")

    # 4. Disk Space
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / 1024**3
        if free_gb > 10:
            add_result("Disk Space", True, f"{free_gb:.1f} GB available")
        elif free_gb > 2:
            add_result("Disk Space", True, f"Low space: {free_gb:.1f} GB available (Warning)")
        else:
            add_result("Disk Space", False, f"Critical low space: {free_gb:.1f} GB")
    except Exception as e:
        add_result("Disk Space", False, f"Error: {str(e)}")

    # 5. Network Connectivity (HuggingFace)
    try:
        start = time.time()
        requests.head("https://huggingface.co", timeout=3)
        latency = (time.time() - start) * 1000
        add_result("Network (HuggingFace)", True, f"Reachable ({latency:.0f}ms)")
    except Exception:
        add_result("Network (HuggingFace)", False, "Unreachable - Models cannot download")

    # 6. FFmpeg Check
    if shutil.which("ffmpeg"):
        add_result("FFmpeg", True, "Installed (Video generation ready)")
    else:
        add_result("FFmpeg", False, "Not found (Video/Audio generation will fail)")
        
    # 7. Functional Test (Tensor Op)
    try:
        if torch.cuda.is_available():
            x = torch.rand(5, 5).cuda()
            y = x * x
            add_result("GPU Tensor Op", True, "Pass")
        else:
            add_result("GPU Tensor Op", True, "Skipped (CPU Mode)")
    except Exception as e:
        add_result("GPU Tensor Op", False, f"Failed: {str(e)}")

    return {"checks": checks, "overallStatus": "ok" if all_passed else "failed"}

@router.post("/update-packages")
async def update_packages(request: Request):
    """
    Update specified npm packages.
    """
    try:
        body = await request.json()
        packages = body.get("packages", [])
        
        if not packages or not isinstance(packages, list):
            return JSONResponse(status_code=400, content={"status": "error", "message": "No packages specified"})

        # Security: Validate package names (alphanumeric, -, @, /)
        safe_packages = []
        for pkg in packages:
            if re.match(r"^[@a-zA-Z0-9\-\/\.\^~]+$", pkg):
                safe_packages.append(pkg)
            else:
                print(f"Skipping invalid package name: {pkg}")
        
        if not safe_packages:
            return JSONResponse(status_code=400, content={"status": "error", "message": "No valid packages provided"})

        # Construct command: npm install pkg1@latest pkg2@latest ...
        cmd = ["npm", "install"] + [f"{pkg}@latest" for pkg in safe_packages]
        
        print(f"[System] Updating frontend packages: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        
        if process.returncode == 0:
            return {"status": "success", "message": f"Successfully updated: {', '.join(safe_packages)}"}
        else:
            return JSONResponse(status_code=500, content={
                "status": "error", 
                "message": "Update failed", 
                "details": process.stderr
            })
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@router.post("/gpu-boost")
async def gpu_boost(request: Request):
    """Enable/disable GPU boost mode (max fans + persistence)"""
    try:
        params = dict(request.query_params)
        gpu_id = int(params.get("gpu_id", 0))
        enable = params.get("enable", "true").lower() == "true"
        
        # Validate gpu_id
        if gpu_id < 0:
            raise HTTPException(status_code=400, detail="Invalid gpu_id: must be >= 0")
        
        # Query GPU's actual max power limit dynamically
        try:
            query_result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if query_result.returncode == 0:
                max_power = float(query_result.stdout.strip())
                boost_power = int(max_power)  # Use actual max
                normal_power = int(max_power * 0.75)  # 75% of max
            else:
                # Fallback to conservative defaults if query fails
                boost_power = 250
                normal_power = 200
        except Exception:
            # Fallback defaults
            boost_power = 250
            normal_power = 200
        
        if enable:
            # Boost Mode: Max fans + Persistence
            commands = [
                ["sudo", "-n", "nvidia-smi", "-i", str(gpu_id), "-pm", "1"],  # Persistence mode ON
                ["sudo", "-n", "nvidia-smi", "-i", str(gpu_id), "-pl", str(boost_power)]  # Max power
            ]
            # Try to set fan speed if supported
            try:
                subprocess.run(["sudo", "-n", "nvidia-settings", "-a", f"[gpu:{gpu_id}]/GPUFanControlState=1"], timeout=5)
                subprocess.run(["sudo", "-n", "nvidia-settings", "-a", f"[fan:{gpu_id}]/GPUTargetFanSpeed=100"], timeout=5)
            except Exception:
                pass  # Fan control not always supported
        else:
            # Normal Mode: Auto fans + Default settings
            commands = [
                ["sudo", "-n", "nvidia-smi", "-i", str(gpu_id), "-pm", "0"],  # Persistence mode OFF
                ["sudo", "-n", "nvidia-smi", "-i", str(gpu_id), "-pl", str(normal_power)]  # 75% power
            ]
            # Reset fan to auto
            try:
                subprocess.run(["sudo", "-n", "nvidia-settings", "-a", f"[gpu:{gpu_id}]/GPUFanControlState=0"], timeout=5)
            except Exception:
                pass

        # Execute commands
        for cmd in commands:
            subprocess.run(cmd, check=True, timeout=5)
            
        return {"status": "success", "mode": "boost" if enable else "normal", "power_limit": boost_power if enable else normal_power}
        
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"GPU command failed. Ensure sudo is configured without password for nvidia-smi. Error: {str(e)}")
    except Exception as e:
         if hasattr(e, "status_code"): raise e
         raise HTTPException(status_code=500, detail=str(e))
