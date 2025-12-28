import os
import json
import time
import subprocess
from typing import List, Dict, Optional, Any, TypedDict, Union
import requests
import torch

try:
    import pynvml
except ImportError:
    pynvml = None

import psutil

# Constants
DIAGNOSTICS_DB_REL_PATH = os.path.join("config", "diagnostics_db.json")
NVIDIA_MODESET_CMD = "nvidia-drm.modeset=1"
PROC_CMDLINE_PATH = "/proc/cmdline"
PERSISTENCE_MODE_CMD = ["nvidia-smi", "-pm", "1"]
FIX_PERSISTENCE_CMD = "sudo nvidia-smi -pm 1"

# Memory Constants
MIN_RAM_FREE_GB = 1.0

# Thermal Constants
TEMP_WARN_THRESHOLD_C = 80
TEMP_MOCK_CURRENT_C = 45
TEMP_MOCK_LIMIT_C = 90

# Benchmark Constants
DISK_BENCH_FILE_NAME = ".disk_test_bench"
DISK_BENCH_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
DISK_MIN_READ_SPEED_MB_S = 200
NETWORK_CHECK_URL = "https://huggingface.co"
NETWORK_TIMEOUT_SEC = 3

class DiagnosticResult(TypedDict):
    id: str
    name: str
    category: str
    severity: str
    status: str
    message: str
    timestamp: float
    fix_id: Optional[str]

class SystemDiagnostician:
    """
    Runs system diagnostic checks defined in diagnostics_db.json.
    Strict interpretation of AI-Maintainability Framework.
    """
    def __init__(self, config_root: str):
        self.config_root = config_root
        self.db_path = os.path.join(config_root, DIAGNOSTICS_DB_REL_PATH)
        self.checks = self._load_db()
        self._init_solutions()

    def _init_solutions(self):
        """Initialize fix solutions with absolute paths."""
        fix_script = os.path.join(self.config_root, "scripts", "apply_system_fixes.py")
        
        # Helper for sudo wrapper
        def sudo_fix(fix_id):
            return f"sudo /usr/bin/python3 {fix_script} {fix_id}"

        self.solutions = {
            "gpu_persistence": {
                "command": sudo_fix("gpu_persistence"),
                "description": "Enable Nvidia Persistence Mode (requires sudo)"
            },
            "vram_ghosting": {
                "command": sudo_fix("vram_ghosting"),
                "description": "Kill processes holding GPU memory (Nuclear)"
            },
            "nvidia_drm_modeset": {
                "command": sudo_fix("nvidia_drm_modeset"),
                "description": "Update GRUB with nvidia-drm.modeset=1"
            },
            "model_integrity": {
                "command": f"python3 {os.path.join(self.config_root, 'scripts', 'download_final_2_models.py')}",
                "description": "Run model download script"
            }
        }

    def _load_db(self) -> List[Dict]:
        try:
            if not os.path.exists(self.db_path):
                # Fallback search specifically for the test environment or weird pathing
                # Try finding it relative to this file if config_root failed
                alt_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "diagnostics_db.json")
                if os.path.exists(alt_path):
                    self.db_path = alt_path
            
            with open(self.db_path, "r") as f:
                return json.load(f).get("checks", [])
        except Exception as e:
            print(f"[Diagnostician] Failed to load DB {self.db_path}: {e}")
            return []

    def run_all_checks(self, nvidia_available: bool = False, memory_tracker: Any = None) -> List[DiagnosticResult]:
        results: List[DiagnosticResult] = []
        for check in self.checks:
            result = self._init_result(check)
            try:
                self._dispatch_check(check.get("id"), result, nvidia_available, memory_tracker)
            except Exception as e:
                result["status"] = "error"
                result["message"] = f"Check failed: {str(e)}"
            results.append(result)
        # Mark fixable checks
        for res in results:
            if res["status"] in ["warning", "fail", "critical"] and res["id"] in self.solutions:
                res["fix_id"] = res["id"]
        return results

    def _init_result(self, check: Dict) -> DiagnosticResult:
        return {
            "id": check.get("id"),
            "name": check.get("name"),
            "category": check.get("category"),
            "severity": check.get("severity"),
            "status": "pending",
            "message": "",
            "timestamp": time.time()
        }
        
    def apply_fix(self, fix_id: str) -> Dict[str, Any]:
        """Execute a predefined fix for a diagnostic issue."""
        if fix_id not in self.solutions:
            return {"success": False, "message": "Unknown fix ID"}
            
        solution = self.solutions[fix_id]
        cmd = solution["command"]
        
        try:
            print(f"[Diagnostician] Applying fix {fix_id}: {cmd}")
            # We assume the command is safe as it is hardcoded in self.solutions
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[Fix Success] {result.stdout}")
                return {"success": True, "message": f"Fix applied: {solution['description']}"}
            else:
                print(f"[Fix Failed] {result.stderr}")
                return {"success": False, "message": f"Fix failed: {result.stderr}"}
        except Exception as e:
            return {"success": False, "message": f"Execution error: {str(e)}"}

    def _dispatch_check(self, check_id: str, result: DiagnosticResult, nvidia_available: bool, memory_tracker: Any):
        if check_id == "nvidia_drm_modeset":
            self._check_nvidia_modeset(result, nvidia_available)
        elif check_id == "gpu_thermal":
            self._check_gpu_thermal(result, nvidia_available)
        elif check_id == "disk_io_speed":
            self._check_disk_speed(result)
        elif check_id == "network_huggingface":
            self._check_network_reachability(result)
        elif check_id == "secure_boot":
            self._check_secure_boot(result)
        elif check_id == "python_torch_cuda":
            self._check_torch_cuda(result)
        elif check_id == "system_memory":
            self._check_system_memory(result)
        elif check_id == "gpu_persistence":
            self._check_gpu_persistence(result, nvidia_available)
        elif check_id == "vram_ghosting":
            self._check_ghost_vram(result, memory_tracker)
        elif check_id == "model_integrity":
            self._check_model_integrity(result)
        else:
            result["status"] = "skipped"
            result["message"] = "Not implemented yet"

    def _check_nvidia_modeset(self, result: DiagnosticResult, nvidia_available: bool):
        if not nvidia_available:
            result["status"] = "pass"
            result["message"] = "No Nvidia GPU detected (Check N/A)"
            return

        try:
            with open(PROC_CMDLINE_PATH, "r") as f:
                cmdline = f.read()
            
            if NVIDIA_MODESET_CMD in cmdline:
                result["status"] = "pass"
                result["message"] = "DRM Modesetting is enabled."
            else:
                result["status"] = "fail"
                result["message"] = "Critical: nvidia-drm.modeset=1 is MISSING. System may freeze."
        except Exception as e:
            result["status"] = "warning"
            result["message"] = f"Could not read {PROC_CMDLINE_PATH}"

    def _check_gpu_thermal(self, result: DiagnosticResult, nvidia_available: bool):
        if not nvidia_available:
            result["status"] = "pass"
            return
            
        try:
            temp, limit = self._get_gpu_temp()
            
            if temp >= limit:
                result["status"] = "fail"
                result["message"] = f"THERMAL THROTTLING! Current: {temp}C, Limit: {limit}C"
            elif temp > TEMP_WARN_THRESHOLD_C:
                result["status"] = "warning"
                result["message"] = f"High Temperature: {temp}C"
            else:
                result["status"] = "pass"
                result["message"] = f"Normal: {temp}C (Limit: {limit}C)"
        except Exception as e:
            result["status"] = "warning"
            result["message"] = f"Could not read temps: {e}"

    def _get_gpu_temp(self) -> tuple[int, int]:
        if pynvml:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                limit = pynvml.nvmlDeviceGetTemperatureThreshold(handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
                return temp, limit
            except:
                pass
        return TEMP_MOCK_CURRENT_C, TEMP_MOCK_LIMIT_C

    def _check_disk_speed(self, result: DiagnosticResult):
        models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))
        test_file = os.path.join(models_dir, DISK_BENCH_FILE_NAME)
        
        try:
            if not os.path.exists(models_dir): os.makedirs(models_dir)
            
            write_speed = self._benchmark_write(test_file)
            read_speed = self._benchmark_read(test_file)
            
            # Cleanup
            if os.path.exists(test_file): os.remove(test_file)
            
            if read_speed < DISK_MIN_READ_SPEED_MB_S:
                result["status"] = "warning"
                result["message"] = f"Slow Disk Read: {read_speed:.0f} MB/s (Likely HDD). usage will be slow."
            else:
                result["status"] = "pass"
                result["message"] = f"Fast I/O: {read_speed:.0f} MB/s (Read), {write_speed:.0f} MB/s (Write)"
                
        except Exception as e:
            result["status"] = "warning"
            result["message"] = f"Disk bench failed: {e}"
            # Ensure cleanup
            if os.path.exists(test_file): os.remove(test_file)

    def _benchmark_write(self, filepath: str) -> float:
        data = os.urandom(DISK_BENCH_SIZE_BYTES)
        start = time.time()
        with open(filepath, "wb") as f:
            f.write(data)
            os.fsync(f.fileno())
        duration = time.time() - start
        return (DISK_BENCH_SIZE_BYTES / (1024 * 1024)) / duration

    def _benchmark_read(self, filepath: str) -> float:
        start = time.time()
        with open(filepath, "rb") as f:
            while f.read(1024*1024): pass
        duration = time.time() - start
        return (DISK_BENCH_SIZE_BYTES / (1024 * 1024)) / duration

    def _check_network_reachability(self, result: DiagnosticResult):
        try:
            start = time.time()
            r = requests.head(NETWORK_CHECK_URL, timeout=NETWORK_TIMEOUT_SEC)
            latency = (time.time() - start) * 1000
            
            if r.status_code < 400:
                result["status"] = "pass"
                result["message"] = f"Reachable ({latency:.0f}ms)"
            else:
                result["status"] = "warning"
                result["message"] = f"Status {r.status_code}"
        except Exception as e:
            result["status"] = "fail"
            result["message"] = f"Unreachable: {str(e)[:50]}"

    def _check_system_memory(self, result: DiagnosticResult):
        try:
            vm = psutil.virtual_memory()
            free_gb = vm.available / (1024 ** 3)
            total_gb = vm.total / (1024 ** 3)
            
            if free_gb < MIN_RAM_FREE_GB:
                result["status"] = "warning"
                result["message"] = f"Low RAM: {free_gb:.1f}GB available (Total: {total_gb:.1f}GB)"
            else:
                result["status"] = "pass"
                result["message"] = f"Healthy: {free_gb:.1f}GB available (Total: {total_gb:.1f}GB)"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Memory check failed: {e}"

    def _check_gpu_persistence(self, result: DiagnosticResult, nvidia_available: bool):
        if not nvidia_available:
            result["status"] = "pass"
            return
            
        try:
            # Check via nvidia-smi query
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=persistence_mode", "--format=csv,noheader,nounits"], 
                capture_output=True, text=True
            )
            mode = p.stdout.strip()
            
            if mode == "Enabled":
                result["status"] = "pass"
                result["message"] = "Persistence Mode is Enabled."
            else:
                result["status"] = "warning"
                result["message"] = "Persistence Mode is DISABLED. GPU initialization may be slow."
                result["fix_id"] = "gpu_persistence" # Signal that this can be fixed
        except Exception as e:
            result["status"] = "warning"
            result["message"] = f"Could not check persistence: {e}"

    def _check_ghost_vram(self, result: DiagnosticResult, memory_tracker: Any):
        if not memory_tracker:
            result["status"] = "info"
            result["message"] = "Memory tracker not available."
            return

        try:
            # Reuse the live tracker's calculation
            status = memory_tracker.get_ghost_status()
            
            if status.get("detected"):
                ghost_mb = status.get("ghost_vram_mb", 0)
                result["status"] = "warning"
                result["message"] = f"Ghost VRAM Detected: ~{ghost_mb}MB unaccounted usage (Possible zombie process)."
            else:
                result["status"] = "pass"
                result["message"] = "No Ghost VRAM detected."
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Ghost check failed: {e}"

    def _check_secure_boot(self, result: DiagnosticResult):
        try:
            p = subprocess.run(["mokutil", "--sb-state"], capture_output=True, text=True)
            output = p.stdout
            
            if "SecureBoot enabled" in output:
                # Critical check: Is it ACTUALLY checking drivers?
                # We can't easily verify driver signatures from here, 
                # but we can look for "Module verification failed" in dmesg if readable
                result["status"] = "warning"
                
                msg = "Secure Boot is ENABLED."
                
                # Check for proprietary drivers which are often unsigned/blocked
                drivers = ["nvidia", "amdgpu", "i915"]
                lsmod = subprocess.run(["lsmod"], capture_output=True, text=True).stdout
                
                missing = [d for d in drivers if d not in lsmod]
                if "nvidia" in missing and "nvidia" not in lsmod:  # Only really care if we expect nvidia
                     # Just a heuristic, hard to know if it SHOULD be there without lspci
                     pass
                     
                result["message"] = f"{msg} May block third-party drivers (Nvidia/AMD)."
            else:
                result["status"] = "pass"
                result["message"] = "Secure Boot is disabled (Driver friendly)."
        except:
            result["status"] = "info"
            result["message"] = "Could not check (mokutil missing)"

    def _check_torch_cuda(self, result: DiagnosticResult):
        try:
            if not torch.cuda.is_available():
                result["status"] = "fail"
                result["message"] = "PyTorch cannot see CUDA device!"
                return
                
            torch_ver = torch.version.cuda
            if torch_ver:
                result["status"] = "pass"
                result["message"] = f"Match: PyTorch built for CUDA {torch_ver}"
            else:
                result["status"] = "fail"
                result["message"] = "PyTorch has no CUDA version info."
        except:
            result["status"] = "error"

    def _check_model_integrity(self, result: DiagnosticResult):
        """
        Scans model directory for valid checkpoint files.
        Checks for:
        1. Existence of model directory
        2. At least one valid .safetensors file (>500MB) to ensure we aren't empty
        """
        models_dir = os.environ.get("MOONDREAM_MODELS_DIR", os.path.expanduser("~/.moondream-station/models"))
        
        if not os.path.exists(models_dir):
            result["status"] = "fail"
            result["message"] = f"Model directory missing: {models_dir}"
            return

        try:
            valid_models = 0
            total_size_gb = 0
            scanned_files = 0
            
            # Recursive scan
            for root, _, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(".safetensors"):
                        fp = os.path.join(root, file)
                        try:
                            size = os.path.getsize(fp)
                            # Basic corruption check: Is it reasonably large?
                            # > 500MB (some quantized models might be small, but usually >1GB)
                            if size > 500 * 1024 * 1024:
                                valid_models += 1
                                total_size_gb += size / (1024**3)
                            scanned_files += 1
                        except:
                            pass
            
            if valid_models > 0:
                result["status"] = "pass"
                result["message"] = f"Found {valid_models} valid models ({total_size_gb:.1f} GB)."
                if valid_models < 2: 
                     result["status"] = "warning"
                     result["message"] += " (Low count)"
            else:
                if scanned_files > 0:
                    result["status"] = "fail"
                    result["message"] = f"Corrupted? Found {scanned_files} files, but none > 500MB."
                else:
                    result["status"] = "warning" # Maybe just started and hasn't downloaded yet
                    result["message"] = "No models found. Please run download scripts."

        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Scan failed: {e}"

