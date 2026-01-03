import json
import time
import psutil
import torch
import os
import sys
import subprocess
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

        # Calculate process memory (including children)
        process_memory_mb = 0
        try:
            process = psutil.Process()
            total_rss = process.memory_info().rss
            
            # Sum up memory of all child processes (e.g. workers)
            for child in process.children(recursive=True):
                try:
                    total_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
            process_memory_mb = total_rss / (1024 * 1024)
        except:
            pass

        status = {
            "platform": "CPU",
            "accelerator_available": False,
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, 'cuda', 'Unknown'),
            "hip_version": getattr(torch.version, 'hip', None),
            "execution_type": execution_type,
            "process_memory_mb": process_memory_mb
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

class OtelMonitor:
    """
    Monitor for OpenTelemetry metrics (Placeholder)
    """
    def __init__(self):
        pass
    
    def get_metrics(self):
        # Placeholder implementation
        return {
            "ghost_memory": {
                "detected": False,
                "ghost_vram_mb": 0
            }
        }

