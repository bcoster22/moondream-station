import asyncio

from typing import Any, Dict, Optional

from .simple_worker_pool import SimpleWorkerPool

N_WORKERS = 1
MAX_QUEUE_SIZE = 10
TIMOUT = 30


class InferenceService:
    def __init__(self, config, manifest_manager):
        self.config = config
        self.manifest_manager = manifest_manager
        self.worker_pool = None
        self.current_model = None
        self.worker_backends = []

    def cleanup_memory(self):
        """Force GC and CUDA cache clear to remove initialization artifacts."""
        import gc
        import torch
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem = torch.cuda.memory_allocated() / 1024 / 1024
                # Using print for safety
                print(f"[InferenceService] Post-Load Cleanup Complete. Active VRAM: {mem:.2f} MB")
        except:
            pass

    def start(self, model_id: str):
        # Force unload previous model to free VRAM/RAM
        if self.current_model and self.current_model != model_id:
            print(f"[InferenceService] Switching model from {self.current_model} to {model_id}. Unloading previous...")
            self.unload_model()
            
        n_workers = int(self.config.get("inference_workers", N_WORKERS))
        max_queue_size = int(
            self.config.get("inference_max_queue_size", MAX_QUEUE_SIZE)
        )
        timeout = float(self.config.get("inference_timeout", TIMOUT))

        if self.worker_pool:
            self.worker_pool.shutdown()

        self.manifest_manager.clear_worker_backends()
        
        # Double check cleanup
        import gc
        gc.collect()

        self.current_model = model_id
        self.worker_backends = self.manifest_manager.get_worker_backends(
            model_id, n_workers
        )

        if not self.worker_backends:
            return False

        # --- POST-LOAD CLEANUP ---
        self.cleanup_memory()

        self.worker_pool = SimpleWorkerPool(n_workers, max_queue_size, timeout)
        return True

    async def stop(self):
        if self.worker_pool:
            self.worker_pool.shutdown()
            self.worker_pool = None
        self.worker_backends = []
        self.current_model = None

    async def execute_function(
        self, function_name: str, timeout: Optional[float] = None, **kwargs
    ) -> Dict[str, Any]:
        if not self.worker_pool or not self.worker_backends:
            return {"error": "Inference service not started"}

        backend = self._get_next_backend()
        if not backend:
            import sys
            sys.stderr.write(f"DEBUG: backend is None for '{function_name}'\n")
            return {"error": f"Function '{function_name}' not available"}

        import sys
        sys.stderr.write(f"DEBUG: execute_function called for '{function_name}'\n")
        sys.stderr.write(f"DEBUG: backend object: {backend}\n")
        if hasattr(backend, "__dir__"):
             sys.stderr.write(f"DEBUG: backend dir: {dir(backend)}\n")
        else:
             sys.stderr.write("DEBUG: backend has no __dir__\n")

        if not hasattr(backend, function_name):
            sys.stderr.write(f"DEBUG: hasattr failed for '{function_name}'\n")
            return {"error": f"Function '{function_name}' not available"}

        func = getattr(backend, function_name)

        loop = asyncio.get_event_loop()

        def submit_with_kwargs():
            return self.worker_pool.submit_request(func, timeout, **kwargs)

        result = await loop.run_in_executor(None, submit_with_kwargs)
        return result

    def _get_next_backend(self):
        if not self.worker_backends:
            return None
        return self.worker_backends[0]

    def get_stats(self) -> Dict[str, Any]:
        if not self.worker_pool:
            return {"status": "stopped"}

        stats = self.worker_pool.get_stats()
        stats["model"] = self.current_model
        stats["status"] = "running"
        return stats

        return True

    def unload_model(self):
        print("[InferenceService] Unloading model...")
        if self.worker_pool:
            print("[InferenceService] Shutting down worker pool...")
            self.worker_pool.shutdown()
            self.worker_pool = None
        
        print("[InferenceService] Unloading backends from manifest...")
        self.manifest_manager.unload_all_backends()
        self.current_model = None
        
        # --- BEST PRACTICE CLEANUP ---
        import gc
        import torch
        
        # Force Python Garbage Collection
        gc.collect()
        
        # Reset Dynamo/Inductor to clear workers
        try:
            if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "reset"):
                print("[InferenceService] Resetting torch._dynamo...")
                torch._dynamo.reset()
        except Exception as e:
            print(f"[InferenceService] Failed to reset dynamo: {e}")

        # AGGRESSIVE CLEANUP: Kill child workers (compile workers)
        try:
            import psutil
            import os
            import signal
            
            parent = psutil.Process()
            children = parent.children(recursive=True)
            
            if children:
                print(f"[InferenceService] Found {len(children)} child processes. Terminating...")
                for child in children:
                    try:
                        # Skip if it's not a python process or looks critical
                        if child.pid == os.getpid(): continue
                        
                        print(f"[InferenceService] Killing worker/child PID: {child.pid}")
                        child.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Wait for termination
                _, alive = psutil.wait_procs(children, timeout=3)
                for p in alive:
                     print(f"[InferenceService] Force killing PID: {p.pid}")
                     p.kill()
                     
        except Exception as e:
            print(f"[InferenceService] Error killing child processes: {e}")
        
        # Force CUDA Cache Clear
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                start_mem = torch.cuda.memory_allocated()
                print(f"[InferenceService] Unload complete. CUDA Mem: {start_mem}")
        except:
            pass
            
        return True

    def is_running(self) -> bool:
        return self.worker_pool is not None and self.current_model is not None
