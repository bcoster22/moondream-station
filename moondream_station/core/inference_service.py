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

    def start(self, model_id: str):
        n_workers = int(self.config.get("inference_workers", N_WORKERS))
        max_queue_size = int(
            self.config.get("inference_max_queue_size", MAX_QUEUE_SIZE)
        )
        timeout = float(self.config.get("inference_timeout", TIMOUT))

        if self.worker_pool:
            self.worker_pool.shutdown()

        self.manifest_manager.clear_worker_backends()

        self.current_model = model_id
        self.worker_backends = self.manifest_manager.get_worker_backends(
            model_id, n_workers
        )

        if not self.worker_backends:
            return False

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

    def unload_model(self):
        if self.worker_pool:
            self.worker_pool.shutdown()
            self.worker_pool = None
        
        self.manifest_manager.unload_all_backends()
        self.current_model = None
        return True

    def is_running(self) -> bool:
        return self.worker_pool is not None and self.current_model is not None
