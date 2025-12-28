import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, Future
from typing import Any, Dict, Optional, Callable


class SimpleWorkerPool:
    def __init__(self, n_workers: int = 2, max_queue_size: int = 10, default_timeout: float = 30.0):
        self.n_workers = n_workers
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.processing_count = 0
        self.timeout_count = 0
        self._lock = threading.Lock()
        self._running = True
        
        # Start worker threads to process the queue
        for _ in range(n_workers):
            threading.Thread(target=self._worker, daemon=True).start()
        
    def _worker(self):
        """Worker thread that processes requests from the queue"""
        while self._running:
            try:
                # Get request from queue with timeout
                request_item = self.request_queue.get(timeout=1.0)
                if request_item is None:  # Shutdown signal
                    break
                    
                function, timeout, kwargs, result_future = request_item
                
                # Increment processing count
                with self._lock:
                    self.processing_count += 1
                
                try:
                    # Execute the function
                    result = function(**kwargs)
                    result_future.set_result(result)
                except Exception as e:
                    result_future.set_result({"error": str(e), "status": "error"})
                finally:
                    # Decrement processing count
                    with self._lock:
                        self.processing_count -= 1
                    self.request_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception:
                break
        
    def submit_request(self, function: Callable, timeout: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        # Check if queue is full
        if self.request_queue.full():
            return {"error": "Queue is full", "status": "rejected"}
        
        try:
            # Create a future to get the result
            result_future = Future()
            
            # Put request in queue
            request_item = (function, timeout or self.default_timeout, kwargs, result_future)
            self.request_queue.put(request_item, block=False)
            
            # Wait for result
            try:
                result = result_future.result(timeout=timeout or self.default_timeout)
                return result
            except FutureTimeoutError:
                with self._lock:
                    self.timeout_count += 1
                return {"error": "Request timeout", "status": "timeout"}
                
        except queue.Full:
            return {"error": "Queue is full", "status": "rejected"}
        except Exception as e:
            return {"error": f"Failed to submit request: {str(e)}", "status": "error"}
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "workers": self.n_workers,
                "queue_size": self.request_queue.qsize(),
                "max_queue_size": self.max_queue_size,
                "processing": self.processing_count,
                "timeouts": self.timeout_count,
                "default_timeout": self.default_timeout
            }
    
    def shutdown(self):
        self._running = False
        
        # Signal workers to stop by putting None in queue
        for _ in range(self.n_workers):
            try:
                self.request_queue.put(None, block=False)
            except queue.Full:
                pass
        
        self.executor.shutdown(wait=True)