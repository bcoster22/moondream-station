import time
from typing import Optional
from .config import SERVICE_PORT, SERVICE_HOST
from .rest_server import RestServer


class ServiceManager:
    def __init__(self, config, manifest_manager=None, session_state=None, analytics=None):
        self.config = config
        self.manifest_manager = manifest_manager
        self.session_state = session_state
        self.analytics = analytics
        self.rest_server = None

    def start(self, model_name: str, port: int = SERVICE_PORT) -> bool:
        """Start the REST server with the specified model"""
        if self.is_running():
            return False

        if not self.manifest_manager:
            return False

        try:
            backend = self.manifest_manager.get_backend_for_model(model_name)
            if not backend:
                return False

            self.rest_server = RestServer(self.config, self.manifest_manager, self.session_state, self.analytics)
            host = self.config.get("service_host", SERVICE_HOST)

            if self.rest_server.start(host, port):
                self.config.set("service_port", port)
                return True
            else:
                self.rest_server = None
                return False

        except Exception as e:
            import sys
            sys.stderr.write(f"DEBUG: EXCEPTION in ServiceManager.start: {e}\n")
            import traceback
            traceback.print_exc()
            self.rest_server = None
            return False

    def stop(self) -> bool:
        """Stop the REST server"""
        if self.rest_server:
            result = self.rest_server.stop()
            self.rest_server = None
            return result
        return True

    def is_running(self) -> bool:
        """Check if service is running"""
        return self.rest_server and self.rest_server.is_running()

    def restart(self, model_name: str, port: Optional[int] = None) -> bool:
        """Restart the service"""
        self.stop()
        time.sleep(1)
        return self.start(
            model_name, port or self.config.get("service_port", SERVICE_PORT)
        )

    def get_status(self) -> dict:
        """Get service status information"""
        if not self.is_running():
            return {"status": "stopped", "pid": None, "port": None}

        return {
            "status": "running",
            "pid": "internal",
            "port": self.config.get("service_port", SERVICE_PORT),
        }
