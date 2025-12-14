
import sys
import time
import logging
import os
import signal

# Ensure we can import moondream_station
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from moondream_station.core.config import ConfigManager
from moondream_station.core.manifest import ManifestManager
from moondream_station.core.analytics import Analytics
from moondream_station.core.service import ServiceManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MoondreamServer")

def main():
    try:
        config = ConfigManager()
        
        # Manifest path
        manifest_path = os.path.join(current_dir, "local_manifest.json")
        if not os.path.exists(manifest_path):
             logger.error(f"Manifest not found at {manifest_path}")
             sys.exit(1)
             
        logger.info(f"Loading manifest from {manifest_path}")
        manifest_manager = ManifestManager(config)
        manifest_manager.load_manifest(manifest_path)
        
        analytics = Analytics(config, manifest_manager)
        service = ServiceManager(config, manifest_manager, None, analytics)
        
        # Determine model to load: from config or default
        model_name = config.get("current_model")
        if not model_name:
             logger.info("No current model in config. Checking default...")
             model_name = manifest_manager.get_available_default_model()
        
        # Fallback if still None
        if not model_name:
             logger.info("No default model found. Using z-image-6b.")
             model_name = "z-image-6b" # Default fallback
             
        logger.info(f"Starting service with model: {model_name}")
        
        if service.start(model_name):
            port = config.get("service_port", 2020)
            logger.info(f"Service started successfully on port {port}")
            
            # Handle graceful shutdown
            def signal_handler(sig, frame):
                logger.info("Stopping service...")
                service.stop()
                sys.exit(0)
                
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            while True:
                time.sleep(1)
        else:
            logger.error("Failed to start service.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
