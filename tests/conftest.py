
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import logging
from httpx import AsyncClient, ASGITransport
from moondream_station.core.rest_server import RestServer
from moondream_station.core.config import ConfigManager

# Disable logging during tests
logging.getLogger("MoondreamServer").setLevel(logging.CRITICAL)

@pytest.fixture
def mock_config():
    """Mock ConfigManager"""
    config = MagicMock(spec=ConfigManager)
    config.get.return_value = "moondream-2"
    return config

@pytest.fixture
def mock_manifest():
    """Mock ManifestManager"""
    manifest = MagicMock()
    manifest.get_models.return_value = {
        "moondream-2": MagicMock(name="moondream-2", backend="moondream_backend"),
        "test-model": MagicMock(name="test-model", backend="test_backend")
    }
    manifest.get_backend_for_model.return_value = {"name": "test_backend"}
    return manifest

@pytest.fixture
def mock_inference_service():
    """Mock InferenceService to avoid loading real models"""
    with patch("moondream_station.core.rest_server.InferenceService") as mock:
        service_instance = mock.return_value
        service_instance.start.return_value = True
        service_instance.is_running.return_value = True
        service_instance.execute_function = AsyncMock(return_value={"result": ["base64_result_image"]})
        service_instance.unload_model = MagicMock()
        yield service_instance

@pytest.fixture
def api_server(mock_config, mock_manifest, mock_inference_service):
    """Create a RestServer instance with mocked dependencies"""
    # Patch internals that might side-effect
    with patch("moondream_station.core.rest_server.hw_monitor"), \
         patch("moondream_station.core.rest_server.model_memory_tracker"), \
         patch("moondream_station.core.rest_server.sdxl_backend_new", None): 
        
        server = RestServer(mock_config, mock_manifest)
        # Ensure the app thinks dependency is running
        server.inference_service.is_running.return_value = True
        return server

@pytest.fixture
async def api_client(api_server):
    """AsyncClient for the RestServer FastAPI app"""
    async with AsyncClient(
        transport=ASGITransport(app=api_server.app), 
        base_url="http://test"
    ) as client:
        yield client
