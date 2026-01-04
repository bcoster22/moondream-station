
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_check(api_client: AsyncClient):
    """Verify the health endpoint returns OK"""
    resp = await api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "server": "moondream-station"}

@pytest.mark.asyncio
async def test_metrics_endpoint(api_client: AsyncClient):
    """Verify metrics endpoint returns valid structure"""
    resp = await api_client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "cpu" in data
    assert "memory" in data
    assert "loaded_models" in data

@pytest.mark.asyncio
async def test_generate_image(api_client: AsyncClient):
    """Verify basic image generation flow (mocked)"""
    # This hits /v1/<something>, captured by dynamic_route -> inference_service.execute_function
    payload = {
        "model": "test-model",
        "prompt": "Test Prompt",
        "steps": 1
    }
    resp = await api_client.post("/v1/images/generate", json=payload)
    
    assert resp.status_code == 200
    data = resp.json()
    # The conftest mock returns {"result": ["base64_result_image"]}
    assert "result" in data
    assert data["result"][0] == "base64_result_image"
