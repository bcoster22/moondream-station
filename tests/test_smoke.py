import pytest

@pytest.mark.asyncio
async def test_backend_smoke():
    assert 1 + 1 == 2
