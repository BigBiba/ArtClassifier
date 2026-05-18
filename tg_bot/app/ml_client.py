import httpx
from .schemas import PredictionResult

class MLClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()

    async def predict(self, image_bytes: bytes) -> PredictionResult:
        files = {
            "file": ("image.jpg", image_bytes, "image/jpeg")
        }

        response = await self.client.post(
            f"{self.base_url}/predict",
            files=files
        )

        response.raise_for_status()
        data = response.json()

        return PredictionResult(**data)