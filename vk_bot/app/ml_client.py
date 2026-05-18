import httpx
from .schemas import PredictionResult

class MLClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.client.close()

    def predict(self, image_bytes: bytes) -> PredictionResult:
        files = {
            "file": ("image.jpg", image_bytes, "image/jpeg")
        }

        response = self.client.post(
            f"{self.base_url}/predict",
            files=files
        )

        response.raise_for_status()
        data = response.json()

        return PredictionResult(**data)