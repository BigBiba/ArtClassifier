from fastapi import FastAPI, UploadFile, File, HTTPException

from .schemas import PredictionResponse
from .service import predict_image

import logging

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Image Classification ML Service",
    description="Сервис классификации изображений по классам и подклассам",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть изображением"
        )

    image_bytes = await file.read()

    try:
        result = predict_image(image_bytes)
        return result

    except Exception as error:
        logger.exception("Ошибка при обработке изображения")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке изображения: {str(error)}"
        )