from .predictor import get_predictor
from .preprocessing import preprocess_image


def predict_image(image_bytes: bytes) -> dict:
    input_tensor = preprocess_image(image_bytes)

    predictor = get_predictor()
    return predictor.predict(input_tensor)