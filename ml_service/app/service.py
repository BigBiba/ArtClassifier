import torch

from .config import DEVICE, CLASS_NAMES, SUBCLASS_NAMES
from .model import get_model
from .preprocessing import preprocess_image


def predict_image(image_bytes: bytes) -> dict:
    model = get_model()

    input_tensor = preprocess_image(image_bytes)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)

    class_logits, subclass_logits = output

    class_probs = torch.softmax(class_logits, dim=1)
    subclass_probs = torch.softmax(subclass_logits, dim=1)

    class_confidence, class_id = torch.max(class_probs, dim=1)
    subclass_confidence, subclass_id = torch.max(subclass_probs, dim=1)

    class_id = int(class_id.item())
    subclass_id = int(subclass_id.item())

    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES.get(class_id, "unknown"),

        "subclass_id": subclass_id,
        "subclass_name": SUBCLASS_NAMES.get(class_id, {}).get(subclass_id, "unknown"),
    }