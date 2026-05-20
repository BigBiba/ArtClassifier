import torch

from .config import DEVICE, CLASS_NAMES, SUBCLASS_NAMES
from .model import get_model


class HierarchicalPredictor:
    def __init__(self, model, device, class_names: dict, subclass_names: dict):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.subclass_names = subclass_names

    def predict(self, input_tensor: torch.Tensor) -> dict:
        input_tensor = input_tensor.to(self.device)

        with torch.inference_mode():
            class_logits, subclass_logits_list = self.model(input_tensor)

            class_probs = torch.softmax(class_logits, dim=1)
            _, class_id_tensor = torch.max(class_probs, dim=1)
            class_id = int(class_id_tensor.item())

            subclass_logits = subclass_logits_list[class_id]
            subclass_probs = torch.softmax(subclass_logits, dim=1)
            _, subclass_id_tensor = torch.max(subclass_probs, dim=1)
            subclass_id = int(subclass_id_tensor.item())

        return {
            "class_id": class_id,
            "class_name": self.class_names.get(class_id, "unknown"),

            "subclass_id": subclass_id,
            "subclass_name": self.subclass_names.get(class_id, {}).get(subclass_id, "unknown"),
        }


_predictor = HierarchicalPredictor(
    model=get_model(),
    device=DEVICE,
    class_names=CLASS_NAMES,
    subclass_names=SUBCLASS_NAMES,
)


def get_predictor() -> HierarchicalPredictor:
    return _predictor