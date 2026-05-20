import torch

from .config import DEVICE, WEIGHTS_PATH, NUM_CLASSES, NUM_SUBCLASSES
from .efficient_net import HierarchicalEfficientNet


def load_model():
    model = HierarchicalEfficientNet(
        num_classes=NUM_CLASSES,
        num_subclasses_per_class=NUM_SUBCLASSES,
    )

    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


def get_model():
    return model