import torch

from .config import DEVICE, WEIGHTS_PATH, NUM_CLASSES, NUM_SUBCLASSES
from .neural_network import HierarchicalResNet


def load_model():
    model = HierarchicalResNet(
        num_classes=NUM_CLASSES,
        num_subclasses_per_class=NUM_SUBCLASSES,
    )

    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


def get_model():
    return model