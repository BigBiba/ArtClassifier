import torch.nn as nn
import torch
import torchvision.models as models


class HierarchicalResNet(nn.Module):
    def __init__(self, num_classes: int, num_subclasses_per_class: list[int]):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.class_head = nn.Linear(2048, num_classes)
        self.subclass_heads = nn.ModuleList([
            nn.Linear(2048, num_sub) for num_sub in num_subclasses_per_class
        ])

    def forward(self, x):
        features = self.backbone(x)

        class_logits = self.class_head(features)
        subclass_logits = []

        for head in self.subclass_heads:
            subclass_logits.append(head(features))
        subclass_logits = torch.cat(subclass_logits, dim=1)

        return class_logits, subclass_logits
