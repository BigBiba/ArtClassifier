import torch.nn as nn
import torch
import torchvision.models as models


class HierarchicalEfficientNet(nn.Module):
    def __init__(self, num_classes: int, num_subclasses_per_class: list[int], dropout: float = 0.3):
        super().__init__()

        self.num_classes = num_classes
        self.num_subclasses_per_class = num_subclasses_per_class

        self.backbone = models.efficientnet_b3(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.class_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        self.subclass_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_subclasses)
            )
            for num_subclasses in num_subclasses_per_class
        ])

    def forward(self, x):
        features = self.backbone(x)

        class_logits = self.class_head(features)

        subclass_logits = [
            head(features) for head in self.subclass_heads
        ]

        return class_logits, subclass_logits
