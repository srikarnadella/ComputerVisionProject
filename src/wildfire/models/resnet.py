from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = False, dropout: float = 0.2) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)
