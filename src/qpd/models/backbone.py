import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

EMBED_DIM = 512


def build_backbone(pretrained: bool = True) -> nn.Module:
    """ResNet-18 with the final FC layer removed; outputs 512-d embeddings."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Identity()
    return model
