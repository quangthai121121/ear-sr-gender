"""
Model factory for the 6 backbones in Section 3.4 of the paper draft.
All use ImageNet-pretrained weights from torchvision, with the final
classification layer replaced by a 2-way (male/female) head.
"""
from __future__ import annotations
import torch.nn as nn
import torchvision.models as tvm

NUM_CLASSES = 2

MODEL_NAMES = [
    "vgg19", "mobilenet_v2", "resnet50",
    "efficientnet_b0", "swin_t", "maxvit_t",
]


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()

    if name == "vgg19":
        m = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
        return m

    if name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
        return m

    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, NUM_CLASSES)
        return m

    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
        return m

    if name == "swin_t":
        m = tvm.swin_t(weights=tvm.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = m.head.in_features
        m.head = nn.Linear(in_f, NUM_CLASSES)
        return m

    if name == "maxvit_t":
        m = tvm.maxvit_t(weights=tvm.MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None)
        # maxvit's classifier is a Sequential ending in Linear
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
        return m

    raise ValueError(f"Unknown model name: {name}. Choose from {MODEL_NAMES}")
