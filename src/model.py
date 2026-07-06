from typing import Tuple

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, model_name: str = "resnet18", pretrained: bool = True) -> nn.Module:
	if model_name == "resnet18":
		weights = models.ResNet18_Weights.DEFAULT if pretrained else None
		model = models.resnet18(weights=weights)
		in_features = model.fc.in_features
		model.fc = nn.Linear(in_features, num_classes)
		return model

	if model_name == "efficientnet_b0":
		weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
		model = models.efficientnet_b0(weights=weights)
		last_layer = model.classifier[-1]
		if not isinstance(last_layer, nn.Linear):
			raise TypeError("Unexpected EfficientNet classifier head type")
		in_features = last_layer.in_features
		model.classifier[-1] = nn.Linear(in_features, num_classes)
		return model

	raise ValueError(f"Unsupported model_name: {model_name}")


def mean_std() -> Tuple[list[float], list[float]]:
	return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

