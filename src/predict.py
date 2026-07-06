from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, cast

import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms

from src.config import Config
from src.model import build_model


def load_checkpoint(checkpoint_path: Path, device: torch.device):
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model = build_model(
		num_classes=checkpoint["num_classes"],
		model_name=checkpoint.get("model_name", "resnet18"),
		pretrained=False,
	).to(device)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	return model, checkpoint


def preprocess(image: Image.Image, image_size: int, mean: List[float], std: List[float]) -> torch.Tensor:
	tfm = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)
	tensor = cast(Tensor, tfm(image.convert("RGB")))
	return tensor.unsqueeze(0)


def predict_image(image_path: Path, top_k: int = 3) -> List[Tuple[str, float]]:
	config = Config()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model, checkpoint = load_checkpoint(config.best_model_path, device)

	idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
	image_size = checkpoint.get("image_size", config.image_size)
	mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
	std = checkpoint.get("std", [0.229, 0.224, 0.225])

	image = Image.open(image_path)
	x = preprocess(image, image_size=image_size, mean=mean, std=std).to(device)

	with torch.no_grad():
		logits = model(x)
		probs = torch.softmax(logits, dim=1)
		top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

	results: List[Tuple[str, float]] = []
	for score, idx in zip(top_probs.squeeze(0).cpu().numpy(), top_indices.squeeze(0).cpu().numpy()):
		results.append((idx_to_class[int(idx)], float(score)))
	return results


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Predict electronic component class from one image")
	parser.add_argument("image_path", type=Path, help="Path to image")
	parser.add_argument("--top-k", type=int, default=3, help="Return top-k predictions")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	predictions = predict_image(args.image_path, top_k=args.top_k)
	for label, score in predictions:
		print(f"{label}: {score:.4f}")

