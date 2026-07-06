from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config
from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import ensure_dirs, write_json


def evaluate() -> None:
	config = Config()
	ensure_dirs(config.figures_dir)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint = torch.load(config.best_model_path, map_location=device)

	_, _, test_loader, class_to_idx = build_dataloaders(config)

	idx_to_class = {v: k for k, v in class_to_idx.items()}
	model = build_model(
		num_classes=checkpoint["num_classes"],
		model_name=checkpoint.get("model_name", "resnet18"),
		pretrained=False,
	).to(device)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()

	y_true = []
	y_pred = []

	with torch.no_grad():
		for images, targets in test_loader:
			images = images.to(device)
			logits = model(images)
			preds = torch.argmax(logits, dim=1).cpu().numpy()

			y_pred.extend(preds.tolist())
			y_true.extend(targets.numpy().tolist())

	labels = sorted(idx_to_class.keys())
	class_names = [idx_to_class[i] for i in labels]

	report = classification_report(
		y_true,
		y_pred,
		labels=labels,
		target_names=class_names,
		output_dict=True,
		zero_division=0,
	)
	write_json(config.reports_dir / "classification_report.json", report)

	cm = confusion_matrix(y_true, y_pred, labels=labels)
	plt.figure(figsize=(16, 12))
	sns.heatmap(cm, cmap="Blues", cbar=True)
	plt.title("Confusion Matrix")
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.tight_layout()
	plt.savefig(config.figures_dir / "confusion_matrix.png", dpi=200)
	plt.close()

	print("Saved reports/classification_report.json")
	print("Saved reports/figures/confusion_matrix.png")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate best model on test split")
	return parser.parse_args()


if __name__ == "__main__":
	_ = parse_args()
	evaluate()

