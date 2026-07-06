from __future__ import annotations

import argparse
from typing import Dict, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from tqdm import tqdm
from torchvision.datasets import ImageFolder

from src.config import Config
from src.dataset import build_dataloaders
from src.model import build_model, mean_std
from src.utils import ensure_dirs, seed_everything, write_json


def evaluate_epoch(model: nn.Module, loader, device: torch.device, loss_fn: nn.Module) -> Tuple[float, float, float]:
	model.eval()
	losses = []
	all_preds = []
	all_targets = []

	with torch.no_grad():
		for images, targets in loader:
			images, targets = images.to(device), targets.to(device)
			logits = model(images)
			loss = loss_fn(logits, targets)
			losses.append(loss.item())

			preds = torch.argmax(logits, dim=1)
			all_preds.extend(preds.cpu().numpy())
			all_targets.extend(targets.cpu().numpy())

	avg_loss = float(np.mean(losses)) if losses else 0.0
	acc = float(accuracy_score(all_targets, all_preds)) if all_targets else 0.0
	macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0)) if all_targets else 0.0
	return avg_loss, acc, macro_f1


def save_history_plots(history: list[dict], config: Config) -> None:
	epochs = [entry["epoch"] for entry in history]
	train_loss = [entry["train_loss"] for entry in history]
	val_loss = [entry["val_loss"] for entry in history]
	train_f1 = [entry["train_macro_f1"] for entry in history]
	val_f1 = [entry["val_macro_f1"] for entry in history]

	plt.figure(figsize=(9, 5))
	plt.plot(epochs, train_loss, marker="o", label="Train Loss")
	plt.plot(epochs, val_loss, marker="o", label="Val Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training vs Validation Loss")
	plt.grid(alpha=0.25)
	plt.legend()
	plt.tight_layout()
	plt.savefig(config.figures_dir / "loss_curve.png", dpi=180)
	plt.close()

	plt.figure(figsize=(9, 5))
	plt.plot(epochs, train_f1, marker="o", label="Train Macro-F1")
	plt.plot(epochs, val_f1, marker="o", label="Val Macro-F1")
	plt.xlabel("Epoch")
	plt.ylabel("Macro-F1")
	plt.title("Training vs Validation Macro-F1")
	plt.grid(alpha=0.25)
	plt.legend()
	plt.tight_layout()
	plt.savefig(config.figures_dir / "macro_f1_curve.png", dpi=180)
	plt.close()


def train(args: argparse.Namespace) -> None:
	config = Config()
	seed_everything(config.seed)
	ensure_dirs(config.models_dir, config.reports_dir, config.figures_dir)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(config)
	num_classes = len(class_to_idx)
	model = build_model(num_classes=num_classes, model_name=args.model, pretrained=True).to(device)

	train_dataset = cast(ImageFolder, train_loader.dataset)
	targets = np.array(train_dataset.targets)
	class_counts = np.bincount(targets)
	class_weights = 1.0 / np.maximum(class_counts, 1)
	class_weights = class_weights / class_weights.mean()
	class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

	loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
	optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

	history = []
	best_val_f1 = -1.0
	best_epoch = -1
	epochs_without_improvement = 0

	for epoch in range(1, config.epochs + 1):
		model.train()
		train_losses = []
		train_preds = []
		train_targets = []

		progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
		for images, targets in progress:
			images, targets = images.to(device), targets.to(device)

			optimizer.zero_grad()
			logits = model(images)
			loss = loss_fn(logits, targets)
			loss.backward()
			optimizer.step()

			train_losses.append(loss.item())
			preds = torch.argmax(logits, dim=1)
			train_preds.extend(preds.detach().cpu().numpy())
			train_targets.extend(targets.detach().cpu().numpy())
			progress.set_postfix(loss=float(np.mean(train_losses)))

		train_loss = float(np.mean(train_losses)) if train_losses else 0.0
		train_acc = float(accuracy_score(train_targets, train_preds)) if train_targets else 0.0
		train_f1 = float(f1_score(train_targets, train_preds, average="macro", zero_division=0)) if train_targets else 0.0

		val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, device, loss_fn)

		history.append(
			{
				"epoch": epoch,
				"train_loss": train_loss,
				"train_acc": train_acc,
				"train_macro_f1": train_f1,
				"val_loss": val_loss,
				"val_acc": val_acc,
				"val_macro_f1": val_f1,
			}
		)

		if val_f1 > best_val_f1:
			best_val_f1 = val_f1
			best_epoch = epoch
			epochs_without_improvement = 0
			mean, std = mean_std()
			torch.save(
				{
					"model_name": args.model,
					"num_classes": num_classes,
					"class_to_idx": class_to_idx,
					"image_size": config.image_size,
					"mean": mean,
					"std": std,
					"state_dict": model.state_dict(),
				},
				config.best_model_path,
			)
		else:
			epochs_without_improvement += 1

		print(
			f"Epoch {epoch}: train_loss={train_loss:.4f}, train_f1={train_f1:.4f}, "
			f"val_loss={val_loss:.4f}, val_f1={val_f1:.4f}"
		)

		if epochs_without_improvement >= config.patience:
			print(f"Early stopping at epoch {epoch}.")
			break

	checkpoint = torch.load(config.best_model_path, map_location=device)
	model.load_state_dict(checkpoint["state_dict"])
	test_loss, test_acc, test_f1 = evaluate_epoch(model, test_loader, device, loss_fn)

	metrics: Dict[str, object] = {
		"model_name": checkpoint["model_name"],
		"best_epoch": best_epoch,
		"best_val_macro_f1": best_val_f1,
		"test_loss": test_loss,
		"test_acc": test_acc,
		"test_macro_f1": test_f1,
		"history": history,
	}

	write_json(config.metrics_path, metrics)
	write_json(config.class_index_path, class_to_idx)
	if history:
		save_history_plots(history, config)

	print(f"Best model saved to: {config.best_model_path}")
	print(f"Metrics saved to: {config.metrics_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train CNN classifier for electronic part images")
	parser.add_argument(
		"--model",
		type=str,
		default="resnet18",
		choices=["resnet18", "efficientnet_b0"],
		help="Backbone architecture",
	)
	return parser.parse_args()


if __name__ == "__main__":
	train(parse_args())

