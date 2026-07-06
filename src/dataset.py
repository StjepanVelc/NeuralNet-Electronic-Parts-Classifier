import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from src.config import Config
from src.utils import ensure_dirs


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _class_image_paths(raw_data_dir: Path) -> Dict[str, List[Path]]:
	class_to_paths: Dict[str, List[Path]] = {}
	for class_dir in sorted(raw_data_dir.iterdir()):
		if not class_dir.is_dir():
			continue
		image_paths = [
			p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
		]
		if image_paths:
			class_to_paths[class_dir.name] = sorted(image_paths)
	return class_to_paths


def split_raw_dataset(config: Config) -> None:
	ensure_dirs(config.train_dir, config.val_dir, config.test_dir)
	class_to_paths = _class_image_paths(config.raw_data_dir)
	if not class_to_paths:
		raise FileNotFoundError(f"No class images found in {config.raw_data_dir}")

	rng = np.random.RandomState(config.seed)
	split_roots = {
		"train": config.train_dir,
		"val": config.val_dir,
		"test": config.test_dir,
	}

	for class_name, image_paths in class_to_paths.items():
		if len(image_paths) < 3:
			continue

		train_paths, temp_paths = train_test_split(
			image_paths,
			test_size=(1.0 - config.train_ratio),
			random_state=rng.randint(0, 1_000_000),
			shuffle=True,
		)

		val_fraction_within_temp = config.val_ratio / (config.val_ratio + config.test_ratio)
		val_paths, test_paths = train_test_split(
			temp_paths,
			test_size=(1.0 - val_fraction_within_temp),
			random_state=rng.randint(0, 1_000_000),
			shuffle=True,
		)

		for split_name, split_paths in {
			"train": train_paths,
			"val": val_paths,
			"test": test_paths,
		}.items():
			split_root = split_roots[split_name] / class_name
			split_root.mkdir(parents=True, exist_ok=True)
			for source_path in split_paths:
				destination = split_root / source_path.name
				if not destination.exists():
					shutil.copy2(source_path, destination)


def has_existing_split(config: Config) -> bool:
	return (
		config.train_dir.exists()
		and config.val_dir.exists()
		and config.test_dir.exists()
		and any(config.train_dir.rglob("*.*"))
		and any(config.val_dir.rglob("*.*"))
		and any(config.test_dir.rglob("*.*"))
	)


def get_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	train_tfms = transforms.Compose(
		[
			transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomVerticalFlip(p=0.1),
			transforms.RandomRotation(degrees=20),
			transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=6),
			transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
			transforms.RandomPerspective(distortion_scale=0.18, p=0.25),
			transforms.ToTensor(),
			transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
			transforms.Normalize(mean=mean, std=std),
		]
	)

	eval_tfms = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)

	return train_tfms, eval_tfms


def build_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
	if not has_existing_split(config):
		split_raw_dataset(config)

	train_tfms, eval_tfms = get_transforms(config.image_size)

	train_ds: ImageFolder = datasets.ImageFolder(config.train_dir, transform=train_tfms)
	val_ds: ImageFolder = datasets.ImageFolder(config.val_dir, transform=eval_tfms)
	test_ds: ImageFolder = datasets.ImageFolder(config.test_dir, transform=eval_tfms)

	targets: NDArray[np.int64] = np.asarray(train_ds.targets, dtype=np.int64)
	class_sample_count = np.bincount(targets)
	class_weights = 1.0 / np.maximum(class_sample_count, 1)
	sample_weights = class_weights[targets]

	sampler = WeightedRandomSampler(
		weights=sample_weights.tolist(),
		num_samples=len(sample_weights),
		replacement=True,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=config.batch_size,
		sampler=sampler,
		num_workers=config.num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=config.num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	test_loader = DataLoader(
		test_ds,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=config.num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	return train_loader, val_loader, test_loader, train_ds.class_to_idx

