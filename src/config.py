from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
	project_root: Path = Path(__file__).resolve().parents[1]

	raw_data_dir: Path = project_root / "data" / "raw"
	train_dir: Path = project_root / "data" / "train"
	val_dir: Path = project_root / "data" / "val"
	test_dir: Path = project_root / "data" / "test"
	processed_dir: Path = project_root / "data" / "processed"

	models_dir: Path = project_root / "models"
	reports_dir: Path = project_root / "reports"
	figures_dir: Path = reports_dir / "figures"

	best_model_path: Path = models_dir / "best_model.pt"
	metrics_path: Path = reports_dir / "metrics.json"
	class_index_path: Path = reports_dir / "class_to_idx.json"

	image_size: int = 224
	batch_size: int = 32
	learning_rate: float = 1e-3
	weight_decay: float = 1e-4
	epochs: int = 12
	num_workers: int = 0
	patience: int = 4
	seed: int = 42

	train_ratio: float = 0.8
	val_ratio: float = 0.1
	test_ratio: float = 0.1

