import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def ensure_dirs(*paths: Path) -> None:
	for path in paths:
		path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def read_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)

