from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

# Workaround for Python 3.14 + protobuf binary compatibility in Streamlit.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import streamlit as st
import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.model import build_model


@st.cache_resource
def load_model_and_metadata():
	config = Config()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint = torch.load(config.best_model_path, map_location=device)

	model = build_model(
		num_classes=checkpoint["num_classes"],
		model_name=checkpoint.get("model_name", "resnet18"),
		pretrained=False,
	).to(device)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()

	idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
	image_size = checkpoint.get("image_size", config.image_size)
	mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
	std = checkpoint.get("std", [0.229, 0.224, 0.225])

	return model, idx_to_class, image_size, mean, std, device


def preprocess(image: Image.Image, image_size: int, mean, std) -> Tensor:
	tfm = transforms.Compose(
		[
			transforms.Resize((image_size, image_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)
	tensor = cast(Tensor, tfm(image.convert("RGB")))
	return tensor.unsqueeze(0)


def main() -> None:
	st.set_page_config(page_title="Electronic Parts Classifier", page_icon="🔌", layout="wide")
	st.title("Electronic Parts CNN Classifier")
	st.write("Uploadaj sliku komponente i dobit ćeš top-3 predikcije modela.")

	config = Config()
	if not config.best_model_path.exists():
		st.error("Model ne postoji. Prvo pokreni trening: python -m src.train")
		st.stop()

	model, idx_to_class, image_size, mean, std, device = load_model_and_metadata()

	uploaded = st.file_uploader("Odaberi sliku", type=["jpg", "jpeg", "png", "bmp", "webp"])
	if not uploaded:
		return

	image = Image.open(uploaded)
	col1, col2 = st.columns([1, 1])
	with col1:
		st.image(image, caption="Ulazna slika", use_container_width=True)

	x = preprocess(image, image_size=image_size, mean=mean, std=std).to(device)
	with torch.no_grad():
		logits = model(x)
		probs = torch.softmax(logits, dim=1)
		top_probs, top_idxs = torch.topk(probs, k=3, dim=1)

	with col2:
		st.subheader("Top-3 predikcije")
		for score, idx in zip(top_probs.squeeze(0).cpu().numpy(), top_idxs.squeeze(0).cpu().numpy()):
			st.write(f"{idx_to_class[int(idx)]}: {float(score) * 100:.2f}%")


if __name__ == "__main__":
	main()

