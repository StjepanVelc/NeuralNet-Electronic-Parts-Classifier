from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from typing import cast

image_path = "data/raw/Bypass-capacitor/Bypass-capacitor001.jpg"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open(image_path).convert("RGB")
tensor = cast(torch.Tensor, transform(img))

print("=== ORIGINAL IMAGE TENSOR ===")
print("Shape:", tensor.shape)
print("Dtype:", tensor.dtype)
print("Min:", tensor.min())
print("Max:", tensor.max())

# Jedan 3x3 patch iz R kanala
r_patch = tensor[0, 0:3, 0:3]

print("\n=== FIRST 3x3 PATCH, R CHANNEL ===")
print(r_patch)

print("\nMax iz tog patcha:")
print(r_patch.max())

# MaxPool očekuje [batch, channel, height, width]
x = tensor.unsqueeze(0)

pool = nn.MaxPool2d(kernel_size=3, stride=3)
pooled = pool(x)

print("\n=== AFTER MAX POOLING 3x3 ===")
print("Shape:", pooled.shape)
print("First pooled value, R channel:", pooled[0, 0, 0, 0])

pooled_image_tensor = pooled.squeeze(0)

pooled_img = to_pil_image(pooled_image_tensor)
pooled_img.save("maxpool_3x3_result.jpg")

print("\nSaved image: maxpool_3x3_result.jpg")