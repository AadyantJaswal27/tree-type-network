import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from models.tree_network import TreeNetwork
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs("results/cifar10", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=transform
)

model = TreeNetwork().to(device)

checkpoint = torch.load(
    "checkpoints/cifar10/overfit_final.pt",
    map_location=device
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

image, label = dataset[0]
x = image.unsqueeze(0).to(device)
x.requires_grad_(True)

with torch.no_grad():
    prediction = model(x).argmax(dim=1).item()

nodes = [0, 1, 2, 3, 4, 5, 6]

fig, axes = plt.subplots(
    2, 4,
    figsize=(12, 6)
)

image_np = image.permute(1, 2, 0).cpu().numpy()

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2470, 0.2435, 0.2616])

image_np = image_np * std + mean
image_np = np.clip(image_np, 0, 1)

axes[0, 0].imshow(image_np)
axes[0, 0].set_title(
    f"Input\nTrue: {dataset.classes[label]}\nPred: {dataset.classes[prediction]}"
)
axes[0, 0].axis("off")

for i, node_idx in enumerate(nodes):
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()

    features = model.features(x)
    features = model.projection(features)

    node_output = model.nodes[node_idx](features)

    node_output[0].backward(retain_graph=True)

    gradient = x.grad[0].detach().abs()
    saliency = gradient.max(dim=0)[0].cpu().numpy()

    saliency = saliency / (saliency.max() + 1e-8)

    ax = axes.flat[i + 1]
    ax.imshow(saliency, cmap="hot")
    ax.set_title(f"Tree Node {node_idx}")
    ax.axis("off")

plt.tight_layout()

path = "results/cifar10/receptive_fields.png"
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", path)