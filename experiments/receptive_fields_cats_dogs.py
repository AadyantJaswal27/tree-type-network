import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.cats_dogs_tree import CatsDogsTree


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data = torch.load(
    os.path.expandvars(
        r"%USERPROFILE%\Downloads\Catsdogs.pt"
    ),
    map_location="cpu",
    weights_only=False
)

features = data["ftr"].float()
labels = data["ytr"]
images = data["itr"]

model = CatsDogsTree(
    hidden_dim=2048
).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/cats_dogs/overfit_large_final.pt",
        map_location=device,
        weights_only=True
    )
)

model.eval()

idx = 0

x = features[idx:idx + 1].to(device)
x.requires_grad_(True)

model.zero_grad()

output = model(x)

output[0].backward()

gradient = x.grad[0].abs()

saliency = gradient.mean(
    dim=0,
    keepdim=True
).unsqueeze(0)

saliency = F.interpolate(
    saliency,
    size=(64, 64),
    mode="bilinear",
    align_corners=False
)

saliency = saliency.squeeze().cpu().numpy()

saliency = saliency / (
    saliency.max() + 1e-8
)

image = images[idx].permute(
    1, 2, 0
).numpy()

image = image.astype(np.float32) / 255.0

prediction = torch.sigmoid(
    output[0]
).item()

predicted_label = int(
    prediction >= 0.5
)

true_label = labels[idx].item()

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 5)
)

axes[0].imshow(image)

axes[0].set_title(
    f"Original Image\n"
    f"True: {true_label}  Pred: {predicted_label}"
)

axes[0].axis("off")

axes[1].imshow(
    image
)

axes[1].imshow(
    saliency,
    cmap="hot",
    alpha=0.55
)

axes[1].set_title(
    "Node Spatial Sensitivity"
)

axes[1].axis("off")

plt.tight_layout()

os.makedirs(
    "results/cats_dogs",
    exist_ok=True
)

path = (
    "results/cats_dogs/"
    "receptive_fields.png"
)

plt.savefig(
    path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved:", path)
print("True label:", true_label)
print("Predicted label:", predicted_label)
print("Prediction score:", prediction)