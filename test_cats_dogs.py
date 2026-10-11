import torch
from models.cats_dogs_tree import CatsDogsTree


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CatsDogsTree().to(device)

x = torch.randn(8, 128, 8, 8).to(device)

with torch.no_grad():
    y = model(x)

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Device:", device)
print("Input:", x.shape)
print("Output:", y.shape)
print("Tree nodes: 1")
print("Parameters:", parameters)
