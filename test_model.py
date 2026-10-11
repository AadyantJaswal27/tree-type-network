import torch
from models.tree_network import TreeNetwork


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TreeNetwork().to(device)

x = torch.randn(8, 3, 32, 32).to(device)

with torch.no_grad():
    y = model(x)

parameters = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Device:", device)
print("Input:", x.shape)
print("Output:", y.shape)
print("Parameters:", parameters)