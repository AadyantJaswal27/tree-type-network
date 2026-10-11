import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.tree_network import TreeNetwork


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

config = {
    "name": "xhuge",
    "channels": (128, 256, 256),
    "feature_dim": 512,
    "node_hidden": 2048
}

batch_size = 8
epochs = 20
lr = 1e-3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

os.makedirs(
    "results/cifar10/double_descent",
    exist_ok=True
)

os.makedirs(
    "checkpoints/cifar10/double_descent",
    exist_ok=True
)

print()
print("=" * 60)
print("MODEL:", config["name"])
print("=" * 60)

model = TreeNetwork(
    num_classes=10,
    channels=config["channels"],
    feature_dim=config["feature_dim"],
    node_hidden=config["node_hidden"],
    depth=4
).to(device)

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Device:", device)
print("Parameters:", parameters)
print("Batch size:", batch_size)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=0
)

history = []

best_train = 0
best_test = 0

for epoch in range(epochs):

    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()
        optimizer.step()

        train_loss += (
            loss.item() * y.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        train_correct += (
            predictions == y
        ).sum().item()

        train_total += y.size(0)

    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            predictions = logits.argmax(
                dim=1
            )

            test_correct += (
                predictions == y
            ).sum().item()

            test_total += y.size(0)

    train_accuracy = (
        100 * train_correct / train_total
    )

    test_accuracy = (
        100 * test_correct / test_total
    )

    best_train = max(
        best_train,
        train_accuracy
    )

    best_test = max(
        best_test,
        test_accuracy
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss / train_total,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    })

    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"Train {train_accuracy:.2f}% "
        f"Test {test_accuracy:.2f}%"
    )

result = {
    "model": config["name"],
    "parameters": parameters,
    "config": config,
    "batch_size": batch_size,
    "epochs": epochs,
    "best_train_accuracy": best_train,
    "best_test_accuracy": best_test,
    "final_train_accuracy": history[-1]["train_accuracy"],
    "final_test_accuracy": history[-1]["test_accuracy"],
    "history": history
}

with open(
    "results/cifar10/double_descent/xhuge.json",
    "w"
) as f:
    json.dump(
        result,
        f,
        indent=2
    )

torch.save(
    model.state_dict(),
    "checkpoints/cifar10/double_descent/xhuge.pt"
)

print()
print("=" * 60)
print("XHUGE COMPLETE")
print("=" * 60)
print("Parameters:", parameters)
print("Best Train:", best_train)
print("Best Test:", best_test)
print("Final Train:", history[-1]["train_accuracy"])
print("Final Test:", history[-1]["test_accuracy"])

del model
del optimizer

torch.cuda.empty_cache()

print()
print("CIFAR-10 XHUGE EXPERIMENT COMPLETE.")