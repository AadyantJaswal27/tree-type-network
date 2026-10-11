import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minimized_tree_network import MinimizedTreeNetwork


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

batch_size = 128
epochs = 50
lr = 1e-3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
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

model = MinimizedTreeNetwork().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=0
)

os.makedirs("results/cifar10", exist_ok=True)
os.makedirs("checkpoints/cifar10", exist_ok=True)

history = []

best_test = 0

for epoch in range(epochs):
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    model.eval()

    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            test_correct += (
                predictions == labels
            ).sum().item()

            test_total += labels.size(0)

    train_loss /= train_total
    test_loss /= test_total

    train_accuracy = 100 * train_correct / train_total
    test_accuracy = 100 * test_correct / test_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Test Acc: {test_accuracy:.2f}%"
    )

    if test_accuracy > best_test:
        best_test = test_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "test_accuracy": test_accuracy
            },
            "checkpoints/cifar10/minimized_best.pt"
        )

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "epoch": epochs
    },
    "checkpoints/cifar10/minimized_final.pt"
)

result = {
    "model": "minimized_9_nodes",
    "parameters": sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ),
    "tree_nodes": len(model.nodes),
    "history": history
}

with open(
    "results/cifar10/minimized_9_nodes.json",
    "w"
) as f:
    json.dump(result, f, indent=2)

print()
print("TRAINING COMPLETE")
print("Parameters:", result["parameters"])
print("Tree nodes:", result["tree_nodes"])
print("Best test accuracy:", best_test)