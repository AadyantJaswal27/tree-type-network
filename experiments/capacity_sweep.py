import os
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.tree_network import TreeNetwork


SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


CONFIGS = {
    "tiny": {
        "channels": (8, 16, 16),
        "feature_dim": 32,
        "node_hidden": 16
    },
    "small": {
        "channels": (16, 32, 32),
        "feature_dim": 64,
        "node_hidden": 32
    },
    "baseline": {
        "channels": (32, 64, 64),
        "feature_dim": 128,
        "node_hidden": 64
    },
    "large": {
        "channels": (64, 128, 128),
        "feature_dim": 256,
        "node_hidden": 128
    }
}


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)


trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        100.0 * correct / total
    )


os.makedirs(
    "results/cifar10/capacity",
    exist_ok=True
)

for name, config in CONFIGS.items():

    print()
    print("=" * 60)
    print("MODEL:", name)
    print("=" * 60)

    model = TreeNetwork(
        channels=config["channels"],
        feature_dim=config["feature_dim"],
        node_hidden=config["node_hidden"]
    ).to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Parameters:", parameters)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=0
    )

    history = []

    start = time.time()

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * labels.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        test_loss, test_acc = evaluate(
            model,
            testloader,
            criterion
        )

        elapsed = time.time() - start

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"Train Acc: {train_acc:.2f}% "
            f"Test Acc: {test_acc:.2f}% "
            f"Train Loss: {train_loss:.4f} "
            f"Test Loss: {test_loss:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })

    with open(
        f"results/cifar10/capacity/{name}.json",
        "w"
    ) as f:

        json.dump(
            {
                "model": name,
                "parameters": parameters,
                "config": config,
                "history": history
            },
            f,
            indent=2
        )

    torch.save(
        model.state_dict(),
        f"results/cifar10/capacity/{name}.pt"
    )

    print(
        f"{name} complete. "
        f"Time: {elapsed:.1f}s"
    )

print()
print("ALL CAPACITY EXPERIMENTS COMPLETE.")
