import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tree_network import TreeNetwork


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

batch_size = 32
epochs = 20
lr = 1e-3

configs = {
    "xxxl": {
        "channels": (128, 256, 256),
        "feature_dim": 512,
        "node_hidden": 768
    },
    "4xl": {
        "channels": (128, 256, 256),
        "feature_dim": 512,
        "node_hidden": 1024
    },
    "5xl": {
        "channels": (128, 256, 256),
        "feature_dim": 512,
        "node_hidden": 2048
    },
    "6xl": {
        "channels": (128, 256, 256),
        "feature_dim": 512,
        "node_hidden": 4096
    }
}

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

os.makedirs(
    "results/cifar10/capacity_extended",
    exist_ok=True
)

os.makedirs(
    "checkpoints/cifar10/capacity_extended",
    exist_ok=True
)


for name, config in configs.items():

    print()
    print("=" * 60)
    print("STARTING:", name)
    print(config)
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

    print("Parameters:", parameters)

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

        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item() * labels.size(0)
            )

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

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                test_loss += (
                    loss.item() * labels.size(0)
                )

                predictions = outputs.argmax(dim=1)

                test_correct += (
                    predictions == labels
                ).sum().item()

                test_total += labels.size(0)

        train_loss /= train_total
        test_loss /= test_total

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
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        print(
            f"{name} "
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"Train {train_accuracy:.2f}% "
            f"Test {test_accuracy:.2f}%"
        )

    result = {
        "model": name,
        "parameters": parameters,
        "config": config,
        "best_train_accuracy": best_train,
        "best_test_accuracy": best_test,
        "history": history
    }

    with open(
        f"results/cifar10/capacity_extended/{name}.json",
        "w"
    ) as f:
        json.dump(
            result,
            f,
            indent=2
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "parameters": parameters
        },
        f"checkpoints/cifar10/capacity_extended/{name}.pt"
    )

    print()
    print(
        f"FINISHED {name}: "
        f"Best Train={best_train:.2f}% "
        f"Best Test={best_test:.2f}%"
    )

    del model
    del optimizer

    torch.cuda.empty_cache()


print()
print("EXTENDED-EXTENDED CAPACITY EXPERIMENT COMPLETE.")