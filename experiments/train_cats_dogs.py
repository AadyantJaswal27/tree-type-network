import os
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

x_train = data["ftr"].float()
y_train = data["ytr"].float()

x_val = data["fva"].float()
y_val = data["yva"].float()

x_test = data["fte"].float()
y_test = data["yte"].float()

train_dataset = TensorDataset(
    x_train,
    y_train
)

val_dataset = TensorDataset(
    x_val,
    y_val
)

test_dataset = TensorDataset(
    x_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

model = CatsDogsTree().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

os.makedirs(
    "results/cats_dogs",
    exist_ok=True
)

os.makedirs(
    "checkpoints/cats_dogs",
    exist_ok=True
)

epochs = 20

history = []

best_val = 0

for epoch in range(epochs):

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

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

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        train_correct += (
            predictions == y
        ).sum().item()

        train_total += y.size(0)

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            val_loss += (
                loss.item() * y.size(0)
            )

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            val_correct += (
                predictions == y
            ).sum().item()

            val_total += y.size(0)

    train_loss /= train_total
    val_loss /= val_total

    train_accuracy = (
        100 * train_correct / train_total
    )

    val_accuracy = (
        100 * val_correct / val_total
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val:

        best_val = val_accuracy

        torch.save(
            model.state_dict(),
            "checkpoints/cats_dogs/baseline_best.pt"
        )

model.load_state_dict(
    torch.load(
        "checkpoints/cats_dogs/baseline_best.pt",
        map_location=device,
        weights_only=True
    )
)

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        test_correct += (
            predictions == y
        ).sum().item()

        test_total += y.size(0)

test_accuracy = (
    100 * test_correct / test_total
)

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

result = {
    "model": "cats_dogs_baseline",
    "parameters": parameters,
    "tree_nodes": 1,
    "best_val_accuracy": best_val,
    "test_accuracy": test_accuracy,
    "history": history
}

with open(
    "results/cats_dogs/baseline.json",
    "w"
) as f:
    json.dump(
        result,
        f,
        indent=2
    )

print()
print("TRAINING COMPLETE")
print("Parameters:", parameters)
print("Tree nodes: 1")
print("Best validation accuracy:", best_val)
print("Test accuracy:", test_accuracy)