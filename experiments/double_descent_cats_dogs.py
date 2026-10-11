import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.cats_dogs_tree import CatsDogsTree


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

hidden_dims = [
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192
]

epochs = 50
batch_size = 128
lr = 1e-3

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

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    TensorDataset(x_val, y_val),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    TensorDataset(x_test, y_test),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

os.makedirs(
    "results/cats_dogs/double_descent",
    exist_ok=True
)

for hidden_dim in hidden_dims:

    print()
    print("=" * 60)
    print("HIDDEN DIMENSION:", hidden_dim)
    print("=" * 60)

    model = CatsDogsTree(
        hidden_dim=hidden_dim
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Parameters:", parameters)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0
    )

    history = []

    best_train = 0
    best_val = 0
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

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            train_correct += (
                predictions == y
            ).sum().item()

            train_total += y.size(0)

        model.eval()

        val_correct = 0
        val_total = 0

        test_correct = 0
        test_total = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                logits = model(x)

                predictions = (
                    torch.sigmoid(logits) >= 0.5
                ).float()

                val_correct += (
                    predictions == y
                ).sum().item()

                val_total += y.size(0)

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

        train_accuracy = (
            100 * train_correct / train_total
        )

        val_accuracy = (
            100 * val_correct / val_total
        )

        test_accuracy = (
            100 * test_correct / test_total
        )

        best_train = max(
            best_train,
            train_accuracy
        )

        best_val = max(
            best_val,
            val_accuracy
        )

        best_test = max(
            best_test,
            test_accuracy
        )

        history.append({
            "epoch": epoch + 1,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy
        })

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"Train {train_accuracy:.2f}% "
            f"Val {val_accuracy:.2f}% "
            f"Test {test_accuracy:.2f}%"
        )

    result = {
        "hidden_dim": hidden_dim,
        "parameters": parameters,
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "final_test_accuracy": history[-1]["test_accuracy"],
        "best_train_accuracy": best_train,
        "best_val_accuracy": best_val,
        "best_test_accuracy": best_test,
        "history": history
    }

    with open(
        f"results/cats_dogs/double_descent/h{hidden_dim}.json",
        "w"
    ) as f:
        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print(
        f"FINISHED h={hidden_dim} "
        f"params={parameters} "
        f"Final Train={result['final_train_accuracy']:.2f}% "
        f"Final Test={result['final_test_accuracy']:.2f}%"
    )

    del model
    del optimizer

    torch.cuda.empty_cache()

print()
print("CATS VS DOGS DOUBLE DESCENT COMPLETE.")
