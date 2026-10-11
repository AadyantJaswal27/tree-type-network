
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.tree_network import TreeNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "results/cifar10/capacity/baseline.pt"
OUT = "results/cifar10/innovations"
BATCH_SIZE = 128

os.makedirs(OUT, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

loader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

model = TreeNetwork().to(DEVICE)
state = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)
model.load_state_dict(state)
model.eval()

print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT)
print("Internal nodes:", model.num_nodes)


@torch.no_grad()
def evaluate(disabled_nodes=()):
    disabled = set(disabled_nodes)
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        features = model.features(images)
        features = model.projection(features)

        node_outputs = torch.stack(
            [node(features) for node in model.nodes],
            dim=1
        )

        if disabled:
            node_outputs[:, list(disabled)] = 0.0

        class_scores = []

        for class_idx in range(model.num_classes):
            leaf = class_idx
            score = torch.zeros(images.size(0), device=DEVICE)
            node_idx = 0

            for level in range(model.depth):
                bit = (leaf >> (model.depth - 1 - level)) & 1

                if bit == 0:
                    score = score - node_outputs[:, node_idx]
                    node_idx = 2 * node_idx + 1
                else:
                    score = score + node_outputs[:, node_idx]
                    node_idx = 2 * node_idx + 2

            class_scores.append(score)

        logits = torch.stack(class_scores, dim=1)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


baseline = evaluate()
print(f"\nBaseline test accuracy: {baseline:.2f}%")

rows = []

for node_idx in range(model.num_nodes):
    acc = evaluate((node_idx,))
    drop = baseline - acc

    row = {
        "node": node_idx,
        "test_accuracy": acc,
        "accuracy_drop_pp": drop
    }
    rows.append(row)

    print(
        f"Node {node_idx:2d}: "
        f"accuracy={acc:.2f}% | "
        f"drop={drop:+.2f} percentage points"
    )

ranking = sorted(rows, key=lambda r: r["accuracy_drop_pp"])

for rank, row in enumerate(ranking, start=1):
    row["least_important_rank"] = rank

pruning_rows = []
disabled = []

for row in ranking:
    disabled.append(row["node"])
    acc = evaluate(disabled)

    pruning_rows.append({
        "num_disabled_nodes": len(disabled),
        "disabled_nodes": json.dumps(disabled),
        "test_accuracy": acc,
        "accuracy_drop_pp": baseline - acc
    })

    print(
        f"Disabled {len(disabled):2d} nodes: "
        f"accuracy={acc:.2f}% | "
        f"drop={baseline - acc:+.2f} pp"
    )


def write_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


write_csv(os.path.join(OUT, "node_ablation.csv"), rows)
write_csv(os.path.join(OUT, "node_pruning_curve.csv"), pruning_rows)

ordered = sorted(rows, key=lambda r: r["node"])

plt.figure(figsize=(10, 5))
plt.bar(
    [r["node"] for r in ordered],
    [r["accuracy_drop_pp"] for r in ordered]
)
plt.axhline(0, linewidth=1)
plt.xlabel("Internal node index")
plt.ylabel("Baseline accuracy - ablated accuracy (pp)")
plt.title(f"Single-node ablation (baseline {baseline:.2f}%)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "node_importance.png"), dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(
    [0] + [r["num_disabled_nodes"] for r in pruning_rows],
    [baseline] + [r["test_accuracy"] for r in pruning_rows],
    marker="o"
)
plt.xlabel("Number of cumulatively disabled nodes")
plt.ylabel("Test accuracy (%)")
plt.title("Cumulative node ablation")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pruning_curve.png"), dpi=200)
plt.close()

summary = {
    "checkpoint": CHECKPOINT,
    "baseline_test_accuracy": baseline,
    "num_nodes": model.num_nodes,
    "least_impact_first_order": [r["node"] for r in ranking],
    "single_node_results": rows,
    "cumulative_ablation_results": pruning_rows
}

with open(
    os.path.join(OUT, "node_ablation_summary.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\nExperiment complete.")
print("Outputs saved to:", OUT)
print(" - node_ablation.csv")
print(" - node_pruning_curve.csv")
print(" - node_ablation_summary.json")
print(" - node_importance.png")
print(" - pruning_curve.png")
