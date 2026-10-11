
import os
import sys
import json
import time
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.tree_network import TreeNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "results/cifar10/capacity/baseline.pt"
OUT = "results/cifar10/innovations/structural_pruning"
BATCH_SIZE = 128

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42)

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

original = TreeNetwork().to(DEVICE)
state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
original.load_state_dict(state)
original.eval()

def used_node_indices(num_classes, depth):
    used = set()

    for class_idx in range(num_classes):
        node_idx = 0

        for level in range(depth):
            used.add(node_idx)
            bit = (class_idx >> (depth - 1 - level)) & 1

            if bit == 0:
                node_idx = 2 * node_idx + 1
            else:
                node_idx = 2 * node_idx + 2

    return sorted(used)


USED = used_node_indices(original.num_classes, original.depth)
REMOVED = sorted(set(range(original.num_nodes)) - set(USED))
NODE_MAP = {old: new for new, old in enumerate(USED)}


class CompactTree(torch.nn.Module):
    def __init__(self, source, used_indices):
        super().__init__()

        self.num_classes = source.num_classes
        self.depth = source.depth
        self.num_nodes = len(used_indices)
        self.features = copy.deepcopy(source.features)
        self.projection = copy.deepcopy(source.projection)
        self.original_indices = list(used_indices)
        self.nodes = torch.nn.ModuleList([
            copy.deepcopy(source.nodes[i]) for i in used_indices
        ])

    def forward(self, images):
        x = self.features(images)
        x = self.projection(x)

        node_outputs = {
            old_idx: self.nodes[new_idx](x)
            for new_idx, old_idx in enumerate(self.original_indices)
        }

        scores = []

        for class_idx in range(self.num_classes):
            score = torch.zeros(images.size(0), device=images.device)
            node_idx = 0

            for level in range(self.depth):
                bit = (class_idx >> (self.depth - 1 - level)) & 1

                if bit == 0:
                    score = score - node_outputs[node_idx]
                    node_idx = 2 * node_idx + 1
                else:
                    score = score + node_outputs[node_idx]
                    node_idx = 2 * node_idx + 2

            scores.append(score)

        return torch.stack(scores, dim=1)


compact = CompactTree(original, USED).to(DEVICE)
compact.eval()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

original_params = count_params(original)
compact_params = count_params(compact)
removed_params = original_params - compact_params
reduction_pct = 100.0 * removed_params / original_params

print("Device:", DEVICE)
print("Original parameters:", original_params)
print("Used node indices:", USED)
print("Removed node indices:", REMOVED)
print("Compact parameters:", compact_params)
print(f"Parameter reduction: {removed_params} ({reduction_pct:.2f}%)")

@torch.no_grad()
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total

@torch.no_grad()
def verify_equivalence():
    max_abs_error = 0.0
    identical_predictions = True

    for images, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        a = original(images)
        b = compact(images)

        max_abs_error = max(
            max_abs_error,
            (a - b).abs().max().item()
        )

        if not torch.equal(a.argmax(1), b.argmax(1)):
            identical_predictions = False

    return max_abs_error, identical_predictions

@torch.no_grad()
def benchmark(model, warmup=10, iterations=50):
    model.eval()
    sample = next(iter(loader))[0][:min(BATCH_SIZE, 128)].to(DEVICE)

    for _ in range(warmup):
        model(sample)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iterations):
        model(sample)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    return 1000.0 * (time.perf_counter() - start) / iterations

print("\nEvaluating full test set...")
original_acc = evaluate(original)
compact_acc = evaluate(compact)

print("Verifying logits and predictions...")
max_error, same_predictions = verify_equivalence()

print("Benchmarking inference...")
original_ms = benchmark(original)
compact_ms = benchmark(compact)

result = {
    "checkpoint": CHECKPOINT,
    "original_parameters": original_params,
    "compact_parameters": compact_params,
    "removed_parameters": removed_params,
    "parameter_reduction_percent": reduction_pct,
    "used_node_indices": USED,
    "removed_node_indices": REMOVED,
    "original_test_accuracy_percent": original_acc,
    "compact_test_accuracy_percent": compact_acc,
    "max_absolute_logit_error": max_error,
    "predictions_identical": same_predictions,
    "original_latency_ms_per_batch": original_ms,
    "compact_latency_ms_per_batch": compact_ms,
    "latency_change_percent": 100.0 * (compact_ms - original_ms) / original_ms
}

torch.save(
    {
        "state_dict": compact.state_dict(),
        "used_node_indices": USED,
        "num_classes": compact.num_classes,
        "depth": compact.depth
    },
    os.path.join(OUT, "compact_tree.pt")
)

with open(os.path.join(OUT, "structural_pruning_results.json"), "w") as f:
    json.dump(result, f, indent=2)

print("\n========== FINAL RESULTS ==========")
print(f"Original accuracy: {original_acc:.2f}%")
print(f"Compact accuracy:  {compact_acc:.2f}%")
print(f"Max absolute logit error: {max_error:.10g}")
print("Predictions identical:", same_predictions)
print(f"Original latency: {original_ms:.3f} ms/batch")
print(f"Compact latency:  {compact_ms:.3f} ms/batch")
print("Saved compact model and results to:", OUT)

if not same_predictions:
    print("WARNING: predictions differ; investigate before claiming equivalence.")
