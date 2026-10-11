
import os
import sys
import json
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.tree_network import TreeNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = "results/cifar10/capacity/baseline.pt"
COMPACT = "results/cifar10/innovations/structural_pruning/compact_tree.pt"
OUT = "results/cifar10/innovations/structural_pruning"
BATCH_SIZE = 128
WARMUP = 30
REPEATS = 10
ITERATIONS = 100

os.makedirs(OUT, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

original = TreeNetwork().to(DEVICE).eval()
original.load_state_dict(torch.load(BASE, map_location=DEVICE, weights_only=True))

saved = torch.load(COMPACT, map_location=DEVICE, weights_only=False)
from experiments.structural_pruning import CompactTree, USED
compact = CompactTree(original, saved["used_node_indices"]).to(DEVICE).eval()
compact.load_state_dict(saved["state_dict"])

@torch.no_grad()
def accuracy(model):
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100 * correct / total

@torch.no_grad()
def latency(model, x):
    for _ in range(WARMUP):
        model(x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(REPEATS):
        if DEVICE.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITERATIONS):
                model(x)
            end_event.record()
            torch.cuda.synchronize()
            samples.append(start_event.elapsed_time(end_event) / ITERATIONS)
        else:
            start = time.perf_counter()
            for _ in range(ITERATIONS):
                model(x)
            samples.append((time.perf_counter() - start) * 1000 / ITERATIONS)
    return samples

x, _ = next(iter(loader))
x = x.to(DEVICE)

print("Device:", DEVICE)
print("Evaluating accuracy...")
acc_original = accuracy(original)
acc_compact = accuracy(compact)

print("Benchmarking both models with warm-up and repeated measurements...")
original_ms = latency(original, x)
compact_ms = latency(compact, x)

med_original = statistics.median(original_ms)
med_compact = statistics.median(compact_ms)
speedup = 100 * (med_original - med_compact) / med_original

result = {
    "batch_size": int(x.size(0)),
    "warmup_iterations": WARMUP,
    "iterations_per_repeat": ITERATIONS,
    "repeats": REPEATS,
    "original_accuracy_percent": acc_original,
    "compact_accuracy_percent": acc_compact,
    "original_latency_ms_per_batch_median": med_original,
    "compact_latency_ms_per_batch_median": med_compact,
    "original_latency_samples_ms": original_ms,
    "compact_latency_samples_ms": compact_ms,
    "latency_change_percent": speedup,
    "interpretation": "Positive latency change means the compact model was faster in this benchmark."
}

path = os.path.join(OUT, "benchmark_results.json")
with open(path, "w") as f:
    json.dump(result, f, indent=2)

print("\n===== REPEATED BENCHMARK =====")
print(f"Original accuracy: {acc_original:.2f}%")
print(f"Compact accuracy:  {acc_compact:.2f}%")
print(f"Original median:   {med_original:.4f} ms/batch")
print(f"Compact median:    {med_compact:.4f} ms/batch")
print(f"Latency reduction: {speedup:.2f}%")
print("Saved:", path)
