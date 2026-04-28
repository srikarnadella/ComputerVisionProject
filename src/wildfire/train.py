from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wildfire.data import create_dataloader
from wildfire.models import build_model
from wildfire.utils import ensure_dir, load_config, save_json, set_seed


def _compute_metrics(all_logits: torch.Tensor, all_targets: torch.Tensor, loss: float) -> dict:
    """
    Compute metrics over the FULL epoch at once.

    Computing precision/recall per-batch and averaging is mathematically wrong
    — a batch where the model predicts all one class gives a precision of 0 or
    1, and averaging those across hundreds of batches produces a stable but
    meaningless number (the root cause of the frozen val precision bug).

    Collecting all predictions first and computing once gives the true values.
    """
    num_classes = all_logits.size(1)
    preds = all_logits.argmax(dim=1)

    correct = (preds == all_targets).sum().item()
    accuracy = correct / max(all_targets.numel(), 1)

    eps = 1e-8
    precisions, recalls, f1s = [], [], []
    for cls in range(num_classes):
        tp = ((preds == cls) & (all_targets == cls)).sum().item()
        fp = ((preds == cls) & (all_targets != cls)).sum().item()
        fn = ((preds != cls) & (all_targets == cls)).sum().item()
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f = 2 * p * r / (p + r + eps)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "loss":      loss,
        "accuracy":  accuracy,
        "precision": sum(precisions) / num_classes,
        "recall":    sum(recalls)    / num_classes,
        "f1":        sum(f1s)        / num_classes,
    }


def run_train_epoch(model, loader, optimizer, criterion, device, max_batches=None) -> dict:
    model.train()
    total_loss     = 0.0
    total_examples = 0
    all_logits     = []
    all_targets    = []

    for batch_idx, (images, targets) in enumerate(tqdm(loader, leave=False), start=1):
        images  = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss     += loss.item() * targets.size(0)
        total_examples += targets.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if max_batches is not None and batch_idx >= max_batches:
            break

    if total_examples == 0:
        return {"loss": None, "accuracy": None, "precision": None, "recall": None, "f1": None}

    return _compute_metrics(
        torch.cat(all_logits),
        torch.cat(all_targets),
        total_loss / total_examples,
    )


def run_val_epoch(model, loader, criterion, device, max_batches=None) -> dict:
    model.eval()
    total_loss     = 0.0
    total_examples = 0
    all_logits     = []
    all_targets    = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, leave=False), start=1):
            images  = images.to(device)
            targets = targets.to(device)
            logits  = model(images)
            loss    = criterion(logits, targets)

            total_loss     += loss.item() * targets.size(0)
            total_examples += targets.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

            if max_batches is not None and batch_idx >= max_batches:
                break

    if total_examples == 0:
        return {"loss": None, "accuracy": None, "precision": None, "recall": None, "f1": None}

    return _compute_metrics(
        torch.cat(all_logits),
        torch.cat(all_targets),
        total_loss / total_examples,
    )


def _compute_class_weights(dataset) -> torch.Tensor:
    targets      = torch.tensor(dataset.targets, dtype=torch.long)
    num_classes  = len(dataset.classes)
    class_counts = torch.bincount(targets, minlength=num_classes).float().clamp(min=1.0)
    return class_counts.sum() / (class_counts * num_classes)


def _save_training_curves(history: list[dict], output_dir: Path) -> None:
    epochs  = [row["epoch"] for row in history]
    metrics = ["loss", "accuracy", "precision", "recall", "f1"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(epochs, [row["train"][metric] for row in history], marker="o", label="train")
        ax.plot(epochs, [row["val"][metric]   for row in history], marker="o", label="val")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        if metric == "loss":
            ax.legend(loc="upper right")
        else:
            ax.set_ylim(0, 1)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def train_with_config(
    config: dict,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    epochs_override: int | None = None,
) -> dict:
    config = copy.deepcopy(config)
    if epochs_override is not None:
        config["training"]["epochs"] = epochs_override

    set_seed(config["experiment"]["seed"])
    device     = torch.device(config["training"]["device"])
    output_dir = ensure_dir(config["experiment"]["output_dir"])

    train_loader = create_dataloader(config, split="train")
    val_loader   = create_dataloader(config, split="val")
    model        = build_model(config).to(device)
    optimizer    = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    class_weight_mode = config["training"].get("class_weight_mode", "none")
    if class_weight_mode == "balanced":
        class_weights = _compute_class_weights(train_loader.dataset).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(f"Class weights: {class_weights.tolist()}")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    history         = []
    best_val_f1     = -1.0
    best_model_path = Path(output_dir) / "best_model.pt"

    for epoch in range(config["training"]["epochs"]):
        train_metrics = run_train_epoch(
            model, train_loader, optimizer, criterion, device,
            max_batches=max_train_batches or config["training"].get("max_train_batches"),
        )
        val_metrics = run_val_epoch(
            model, val_loader, criterion, device,
            max_batches=max_val_batches or config["training"].get("max_val_batches"),
        )

        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(record)

        current_val_f1 = val_metrics["f1"] if val_metrics["f1"] is not None else -1.0
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best val F1: {best_val_f1:.4f} — checkpoint saved")

    save_json(history, Path(output_dir) / "metrics.json")
    _save_training_curves(history, Path(output_dir))

    return {
        "best_model_path": str(best_model_path),
        "best_val_f1":     best_val_f1,
        "output_dir":      str(output_dir),
        "final_epoch":     history[-1] if history else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",            required=True)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches",   type=int, default=None)
    parser.add_argument("--epochs",            type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_with_config(
        config=config,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()