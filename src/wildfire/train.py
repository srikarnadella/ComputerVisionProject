from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure `src` is on sys.path so `python -m wildfire.train` works without PYTHONPATH
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from tqdm import tqdm

from wildfire.data import create_dataloader
from wildfire.metrics import classification_metrics
from wildfire.models import build_model
from wildfire.utils import ensure_dir, load_config, save_json, set_seed


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train: bool,
    max_batches: int | None = None,
) -> dict:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0
    summed = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for batch_idx, (images, targets) in enumerate(tqdm(loader, disable=False), start=1):
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = classification_metrics(logits.detach(), targets.detach())
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        for key in summed:
            summed[key] += metrics[key] * batch_size
        if max_batches is not None and batch_idx >= max_batches:
            break

    if total_examples == 0:
        return {"loss": None, "accuracy": None, "precision": None, "recall": None, "f1": None}

    return {
        "loss": total_loss / total_examples,
        "accuracy": summed["accuracy"] / total_examples,
        "precision": summed["precision"] / total_examples,
        "recall": summed["recall"] / total_examples,
        "f1": summed["f1"] / total_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit number of training batches per epoch for quick runs",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit number of validation batches per epoch for quick runs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (defaults to config.training.epochs)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    set_seed(config["experiment"]["seed"])

    device = torch.device(config["training"]["device"])
    output_dir = ensure_dir(config["experiment"]["output_dir"])

    train_loader = create_dataloader(config, split="train")
    val_loader = create_dataloader(config, split="val")

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    best_val_f1 = -1.0
    best_model_path = Path(output_dir) / "best_model.pt"

    for epoch in range(config["training"]["epochs"]):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            max_batches=args.max_train_batches or config["training"].get("max_train_batches"),
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=False,
            max_batches=args.max_val_batches or config["training"].get("max_val_batches"),
        )

        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(record)

        current_val_f1 = val_metrics["f1"] if val_metrics["f1"] is not None else -1.0
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            torch.save(model.state_dict(), best_model_path)

    save_json(history, Path(output_dir) / "metrics.json")


if __name__ == "__main__":
    main()

