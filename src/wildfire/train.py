from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from wildfire.data import create_dataloader
from wildfire.metrics import classification_metrics
from wildfire.models import build_model
from wildfire.utils import ensure_dir, load_config, save_json, set_seed


def run_epoch(model, loader, optimizer, criterion, device, train: bool) -> dict:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0
    summed = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for images, targets in tqdm(loader, disable=False):
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
    args = parser.parse_args()

    config = load_config(args.config)
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
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=False,
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

