from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wildfire.data import create_dataloader
from wildfire.metrics import classification_metrics
from wildfire.models import build_model
from wildfire.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["training"]["device"])
    model = build_model(config).to(device)

    checkpoint = Path(config["experiment"]["output_dir"]) / "best_model.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    loader = create_dataloader(config, split=args.split)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    summed = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            metrics = classification_metrics(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            for key in summed:
                summed[key] += metrics[key] * batch_size

    results = {"loss": total_loss / total_examples}
    for key in summed:
        results[key] = summed[key] / total_examples
    print(results)


if __name__ == "__main__":
    main()

