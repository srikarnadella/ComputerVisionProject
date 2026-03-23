from __future__ import annotations

import torch


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    accuracy = correct / max(targets.numel(), 1)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

