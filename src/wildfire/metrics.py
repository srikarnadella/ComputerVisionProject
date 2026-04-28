from __future__ import annotations

import torch


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """
    Compute accuracy, precision, recall, and F1 as macro averages across all
    classes.

    BUG FIXED: the original implementation only computed metrics for class
    index 1 (wildfire).  When the model predicted almost exclusively class 0
    during early training, tp and fp were near zero and the epsilon term
    dominated, producing a frozen val precision of ~0.553 every epoch.
    Macro averaging across both classes fixes this.
    """
    preds = logits.argmax(dim=1)
    num_classes = logits.size(1)

    correct = (preds == targets).sum().item()
    accuracy = correct / max(targets.numel(), 1)

    eps = 1e-8
    precisions, recalls, f1s = [], [], []

    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()

        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f = 2 * p * r / (p + r + eps)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "accuracy":  accuracy,
        "precision": sum(precisions) / num_classes,
        "recall":    sum(recalls)    / num_classes,
        "f1":        sum(f1s)        / num_classes,
    }