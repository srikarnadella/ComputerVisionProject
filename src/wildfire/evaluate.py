from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torchvision.utils import make_grid

# Ensure `src` is on sys.path so `python -m wildfire.evaluate` works without PYTHONPATH
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wildfire.data import create_dataloader
from wildfire.metrics import classification_metrics
from wildfire.models import build_model
from wildfire.utils import load_config, save_json


def _save_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() * 0.7
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_pr_curve(
    y_true: np.ndarray, y_scores: np.ndarray, pos_label: int, out_path: Path, title: str
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
    ap = auc(recall, precision) if len(recall) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ap


def _save_roc_curve(
    y_true: np.ndarray, y_scores: np.ndarray, pos_label: int, out_path: Path, title: str
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr) if len(fpr) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return roc_auc


def _save_per_class_bar(report: dict, class_names: Sequence[str], out_path: Path) -> None:
    # report is a dict from classification_report(output_dict=True)
    precisions = [report.get(name, {}).get("precision", 0.0) for name in class_names]
    recalls = [report.get(name, {}).get("recall", 0.0) for name in class_names]
    f1s = [report.get(name, {}).get("f1-score", 0.0) for name in class_names]

    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-class metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_qualitative_grid(
    images: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str],
    out_path: Path,
) -> None:
    # Unnormalize using ImageNet stats to render correctly
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs = images * std + mean
    imgs = imgs.clamp(0, 1)

    grid = make_grid(imgs, nrow=4, padding=2)
    np_img = grid.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np_img)
    ax.axis("off")
    title_lines = []
    for idx in range(len(preds)):
        title_lines.append(
            f"{idx+1}: pred={class_names[preds[idx]]}, true={class_names[targets[idx]]}"
        )
    ax.set_title("Qualitative predictions\n" + "\n".join(title_lines), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--save-dir", default=None, help="Override output dir for artifacts")
    parser.add_argument("--num-qualitative", type=int, default=12, help="Images to include in grid")
    parser.add_argument(
        "--no-confusion",
        action="store_true",
        help="Skip saving confusion matrix and per-class report",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit number of batches to evaluate (for quick smoke tests)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader num_workers for faster CPU loading",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (e.g., cuda or cpu) for evaluation",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.device is not None:
        config["training"]["device"] = args.device
    device = torch.device(config["training"]["device"])
    model = build_model(config).to(device)

    checkpoint = Path(config["experiment"]["output_dir"]) / "best_model.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    loader = create_dataloader(config, split=args.split)
    class_names = loader.dataset.classes
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0
    summed = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    all_targets = []
    all_preds = []
    all_pos_scores = []
    qual_images = []
    qual_preds = []
    qual_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader, start=1):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            metrics = classification_metrics(logits, targets)
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]  # score for class index 1 (wildfire)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            for key in summed:
                summed[key] += metrics[key] * batch_size
            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())
            all_pos_scores.append(probs.cpu())

            if len(qual_images) < args.num_qualitative:
                keep = min(args.num_qualitative - len(qual_images), images.size(0))
                qual_images.append(images[:keep].cpu())
                qual_preds.append(preds[:keep].cpu())
                qual_targets.append(targets[:keep].cpu())

            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

    results = {"loss": total_loss / total_examples}
    for key in summed:
        results[key] = summed[key] / total_examples
    print(results)

    save_dir = Path(args.save_dir or config["experiment"]["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, save_dir / f"eval_{args.split}.json")

    all_targets_cat = torch.cat(all_targets).numpy()
    all_preds_cat = torch.cat(all_preds).numpy()
    all_scores_cat = torch.cat(all_pos_scores).numpy()

    if not args.no_confusion:
        cm = confusion_matrix(all_targets_cat, all_preds_cat, labels=list(range(len(class_names))))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        _save_confusion_matrix(cm_norm, class_names, save_dir / f"confusion_{args.split}.png")

        report = classification_report(
            all_targets_cat,
            all_preds_cat,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
        save_json(report, save_dir / f"classification_report_{args.split}.json")
        _save_per_class_bar(report, class_names, save_dir / f"per_class_{args.split}.png")

        # Binary curves assume class 1 is positive (wildfire)
        pr_ap = _save_pr_curve(
            all_targets_cat,
            all_scores_cat,
            pos_label=1,
            out_path=save_dir / f"pr_curve_{args.split}.png",
            title=f"Precision–Recall ({args.split})",
        )
        roc_auc = _save_roc_curve(
            all_targets_cat,
            all_scores_cat,
            pos_label=1,
            out_path=save_dir / f"roc_curve_{args.split}.png",
            title=f"ROC ({args.split})",
        )
        results["pr_ap"] = pr_ap
        results["roc_auc"] = roc_auc
        save_json(results, save_dir / f"eval_{args.split}.json")

    if qual_images:
        images_tensor = torch.cat(qual_images, dim=0)[: args.num_qualitative]
        preds_tensor = torch.cat(qual_preds, dim=0)[: args.num_qualitative]
        targets_tensor = torch.cat(qual_targets, dim=0)[: args.num_qualitative]
        _save_qualitative_grid(
            images_tensor, preds_tensor, targets_tensor, class_names, save_dir / f"qualitative_{args.split}.png"
        )


if __name__ == "__main__":
    main()

