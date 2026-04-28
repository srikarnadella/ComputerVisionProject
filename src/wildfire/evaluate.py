from __future__ import annotations

import argparse
import csv
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

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wildfire.data import create_dataloader
from wildfire.models import build_model
from wildfire.utils import load_config, save_json


def _get_wildfire_class_index(class_names: Sequence[str]) -> int:
    lower = [c.lower() for c in class_names]
    if "wildfire" in lower:
        return lower.index("wildfire")
    return len(class_names) - 1


def _save_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() * 0.7
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_pr_curve(y_true, y_scores, pos_label, out_path, title):
    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
    ap = auc(recall, precision) if len(recall) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return ap


def _save_roc_curve(y_true, y_scores, pos_label, out_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr) if len(fpr) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1],"k--",alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return roc_auc


def _save_per_class_bar(report, class_names, out_path):
    precisions = [report.get(n,{}).get("precision",0.0) for n in class_names]
    recalls    = [report.get(n,{}).get("recall",   0.0) for n in class_names]
    f1s        = [report.get(n,{}).get("f1-score", 0.0) for n in class_names]
    x = np.arange(len(class_names)); width = 0.25
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x-width, precisions, width, label="Precision")
    ax.bar(x,       recalls,    width, label="Recall")
    ax.bar(x+width, f1s,        width, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_title("Per-class metrics"); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def _save_qualitative_grid(images, preds, targets, class_names, out_path):
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    imgs = (images * std + mean).clamp(0,1)
    grid = make_grid(imgs, nrow=4, padding=2)
    np_img = grid.permute(1,2,0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(np_img); ax.axis("off")
    title_lines = [f"{i+1}: pred={class_names[preds[i]]}, true={class_names[targets[i]]}"
                   for i in range(len(preds))]
    ax.set_title("Qualitative predictions\n" + "\n".join(title_lines), fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def _save_prediction_table(dataset_samples, y_true, y_pred, y_score, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path","target","prediction","wildfire_probability","correct"])
        for idx, (path, _) in enumerate(dataset_samples[:len(y_true)]):
            writer.writerow([path, int(y_true[idx]), int(y_pred[idx]),
                             float(y_score[idx]), int(y_true[idx]==y_pred[idx])])


def evaluate_with_config(
    config: dict,
    split: str = "test",
    save_dir=None,
    num_qualitative: int = 12,
    no_confusion: bool = False,
    max_batches=None,
) -> dict:
    device = torch.device(config["training"]["device"])
    model  = build_model(config).to(device)

    checkpoint = Path(config["experiment"]["output_dir"]) / "best_model.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    loader       = create_dataloader(config, split=split)
    class_names  = loader.dataset.classes
    wildfire_idx = _get_wildfire_class_index(class_names)
    criterion    = torch.nn.CrossEntropyLoss()

    # Collect ALL predictions first, then compute metrics once.
    # Per-batch averaging of precision/recall is mathematically wrong and
    # produces frozen metrics — same bug that was in train.py.
    total_loss     = 0.0
    total_examples = 0
    all_logits     = []
    all_targets    = []
    all_pos_scores = []
    qual_images, qual_preds, qual_targets = [], [], []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader, start=1):
            images  = images.to(device)
            targets = targets.to(device)
            logits  = model(images)
            loss    = criterion(logits, targets)
            preds   = logits.argmax(dim=1)
            probs   = torch.softmax(logits, dim=1)[:, wildfire_idx]

            total_loss     += loss.item() * targets.size(0)
            total_examples += targets.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_pos_scores.append(probs.cpu())

            collected = sum(c.size(0) for c in qual_images)
            if collected < num_qualitative:
                keep = min(num_qualitative - collected, images.size(0))
                qual_images.append(images[:keep].cpu())
                qual_preds.append(preds[:keep].cpu())
                qual_targets.append(targets[:keep].cpu())

            if max_batches is not None and batch_idx >= max_batches:
                break

    all_logits_cat  = torch.cat(all_logits)
    all_targets_cat = torch.cat(all_targets)
    all_preds_cat   = all_logits_cat.argmax(dim=1)
    all_scores_cat  = torch.cat(all_pos_scores).numpy()
    targets_np      = all_targets_cat.numpy()
    preds_np        = all_preds_cat.numpy()

    # Compute metrics over full split at once
    num_classes = all_logits_cat.size(1)
    correct  = (all_preds_cat == all_targets_cat).sum().item()
    accuracy = correct / max(all_targets_cat.numel(), 1)
    eps = 1e-8
    precisions, recalls, f1s = [], [], []
    for cls in range(num_classes):
        tp = ((all_preds_cat==cls) & (all_targets_cat==cls)).sum().item()
        fp = ((all_preds_cat==cls) & (all_targets_cat!=cls)).sum().item()
        fn = ((all_preds_cat!=cls) & (all_targets_cat==cls)).sum().item()
        p = tp/(tp+fp+eps); r = tp/(tp+fn+eps)
        f = 2*p*r/(p+r+eps)
        precisions.append(p); recalls.append(r); f1s.append(f)

    results = {
        "loss":      total_loss / total_examples,
        "accuracy":  accuracy,
        "precision": sum(precisions) / num_classes,
        "recall":    sum(recalls)    / num_classes,
        "f1":        sum(f1s)        / num_classes,
    }
    print(results)

    save_dir = Path(save_dir or config["experiment"]["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, save_dir / f"eval_{split}.json")

    _save_prediction_table(loader.dataset.samples, targets_np, preds_np,
                           all_scores_cat, save_dir / f"predictions_{split}.csv")

    if not no_confusion:
        cm = confusion_matrix(targets_np, preds_np, labels=list(range(len(class_names))))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        _save_confusion_matrix(cm_norm, class_names, save_dir / f"confusion_{split}.png")

        report = classification_report(targets_np, preds_np,
                                       labels=list(range(len(class_names))),
                                       target_names=class_names,
                                       zero_division=0, output_dict=True)
        save_json(report, save_dir / f"classification_report_{split}.json")
        _save_per_class_bar(report, class_names, save_dir / f"per_class_{split}.png")

        pr_ap = _save_pr_curve(targets_np, all_scores_cat, wildfire_idx,
                               save_dir / f"pr_curve_{split}.png",
                               f"Precision-Recall ({split})")
        roc_auc = _save_roc_curve(targets_np, all_scores_cat, wildfire_idx,
                                  save_dir / f"roc_curve_{split}.png",
                                  f"ROC ({split})")
        results["pr_ap"]   = pr_ap
        results["roc_auc"] = roc_auc
        save_json(results, save_dir / f"eval_{split}.json")

    if qual_images:
        _save_qualitative_grid(
            torch.cat(qual_images)[:num_qualitative],
            torch.cat(qual_preds)[:num_qualitative],
            torch.cat(qual_targets)[:num_qualitative],
            class_names, save_dir / f"qualitative_{split}.png")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          required=True)
    parser.add_argument("--split",           default="test", choices=["train","val","test"])
    parser.add_argument("--save-dir",        default=None)
    parser.add_argument("--num-qualitative", type=int, default=12)
    parser.add_argument("--no-confusion",    action="store_true")
    parser.add_argument("--max-batches",     type=int, default=None)
    parser.add_argument("--num-workers",     type=int, default=None)
    parser.add_argument("--device",          default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.device is not None:
        config["training"]["device"] = args.device

    evaluate_with_config(config=config, split=args.split, save_dir=args.save_dir,
                         num_qualitative=args.num_qualitative, no_confusion=args.no_confusion,
                         max_batches=args.max_batches)


if __name__ == "__main__":
    main()