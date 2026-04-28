"""
analyze_errors.py
=================
Post-hoc error analysis for the wildfire classifier.

Given the predictions CSV produced by evaluate.py, this script:
  1. Loads every misclassified image
  2. Assigns each error to one of four heuristic failure modes:
       SMOKE_HAZE      — low-saturation warm tones (smoke without active fire)
       BRIGHT_TERRAIN  — high-brightness, low-saturation (sunlit soil/roads/snow)
       LOW_CONTRAST    — very dark or flat scenes (night / heavy overcast)
       OTHER           — visually ambiguous; no dominant pattern
  3. Saves all outputs to the output directory — no model re-running needed.

Outputs saved
-------------
  summary.json                   counts, accuracy, confidence stats
  failure_mode_bar.png           bar chart of error categories
  score_distribution.png         confidence histogram: correct vs wrong
  failure_grid_smoke_haze.png    image grid for each failure mode
  failure_grid_bright_terrain.png
  failure_grid_low_contrast.png
  failure_grid_other.png
  failure_grid_false_negatives.png   missed wildfires (safety-critical)
  failure_grid_false_positives.png   false alarms
  error_analysis.md              paste-ready report section

Usage
-----
  python -m src.wildfire.analyze_errors \\
    --predictions outputs/wildfire_resnet18_final/predictions_test.csv \\
    --output-dir  outputs/wildfire_resnet18_final/errors \\
    --image-size  224 \\
    --num-per-category 16
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid


# ── Heuristic failure-mode classifier ───────────────────────────────────────

def _image_features(img_path: str, size: int = 64) -> dict:
    """Fast colour/contrast statistics for a single image (tiny resize)."""
    try:
        img = Image.open(img_path).convert("RGB").resize((size, size))
        rgb = np.array(img, dtype=np.float32) / 255.0
        hsv = np.array(img.convert("HSV"), dtype=np.float32) / 255.0

        warm_mask = (hsv[..., 0] <= 0.18) & (hsv[..., 1] >= 0.25)
        return {
            "mean_brightness": float(rgb.mean()),
            "contrast":        float(rgb.std()),
            "mean_saturation": float(hsv[..., 1].mean()),
            "warm_fraction":   float(warm_mask.mean()),
        }
    except Exception:
        return {"mean_brightness": 0.5, "contrast": 0.1,
                "mean_saturation": 0.1, "warm_fraction": 0.0}


def _classify_failure_mode(f: dict) -> str:
    """Rule-based category assignment. Earlier rules take priority."""
    if f["mean_brightness"] < 0.20 or f["contrast"] < 0.06:
        return "LOW_CONTRAST"
    if f["mean_saturation"] < 0.18 and f["warm_fraction"] < 0.06 and f["mean_brightness"] > 0.30:
        return "SMOKE_HAZE"
    if f["mean_brightness"] > 0.60 and f["mean_saturation"] < 0.20:
        return "BRIGHT_TERRAIN"
    return "OTHER"


# ── Image grid helpers ───────────────────────────────────────────────────────

def _load_tensor(img_path: str, size: int) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB").resize((size, size))
    return TF.to_tensor(img)


def _save_image_grid(paths: list[str], title: str, out_path: Path, size: int, nrow: int = 4) -> None:
    tensors = [_load_tensor(p, size) for p in paths[: nrow * 8]]
    if not tensors:
        return
    grid   = make_grid(tensors, nrow=nrow, padding=3, normalize=False)
    np_img = grid.permute(1, 2, 0).numpy()
    rows   = (len(tensors) + nrow - 1) // nrow
    fig, ax = plt.subplots(figsize=(nrow * 2, rows * 2 + 1))
    ax.imshow(np_img)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_confidence_histogram(correct_scores: list[float], error_scores: list[float], out_path: Path) -> None:
    bins = np.linspace(0, 1, 26)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(correct_scores, bins=bins, alpha=0.6, label=f"Correct (n={len(correct_scores)})", color="#2196F3")
    ax.hist(error_scores,   bins=bins, alpha=0.7, label=f"Errors  (n={len(error_scores)})",   color="#F44336")
    ax.set_xlabel("Wildfire probability (model confidence)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence distribution: correct vs misclassified")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_category_bar(category_counts: dict[str, int], out_path: Path) -> None:
    cats   = list(category_counts.keys())
    counts = [category_counts[c] for c in cats]
    colors = {"SMOKE_HAZE": "#FF9800", "BRIGHT_TERRAIN": "#8BC34A",
              "LOW_CONTRAST": "#607D8B", "OTHER": "#9E9E9E"}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(cats, counts, color=[colors[c] for c in cats], edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Number of errors")
    ax.set_title("Misclassifications by failure mode")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def analyze_errors(
    predictions_csv: str,
    output_dir: str,
    image_size: int = 224,
    num_per_category: int = 16,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(predictions_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "path":       row["image_path"],
                "target":     int(row["target"]),
                "prediction": int(row["prediction"]),
                "score":      float(row["wildfire_probability"]),
                "correct":    int(row["correct"]),
            })

    errors  = [r for r in rows if r["correct"] == 0]
    correct = [r for r in rows if r["correct"] == 1]
    false_positives = [r for r in errors if r["target"] == 0 and r["prediction"] == 1]
    false_negatives = [r for r in errors if r["target"] == 1 and r["prediction"] == 0]

    print(f"Total : {len(rows)}  |  Correct: {len(correct)}  |  Errors: {len(errors)}")
    print(f"  FP (nowildfire→wildfire): {len(false_positives)}")
    print(f"  FN (wildfire missed):     {len(false_negatives)}")

    print("\nScoring error images...")
    categories = ["SMOKE_HAZE", "BRIGHT_TERRAIN", "LOW_CONTRAST", "OTHER"]
    by_category: dict[str, list] = {c: [] for c in categories}
    for r in errors:
        r["failure_mode"] = _classify_failure_mode(_image_features(r["path"]))
        by_category[r["failure_mode"]].append(r)

    category_counts = {c: len(by_category[c]) for c in categories}
    print("\nFailure mode breakdown:")
    for cat, cnt in category_counts.items():
        print(f"  {cat:<20s} {cnt:4d}  ({100*cnt/max(len(errors),1):.1f}%)")

    print("\nSaving grids and plots...")
    descriptions = {
        "SMOKE_HAZE":     "Smoke or haze — warm tones, low saturation",
        "BRIGHT_TERRAIN": "Bright terrain — high brightness, low saturation",
        "LOW_CONTRAST":   "Low contrast — night or heavy overcast",
        "OTHER":          "Other / ambiguous",
    }
    for cat in categories:
        items = by_category[cat][:num_per_category]
        if items:
            _save_image_grid(
                [r["path"] for r in items],
                f"{cat} ({len(by_category[cat])} errors) — {descriptions[cat]}",
                out / f"failure_grid_{cat.lower()}.png",
                size=image_size,
            )

    _save_image_grid(
        [r["path"] for r in false_negatives[:num_per_category]],
        f"False Negatives — wildfire MISSED by model (n={len(false_negatives)})",
        out / "failure_grid_false_negatives.png", size=image_size,
    )
    _save_image_grid(
        [r["path"] for r in false_positives[:num_per_category]],
        f"False Positives — nowildfire predicted as wildfire (n={len(false_positives)})",
        out / "failure_grid_false_positives.png", size=image_size,
    )

    _save_confidence_histogram(
        [r["score"] for r in correct],
        [r["score"] for r in errors],
        out / "score_distribution.png",
    )
    _save_category_bar(category_counts, out / "failure_mode_bar.png")

    summary = {
        "total":                  len(rows),
        "correct":                len(correct),
        "errors":                 len(errors),
        "accuracy":               len(correct) / max(len(rows), 1),
        "false_positives":        len(false_positives),
        "false_negatives":        len(false_negatives),
        "failure_modes":          category_counts,
        "mean_error_confidence":  float(np.mean([r["score"] for r in errors]))  if errors  else 0.0,
        "mean_correct_confidence":float(np.mean([r["score"] for r in correct])) if correct else 0.0,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _write_markdown(summary, category_counts, descriptions, out)
    print(f"\nAll error-analysis artifacts saved to: {out}")
    return summary


def _write_markdown(summary: dict, category_counts: dict, descriptions: dict, out: Path) -> None:
    total_errors = summary["errors"]
    lines = [
        "## Error Analysis",
        "",
        f"Out of **{summary['total']}** test images the model made "
        f"**{total_errors}** errors ({100*(1-summary['accuracy']):.2f}% error rate).",
        "",
        f"- **False Positives** (nowildfire → wildfire): {summary['false_positives']}",
        f"- **False Negatives** (wildfire missed): {summary['false_negatives']}",
        "",
        "### Failure Mode Breakdown",
        "",
        "| Failure Mode | Count | % of Errors | Description |",
        "|---|---:|---:|---|",
    ]
    for cat, cnt in category_counts.items():
        pct = 100 * cnt / max(total_errors, 1)
        lines.append(f"| {cat} | {cnt} | {pct:.1f}% | {descriptions[cat]} |")

    lines += [
        "",
        "### Confidence Distribution",
        "",
        f"Mean wildfire confidence on **correct** predictions: "
        f"**{summary['mean_correct_confidence']:.3f}**  ",
        f"Mean wildfire confidence on **errors**: "
        f"**{summary['mean_error_confidence']:.3f}**",
        "",
        "See `score_distribution.png` for the full histogram.",
        "",
        "> Failure modes are assigned by a heuristic colour/contrast scorer. "
        "Use the image grids for qualitative inspection.",
    ]
    (out / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis for wildfire classifier")
    parser.add_argument("--predictions",       required=True,
                        help="Path to predictions_<split>.csv from evaluate.py")
    parser.add_argument("--output-dir",        default=None,
                        help="Where to save artifacts (default: <predictions_dir>/errors/)")
    parser.add_argument("--image-size",        type=int, default=224)
    parser.add_argument("--num-per-category",  type=int, default=16,
                        help="Max images per failure-mode grid")
    args = parser.parse_args()

    out_dir = args.output_dir or str(Path(args.predictions).parent / "errors")
    analyze_errors(
        predictions_csv=args.predictions,
        output_dir=out_dir,
        image_size=args.image_size,
        num_per_category=args.num_per_category,
    )


if __name__ == "__main__":
    main()