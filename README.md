# Wildfire Detection Midterm Project

This repository is a starter research codebase for the wildfire-monitoring project.

The original proposal focused on multi-temporal segmentation, but the currently available dataset is the backup benchmark:

**binary wildfire image classification from satellite imagery**

## Project Goal

The current project studies:

1. `wildfire` vs `nowildfire` scene classification
2. baseline comparison between a small CNN and ResNet-18

## Repository Layout

```text
configs/                 Training and experiment configuration files
reports/                 Midterm outline and experiment plan
src/wildfire/            Code for data, models, training, and evaluation
requirements.txt         Python dependencies
```

## Expected Dataset Layout

The code expects:

```text
data/
  train/
    wildfire/
    nowildfire/
  valid/
    wildfire/
    nowildfire/
  test/
    wildfire/
    nowildfire/
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the simple baseline (SmallCNN):

```bash
python -m src.wildfire.train --config configs/small_cnn_classifier.yaml
```

Train the stronger baseline (ResNet-18):

```bash
# standard
python -m src.wildfire.train --config configs/resnet18_classifier.yaml

# fast preset (pretrained, 160px, 2 epochs; good for quick iterations)
python -m src.wildfire.train --config configs/resnet18_fast.yaml --max-train-batches 300 --max-val-batches 150
```

Speed/debug knobs (optional):
- `--epochs N` override epoch count.
- `--max-train-batches N` / `--max-val-batches N` cap batches per epoch for quick smoke tests.
- Set `training.device: cuda` in the config if you have a GPU.

Evaluate a trained checkpoint (saves metrics + confusion matrix + qualitative grid):

```bash
# test split
python -m src.wildfire.evaluate --config configs/resnet18_classifier.yaml --split test

# SmallCNN test
python -m src.wildfire.evaluate --config configs/small_cnn_classifier.yaml --split test
```

Evaluation artifacts land in the run output dir (e.g., `outputs/wildfire_resnet18_baseline/`):
- `eval_<split>.json` (overall metrics)
- `classification_report_<split>.json` (per class)
- `confusion_<split>.png`
- `qualitative_<split>.png`

## Midterm Deliverables

Use the files in `reports/` as your writing backbone:

1. `reports/midterm_outline.md`
2. `reports/experiment_matrix.md`
3. `reports/report_claims.md`
4. `reports/midterm_draft.md`

## Immediate Next Steps

1. Install the dependencies.
2. Run the small CNN baseline.
3. Run the ResNet-18 baseline.
4. Add one results table, one confusion matrix, and one qualitative figure to the report.
