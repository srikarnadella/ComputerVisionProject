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

Train the simple baseline:

```bash
PYTHONPATH=src python3 -m wildfire.train --config configs/small_cnn_classifier.yaml
```

Train the stronger baseline:

```bash
PYTHONPATH=src python3 -m wildfire.train --config configs/resnet18_classifier.yaml
```

Evaluate a trained checkpoint:

```bash
PYTHONPATH=src python3 -m wildfire.evaluate --config configs/resnet18_classifier.yaml --split test
```

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
