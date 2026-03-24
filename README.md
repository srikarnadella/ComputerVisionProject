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
- `per_class_<split>.png` (precision/recall/F1 bars)
- `pr_curve_<split>.png`, `roc_curve_<split>.png`

## Midterm Snapshot (compute-limited runs)

These numbers come from short, capped training runs on CPU; use them as provisional baselines in the midterm report and note the limitations.

- **SmallCNN (full test pass)**  
  - Loss 0.1580, Accuracy 0.9454, Precision 0.5530, Recall 0.5222, F1 0.5368  
  - Artifacts: `outputs/wildfire_small_cnn/` (`eval_test.json`, `classification_report_test.json`, `confusion_test.png`, `qualitative_test.png`, `per_class_test.png`, `pr_curve_test.png`, `roc_curve_test.png`)

- **ResNet-18 (capped, 2 epochs, CPU)**  
  - Loss 0.1528, Accuracy 0.9468, Precision 0.5530, Recall 0.5149, F1 0.5327  
  - Artifacts: `outputs/wildfire_resnet18_baseline/` (same filenames as above)

### Caveats to highlight in the report
- Training was capped (`--max-train-batches 400` / `--max-val-batches 197`) and limited to 2–3 epochs on CPU; metrics are lower than expected.
- No class-weighted loss or longer epochs yet; wildfire recall should improve with more training and/or GPU.
- Image size was 224 for these runs; a fast preset (160px, pretrained) exists in `configs/resnet18_fast.yaml` for quick iteration.

### Next steps for the final report
- Run pretrained ResNet-18 without caps (or with higher caps) on GPU.
- Try class-weighted cross-entropy to lift wildfire recall.
- Regenerate eval artifacts (confusion, PR/ROC, qualitative) after full runs.

## Current Evaluation (full test pass, best checkpoints)

All metrics below are on the held-out **test** split (6,300 images: 3,480 wildfire / 2,820 nowildfire), using CPU-trained models with short epochs. Paths to artifacts are included.

**SmallCNN** (`outputs/wildfire_small_cnn/`)
- Overall (positive-class view printed by the script): loss 0.1580, acc 0.9454, precision 0.5530, recall 0.5222, F1 0.5368  
  *Note*: these “overall” precision/recall values are the wildfire class only; see the per-class report for the full picture.
- Per-class (from `classification_report_test.json`): nowildfire P/R/F1 = 0.933/0.946/0.939; wildfire P/R/F1 = 0.956/0.945/0.950; macro F1 ≈ 0.945
- Artifacts: `eval_test.json`, `classification_report_test.json`, `confusion_test.png`, `qualitative_test.png`, `per_class_test.png`, `pr_curve_test.png`, `roc_curve_test.png`

**ResNet-18** (`outputs/wildfire_resnet18_baseline/`)
- Overall (wildfire-only view): loss 0.1528, acc 0.9468, precision 0.5530, recall 0.5149, F1 0.5327
- Per-class: nowildfire P/R/F1 = 0.919/0.966/0.942; wildfire P/R/F1 = 0.971/0.931/0.951; macro F1 ≈ 0.946
- Artifacts: `eval_test.json`, `classification_report_test.json`, `confusion_test.png`, `qualitative_test.png`, `per_class_test.png`, `pr_curve_test.png`, `roc_curve_test.png`

Interpretation (midterm):
- Both models achieve ~0.94–0.95 accuracy; class-level F1 is higher (~0.94–0.95) than the macro numbers due to averaging differences.
- Under the capped/short training, ResNet-18 did not yet outperform SmallCNN on macro F1; likely limited by few epochs and batch caps. Expect gains with longer training, GPU, or class-weighted loss.

## Report-Ready Methods (concise)

- **Task**: binary scene classification on satellite tiles (`wildfire`, `nowildfire`).  
- **Data layout**: `data/{train,valid,test}/{wildfire,nowildfire}`; test split = 6,300 images.  
- **Transforms**: resize to 224 px (160 px for fast runs); train augments (flip, ±10° rotation); ImageNet mean/std normalization.  
- **Models**:  
  - SmallCNN: 3×(conv+ReLU+max-pool) → GAP → dropout → linear(128→2).  
  - ResNet‑18: torchvision backbone, dropout+linear head; `pretrained` flag optional.  
- **Optimization**: cross-entropy + AdamW (lr 1e‑3 SmallCNN, 3e‑4 ResNet; wd 1e‑4); best checkpoint by val F1.  
- **Controls for speed**: `--max-train-batches`, `--max-val-batches`, `--epochs`, smaller `data.image_size`; `--device cuda` or set in config.  
- **Eval outputs**: overall metrics JSON, per-class report JSON, confusion matrix, per-class bar chart, PR curve (AP), ROC curve (AUC), qualitative grid.

## Report-Ready Experiments & Questions
- Q1: How strong is a lightweight CNN baseline on this dataset?  
- Q2: Does adding residual depth (and pretraining) improve wildfire recall/F1?  
- Q3: What error modes dominate? (Use confusion + qualitative grids + PR/ROC.)  
- Testbed: CPU short runs with batch caps; same code supports GPU for full runs.  
- Current answers: both models ~0.95 per-class F1; capped training limited the ResNet lift—expect improvement with longer/GPU runs and class weighting.

## Conclusion & Future Work (midterm)
- Pipeline is reproducible; figures/tables are ready for the midterm.  
- Limitations: capped batches, few epochs, CPU-only → undertrained ResNet.  
- Final report plan: full/uncapped pretrained ResNet on GPU, class-weighted loss, possibly 224 px, refreshed PR/ROC/confusion/qualitative figures, deeper error analysis by scene type.

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
