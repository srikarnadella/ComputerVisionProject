# Report Claims Checklist

## Safe Claims Once The Pipeline Runs

- We implemented a reproducible PyTorch pipeline for binary wildfire image classification.
- The code supports both a lightweight CNN baseline and a ResNet-style classifier.
- We evaluate the models using accuracy, precision, recall, and F1 score.

## Safe Claims After First Results

- The backup dataset is large enough to support a meaningful benchmark (6,300 test images).
- Both SmallCNN and ResNet-18 achieve ~0.94–0.95 test accuracy with P/R/F1 ≈0.94–0.95 per class (short, capped training).
- Deeper convolutional models provide a useful comparison against a simpler baseline, but gains require longer or GPU training.

## Claims That Need Evidence

- The classifier is robust across geographies or seasons.
- Transfer learning consistently improves performance.
- The model generalizes to richer wildfire localization datasets.

