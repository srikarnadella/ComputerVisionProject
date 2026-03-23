# Midterm Report Outline

This version reflects the backup dataset that is currently available: a binary image classification dataset with `wildfire` and `nowildfire` labels.

## 1. Introduction

State the problem as wildfire image recognition from satellite imagery.

Key points:

- Wildfire monitoring matters for disaster response and environmental analysis.
- Image classification is a tractable first step for automated wildfire recognition.
- The project pivoted from the original primary dataset to the backup dataset because the primary benchmark was not feasible within the course timeline.

Suggested thesis statement:

> We study deep learning methods for binary wildfire recognition from satellite imagery and evaluate whether stronger convolutional classifiers improve scene-level detection accuracy over simpler baselines.

## 2. Background and Related Work

Cover:

- CNNs for remote sensing image classification
- Transfer learning with residual networks
- Wildfire detection from satellite imagery
- Why scene-level classification is still operationally useful

## 3. Methods

Describe:

- Dataset splits: `train`, `valid`, `test`
- Small CNN baseline
- ResNet-18 baseline
- Cross-entropy loss and AdamW optimization
- Accuracy, precision, recall, and F1

## 4. Experiments

Main questions:

1. How strong is a simple CNN baseline?
2. Does ResNet-18 improve wildfire recall and F1?
3. What visual patterns cause false positives and false negatives?

Minimum deliverables:

- One metric table
- One confusion matrix
- One qualitative predictions figure

## 5. Conclusion and Future Work

State:

- The project pivoted to the backup dataset
- The current pipeline gives a reproducible classification benchmark
- Future work can add transfer learning, ablations, and richer wildfire datasets

