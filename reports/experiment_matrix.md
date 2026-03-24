# Experiment Matrix

## Must Have Before Midterm

| Experiment | Purpose | Report Value |
|---|---|---|
| Small CNN baseline | Establish a simple reference model | Baseline row |
| ResNet-18 baseline | Stronger convolutional model | Main comparison row |
| Per-class precision, recall, F1 | Understand wildfire vs non-wildfire errors | Better analysis |
| Qualitative predictions | Show representative successes and failures | Figure material |
| Confusion matrix | Visualize error modes | Figure + discussion |

## Good If Time Allows

| Experiment | Purpose | Report Value |
|---|---|---|
| Pretrained vs non-pretrained ResNet-18 | Measure transfer learning benefit | Better methods comparison |
| Image size ablation | Measure compute/performance tradeoff | Practical insight |
| Class-weighted loss | Address imbalance if needed | Stronger robustness analysis |

## Core Questions

1. Can a lightweight classifier separate wildfire from non-wildfire scenes well?
2. Does a stronger CNN improve wildfire recall?
3. What scene types remain difficult?

## Current Status (midterm)

- SmallCNN: trained with batch caps; val accuracy ~0.93, wildfire F1 ~0.52 (needs longer/full run).
- ResNet-18 (pretrained fast config): trained with caps; val accuracy ~0.97, wildfire F1 ~0.40–0.41 (compute-limited).
- Next runs for final report: remove caps or raise them, keep pretrained ResNet-18, consider class-weighted loss and full image size.

