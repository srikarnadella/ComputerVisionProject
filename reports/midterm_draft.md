# Midterm Draft

## Introduction

Wildfires create severe environmental, economic, and public-safety risks, which makes reliable wildfire monitoring an important computer vision problem. Satellite imagery is especially useful for this setting because it enables large-scale coverage across regions that are difficult to monitor from the ground. A practical first step in automated wildfire monitoring is scene-level recognition: determining whether an image patch contains wildfire activity or not. Although this task is simpler than full segmentation or temporal forecasting, it is still valuable for prioritizing human review and supporting downstream monitoring pipelines.

Our original proposal centered on multi-temporal wildfire segmentation using the TS-SatFire benchmark. However, that dataset proved impractical for the current project timeline due to its scale and data-management overhead. Because our proposal explicitly included a backup dataset, we pivoted to a large binary wildfire image classification benchmark with `wildfire` and `nowildfire` classes. This pivot preserves the core application domain while giving us a tractable way to implement, evaluate, and analyze deep learning models within the midterm deadline.

The goal of the current midterm project is to build a reproducible deep learning pipeline for wildfire recognition from satellite imagery and compare a simple convolutional baseline against a stronger residual network. Our working hypothesis is that a deeper convolutional architecture will achieve better class separation, especially in difficult scenes where non-fire patterns resemble smoke, haze, or bright land cover.

## Background and Related Work

Image classification with convolutional neural networks remains a strong baseline for remote sensing tasks. In many earth observation settings, CNNs perform well because they can capture local texture, color, and spatial structure while remaining efficient to train. Transfer learning from large-scale natural-image datasets has also become common practice, especially when the target task has limited annotation diversity relative to model capacity.

Wildfire detection from imagery has been studied using both classical image processing and deep learning approaches. More recent work has expanded from scene classification to localization, segmentation, and temporal prediction. However, scene-level classification is still operationally relevant because it can serve as a screening step before more expensive or specialized models are applied. In our case, the backup dataset supports only binary labels, so classification is the appropriate and honest problem formulation for the current report.

This framing also gives us a clean experimental question: how much performance is gained by moving from a lightweight CNN baseline to a deeper ResNet-style model under the same train/validation/test split? Answering that question provides a solid midterm milestone and establishes a reproducible benchmark that can be extended later if richer wildfire datasets become available.

## Methods

We formulate the task as binary image classification. Each input is an RGB satellite image patch labeled as either `wildfire` or `nowildfire`. The dataset is organized into predefined `train`, `valid`, and `test` splits, which allows us to train models on one subset, tune on validation data, and reserve a final split for held-out evaluation.

We compare two model families. The first is a small custom convolutional neural network designed to serve as a lightweight baseline. This model is intentionally simple so that it provides a clear reference point for how much performance can be obtained without a deep backbone. The second is a ResNet-18 classifier, which represents a stronger convolutional architecture with residual connections and greater representational capacity. Both models are trained with cross-entropy loss and optimized with AdamW.

Our implementation uses a shared PyTorch training pipeline with configurable image size, batch size, optimizer settings, and output directories. We evaluate models using accuracy, precision, recall, and F1 score. In addition to aggregate metrics, we plan to inspect qualitative examples of correct and incorrect predictions in order to identify systematic failure modes, such as confusion between wildfire scenes and visually similar non-wildfire patterns.

## Experiments

The midterm experiments are designed to answer three questions. First, how strong is a small CNN baseline on this dataset? Second, does a ResNet-18 model improve classification quality enough to justify its additional complexity? Third, what visual characteristics are associated with false positives and false negatives?

Our minimum experimental plan is straightforward. We will train the small CNN and the ResNet-18 model on the training split, select the better checkpoint using validation performance, and report final metrics on the test split. The key quantitative outputs are accuracy, precision, recall, and F1 score. We will also produce a confusion matrix and a small qualitative panel showing representative wildfire and non-wildfire scenes that the model handles correctly or incorrectly.

This scope is appropriate for the midterm because it emphasizes reproducible implementation and honest evaluation on a dataset that is available now. It also leaves room for stronger final-report analysis, such as transfer learning comparisons, resolution ablations, and error analysis across different scene types.

## Conclusion and Future Work

This midterm project delivers a practical pivot from an initially proposed segmentation benchmark to a tractable image classification benchmark in the same wildfire-monitoring domain. The central contribution at this stage is a reproducible deep learning pipeline and an initial empirical comparison between lightweight and residual CNN architectures for wildfire recognition.

For the final report, we plan to strengthen the study through broader experiments and deeper analysis. In particular, we aim to test whether pretrained feature extractors improve performance, whether image resolution materially affects results, and which kinds of scenes remain difficult for the classifier. If a richer wildfire dataset becomes feasible later, the pipeline and findings from this classification benchmark will still provide a useful foundation.

