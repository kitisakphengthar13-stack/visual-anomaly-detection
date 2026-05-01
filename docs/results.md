# Results

This document summarizes the project evaluation results for Visual Anomaly Detection with PatchCore on the MVTec AD Cable dataset.

The project evaluates both image-level anomaly detection and pixel-level anomaly localization.

## Performance Summary

| Metric | Result |
|---|---:|
| Image AUROC | 99% |
| Image F1-score | 97% |
| Pixel AUROC | 98% |
| Pixel F1-score | 64% |

These values are reported as project evaluation results on the MVTec AD Cable category.

## Reproducibility Notes

Results may vary depending on:

- Anomalib version
- PyTorch and CUDA versions
- Dataset split and local dataset contents
- Checkpoint used for evaluation or inference
- Threshold selection
- Preprocessing and transforms from the installed Anomalib version
- Hardware, operating system, and dependency environment

Record these details when reproducing or comparing results.

## Example Outputs

The following examples show OK and NG prediction results with anomaly heatmap visualization.

### OK Samples

![OK Sample Result 1](assets/ok_sample_result_1.png)

![OK Sample Result 2](assets/ok_sample_result_2.png)

### NG Samples

![NG Sample Result 1](assets/ng_sample_result_1.png)

![NG Sample Result 2](assets/ng_sample_result_2.png)

## Image-Level Performance

Image-level metrics evaluate whether the model can classify an input image as normal or anomalous.

| Metric | Meaning |
|---|---|
| Image AUROC | Measures how well the model separates normal and anomalous images across thresholds |
| Image F1-score | Measures the balance between precision and recall for OK/NG classification |

The model achieved **99% Image AUROC** and **97% Image F1-score** in this project evaluation.

## Pixel-Level Performance

Pixel-level metrics evaluate how well the model localizes anomalous regions in the image.

| Metric | Meaning |
|---|---|
| Pixel AUROC | Measures how well anomaly scores rank abnormal pixels across thresholds |
| Pixel F1-score | Measures overlap between predicted anomaly regions and ground-truth defect masks at a selected threshold |

The model achieved **98% Pixel AUROC**, showing that the anomaly map can rank abnormal pixels well.

The **Pixel F1-score was 64%**, which is lower than the pixel-level AUROC. This is common in anomaly localization because Pixel F1-score depends strongly on the selected threshold and the exact overlap between the predicted anomaly mask and the ground-truth defect mask.

## OK/NG Threshold Decision

For demonstration, an anomaly score threshold of `0.5` is used to classify each image.

| Condition | Prediction |
|---|---|
| Anomaly score < 0.50 | OK |
| Anomaly score >= 0.50 | NG |

The threshold can be adjusted depending on the inspection requirement. A lower threshold increases sensitivity to defects but may increase false positives. A higher threshold reduces false positives but may miss subtle defects.

## Interpretation

The evaluation results show strong image-level OK/NG classification on the MVTec AD Cable category for this project setup.

Pixel-level localization also shows strong ranking performance through Pixel AUROC, while the lower Pixel F1-score shows that threshold selection matters when converting anomaly maps into binary defect masks.

Practical takeaway:

```text
Image-level OK/NG classification and pixel-level defect localization may require different threshold settings.
```

For real inspection use cases, threshold selection should consider the cost of false positives and false negatives.

## Notes

The model checkpoint, dataset, and generated result folders are not included in this repository.

The sample output images in `docs/assets/` are included to demonstrate the anomaly heatmap and overlay visualization workflow.
