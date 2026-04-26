# Results

## Performance

|     Metric     | Result |
|----------------|-------:|
|   Image AUROC  |   99%  |
| Image F1-score |   97%  |
|   Pixel AUROC  |   98%  |
| Pixel F1-score |   64%  |

## Interpretation

The model achieved strong image-level anomaly detection performance, showing that it can separate normal and anomalous cable images effectively.

Pixel-level AUROC was also high, indicating that the anomaly map can rank abnormal pixels well. However, Pixel F1-score was lower because pixel-level F1 depends on the selected threshold and exact overlap between the predicted anomaly mask and the ground-truth defect mask.

This result reflects an important practical point in industrial inspection: OK/NG classification and defect localization may require different threshold settings depending on the inspection objective.