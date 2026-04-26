# Inference Pipeline

This document explains the inference pipeline used in this visual anomaly detection project.

The inference script loads a trained PatchCore checkpoint, predicts anomaly scores and anomaly maps for input images, classifies each image as OK or NG using a threshold, and saves visualization results.

---

## 1. Overview

The inference script is located at:

```text
inference/inference.py
```

The script performs the following steps:

1. Read command-line arguments
2. Load an input image or folder
3. Load a trained PatchCore checkpoint
4. Create a prediction dataset using Anomalib
5. Run inference using Anomalib's `Engine`
6. Extract anomaly scores and anomaly maps
7. Classify each image as OK or NG using a threshold
8. Generate heatmap overlay visualization
9. Save visualization outputs to an output directory

---

## 2. Command-Line Arguments

The inference script uses command-line arguments so that paths do not need to be hard-coded.

Default placeholder values:

| Argument | Default | Description |
|---|---|---|
| `--input` | `path/to/input_image_or_folder` | Path to an input image or folder |
| `--ckpt` | `path/to/model.ckpt` | Path to the trained PatchCore checkpoint file |
| `--output-dir` | `path/to/output_dir` | Directory for saving inference visualizations |
| `--threshold` | `0.5` | Anomaly score threshold for OK/NG classification |

Example command:

```bash
python inference/inference.py --ckpt path/to/model.ckpt --input path/to/input_image_or_folder --output-dir path/to/output_dir --threshold 0.5
```

---

## 3. Input Images

The script accepts an input image or folder through the `--input` argument.

Example:

```bash
python inference/inference.py --input path/to/input_image_or_folder
```

The input path must exist before running inference.

If the input path is invalid, the script raises an error.

---

## 4. Model Checkpoint

The trained PatchCore checkpoint is passed through the `--ckpt` argument.

Example:

```bash
python inference/inference.py --ckpt path/to/model.ckpt
```

Model checkpoint files are not included in this repository due to file size.

The checkpoint file should be generated from training or copied locally before running inference.

---

## 5. Model

The inference script uses PatchCore from Anomalib:

```python
from anomalib.models import Patchcore
```

The model configuration follows the project training setup:

| Setting | Value |
|---|---|
| Backbone | `wide_resnet50_2` |
| Feature layers | `layer2`, `layer3` |
| Pretrained backbone | `True` |
| Coreset sampling ratio | `0.1` |
| Number of neighbors | `9` |

PatchCore is used to generate:

- Image-level anomaly scores
- Pixel-level anomaly maps
- Heatmap visualization for anomaly localization

---

## 6. Prediction Dataset

The script uses Anomalib's `PredictDataset` for inference:

```python
from anomalib.data import PredictDataset
```

The prediction dataset is created from the input path:

```python
dataset = PredictDataset(
    path=input_path,
)
```

---

## 7. Inference Engine

Inference is handled by Anomalib's `Engine`.

```python
from anomalib.engine import Engine
```

The engine is configured with:

```text
accelerator = auto
devices = 1
```

The prediction step is executed using:

```python
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=ckpt_path,
)
```

---

## 8. OK/NG Classification

Each image receives an anomaly score from the model.

The default threshold is `0.5`, but it can be changed using the `--threshold` argument.

```bash
python inference/inference.py --threshold 0.5
```

The decision rule is:

| Condition | Prediction |
|---|---|
| Anomaly score < 0.50 | OK |
| Anomaly score >= 0.50 | NG |

The classification logic is:

```python
label = "NG" if score >= args.threshold else "OK"
```

The threshold can be adjusted depending on the inspection requirement.

A lower threshold increases sensitivity to defects but may increase false positives.  
A higher threshold reduces false positives but may miss subtle defects.

---

## 9. Anomaly Map Visualization

The script extracts the anomaly map from each prediction result and converts it into a heatmap overlay.

The visualization process follows this workflow:

```text
Input Image
    ↓
PatchCore Prediction
    ↓
Anomaly Score
    ↓
Anomaly Map
    ↓
Heatmap Normalization
    ↓
Heatmap Overlay
    ↓
Save Result Image
```

The result visualization includes:

- Input image
- Heatmap overlay
- Prediction label
- Anomaly score

---

## 10. Console Output

For each valid image, the script prints:

- Image name
- Anomaly score
- OK/NG prediction

Example output:

```text
[RESULT] sample_image.png | Score: 0.7345 | Prediction: NG
```

This makes it easy to inspect the numerical anomaly score and the final classification result.

---

## 11. Output Directory

Inference visualizations are saved to the directory specified by:

```bash
--output-dir path/to/output_dir
```

Example:

```bash
python inference/inference.py --ckpt path/to/model.ckpt --input path/to/input_image_or_folder --output-dir path/to/output_dir
```

Generated outputs are not committed to GitHub because they may vary by run and can become large.

---

## 12. How to Run Inference

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run inference with placeholder/default arguments:

```bash
python inference/inference.py
```

Run inference with custom paths:

```bash
python inference/inference.py --ckpt path/to/model.ckpt --input path/to/input_image_or_folder --output-dir path/to/output_dir --threshold 0.5
```

---

## 13. Notes

The inference script uses placeholder paths. Replace all `path/to/...` values with actual local paths before running.

The model checkpoint file is not included in this repository due to file size.

The inference script is designed for demonstration and inspection workflow understanding. It shows both the final OK/NG decision and anomaly localization through heatmap visualization.