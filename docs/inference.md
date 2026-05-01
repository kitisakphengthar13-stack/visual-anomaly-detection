# Inference Pipeline

This document explains the inference workflow for the PatchCore visual anomaly detection project.

The inference script loads a trained PatchCore checkpoint, predicts anomaly scores and anomaly maps for input images, classifies each image as OK or NG using a threshold, and saves visualization results.

## Overview

The inference script is:

```text
inference/inference.py
```

The script performs the following steps:

1. Read command-line arguments.
2. Load an input image or folder.
3. Load a trained PatchCore checkpoint.
4. Create a prediction dataset using Anomalib.
5. Run inference using Anomalib's `Engine`.
6. Extract anomaly scores and anomaly maps.
7. Classify each image as OK or NG using a threshold.
8. Generate heatmap overlay visualization.
9. Save visualization outputs to an output directory.

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | `path/to/input_image_or_folder` | Path to an input image or folder |
| `--ckpt` | `path/to/model.ckpt` | Path to the trained PatchCore checkpoint file |
| `--output-dir` | `path/to/output_dir` | Directory for saved inference visualizations |
| `--threshold` | `0.5` | Anomaly score threshold for OK/NG classification |

Example command:

```bash
python inference/inference.py --ckpt path/to/model.ckpt --input path/to/input_image_or_folder --output-dir path/to/output_dir --threshold 0.5
```

## Input Images

The script accepts an input image or folder through the `--input` argument:

```bash
python inference/inference.py --input path/to/input_image_or_folder
```

The input path must exist. If the input path is invalid, the script raises an error.

## Model Checkpoint

The trained PatchCore checkpoint is passed through the `--ckpt` argument:

```bash
python inference/inference.py --ckpt path/to/model.ckpt
```

Checkpoint files are not included in this repository due to file size. Generate one through training or copy an existing compatible checkpoint locally before running inference.

## Model

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
| Coreset sampling ratio | `1.0` |
| Number of neighbors | `9` |

PatchCore generates:

- Image-level anomaly scores
- Pixel-level anomaly maps
- Heatmap visualizations for anomaly localization

## Prediction Dataset

The script uses Anomalib's `PredictDataset`:

```python
from anomalib.data import PredictDataset
```

The prediction dataset is created from the input path:

```python
dataset = PredictDataset(
    path=input_path,
)
```

## Inference Engine

Inference is handled by Anomalib's `Engine`:

```python
from anomalib.engine import Engine
```

Engine settings:

```text
accelerator = auto
devices = 1
```

Prediction call:

```python
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=ckpt_path,
)
```

## OK/NG Classification

Each image receives an anomaly score from the model. The default threshold is `0.5`, and it can be changed with:

```bash
python inference/inference.py --threshold 0.5
```

Decision rule:

| Condition | Prediction |
|---|---|
| Anomaly score < 0.50 | OK |
| Anomaly score >= 0.50 | NG |

The classification logic is:

```python
label = "NG" if score >= args.threshold else "OK"
```

A lower threshold increases sensitivity to defects but may increase false positives. A higher threshold reduces false positives but may miss subtle defects.

## Visualization

The script extracts the anomaly map from each prediction result and converts it into a heatmap overlay.

```text
Input image
    -> PatchCore prediction
    -> Anomaly score
    -> Anomaly map
    -> Heatmap normalization
    -> Heatmap overlay
    -> Saved result image
```

The saved result includes:

- Input image
- Heatmap overlay
- Prediction label
- Anomaly score

## Console Output

For each valid image, the script prints:

- Image name
- Anomaly score
- OK/NG prediction

Example:

```text
[RESULT] sample_image.png | Score: 0.7345 | Prediction: NG
```

## Output Directory

Inference visualizations are saved to the directory specified by:

```bash
--output-dir path/to/output_dir
```

Example:

```bash
python inference/inference.py --ckpt path/to/model.ckpt --input path/to/input_image_or_folder --output-dir path/to/output_dir
```

Generated outputs are not committed because they may vary by run and can become large.

## How to Run Inference

Install dependencies:

```bash
pip install -r requirements.txt
```

Run inference with default example paths:

```bash
python inference/inference.py
```

Run inference with local paths:

```bash
python inference/inference.py --ckpt path/to/model.ckpt --input path/to/input_image_or_folder --output-dir path/to/output_dir --threshold 0.5
```

## Reproducibility Notes

Inference results depend on the checkpoint, Anomalib version, PyTorch version, threshold, preprocessing, image inputs, and environment. Use the same checkpoint and threshold when comparing output images or metrics.

## Notes

The script uses example path defaults such as `path/to/model.ckpt`. Replace them with actual local paths before running.

The inference script demonstrates the OK/NG decision and anomaly localization workflow; it is not presented as a production inspection application.
