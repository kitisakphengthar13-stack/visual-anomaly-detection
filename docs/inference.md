# Inference Pipeline

This document explains the inference pipeline used in this visual anomaly detection project.

The inference script loads a trained PatchCore checkpoint, predicts anomaly scores and anomaly maps for test images, classifies each image as OK or NG using a threshold, and visualizes the result using an original image, heatmap, and overlay.

## Overview

The inference script is located at:

```text
inference/inference.py
```

The script performs the following steps:

1. Read command-line arguments
2. Load test images from an input folder
3. Load a trained PatchCore checkpoint
4. Create a prediction dataset using Anomalib
5. Run inference using Anomalib's `Engine`
6. Extract anomaly scores and anomaly maps
7. Classify each image as OK or NG using a threshold
8. Generate heatmap and overlay visualization
9. Display results with keyboard navigation

## Command-Line Arguments

The inference script uses command-line arguments so that paths do not need to be hard-coded in the script.

Default values:

|   Argument    |      Default        |                   Description                    |
|---------------|---------------------|--------------------------------------------------|
|   `--ckpt`    | `models/model.ckpt` |  Path to the trained PatchCore checkpoint file   |
|   `--input`   | `data/test_images`  |    Path to the folder containing input images    |
| `--threshold` |       `0.50`        | Anomaly score threshold for OK/NG classification |
|    `--ext`    |       `.png`        |           Image file extension to load           |

The arguments are defined using Python's `argparse` module:

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PatchCore inference for visual anomaly detection."
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/model.ckpt",
        help="Path to the trained PatchCore checkpoint file.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/test_images",
        help="Path to the folder containing input images.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Anomaly score threshold for OK/NG classification.",
    )

    parser.add_argument(
        "--ext",
        type=str,
        default=".png",
        help="Image file extension to load, for example .png, .jpg, or .jpeg.",
    )

    return parser.parse_args()
```

## Input Images

By default, the script reads images from:

```text
data/test_images
```

The input folder can be changed using the `--input` argument:

```bash
python inference/inference.py --input data/test_images
```

The script loads images based on the selected file extension. By default, it loads `.png` images.

```bash
python inference/inference.py --input data/test_images --ext .png
```

For `.jpg` images:

```bash
python inference/inference.py --input data/test_images --ext .jpg
```

The script checks whether the input folder exists and whether images with the selected extension are available.

```python
def load_image_paths(folder_path: Path, image_extension: str):
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {folder_path}")

    if not image_extension.startswith("."):
        image_extension = f".{image_extension}"

    image_paths = sorted(folder_path.glob(f"*{image_extension}"))

    if not image_paths:
        raise FileNotFoundError(
            f"No {image_extension} images found in: {folder_path}"
        )

    return image_paths
```

## Model Checkpoint

By default, the script expects the trained PatchCore checkpoint at:

```text
models/model.ckpt
```

The checkpoint path can be changed using the `--ckpt` argument:

```bash
python inference/inference.py --ckpt models/model.ckpt
```

The script checks whether the checkpoint file exists before running inference:

```python
if not ckpt_path.exists():
    raise FileNotFoundError(
        f"Checkpoint file not found: {ckpt_path}\n"
        "Place your trained checkpoint in the models/ directory or pass it with --ckpt."
    )
```

Model checkpoint files are not included in this repository due to file size.

## Model

The inference script uses PatchCore from Anomalib:

```python
from anomalib.models import Patchcore
```

The model is initialized with a coreset sampling ratio of `1.0`:

```python
def build_model():
    return Patchcore(
        coreset_sampling_ratio=1.0,
    )
```

PatchCore is used to generate:

- Image-level anomaly scores
- Pixel-level anomaly maps
- Heatmap visualization for anomaly localization

## Prediction Dataset

The script uses Anomalib's `PredictDataset` for inference:

```python
from anomalib.data import PredictDataset
```

The prediction dataset is created from the input image folder:

```python
dataset = PredictDataset(
    path=folder_path,
)
```

## Inference Engine

Inference is handled by Anomalib's `Engine`:

```python
from anomalib.engine import Engine
```

The engine automatically uses GPU acceleration if CUDA is available. Otherwise, it falls back to CPU.

```python
def build_engine():
    return Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )
```

The prediction step is executed using:

```python
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=str(ckpt_path),
)
```

If no predictions are returned, the script raises an error:

```python
if predictions is None:
    raise RuntimeError("No predictions returned.")
```

## OK/NG Classification

Each image receives an anomaly score from the model.

The default threshold is `0.50`, but it can be changed using the `--threshold` argument:

```bash
python inference/inference.py --threshold 0.5
```

The decision rule is:

|       Condition       | Prediction |
|-----------------------|------------|
| Anomaly score < 0.50  |     OK     |
| Anomaly score >= 0.50 |     NG     |

The classification logic in the script is:

```python
status = "NG" if score >= threshold else "OK"
```

The threshold can be adjusted depending on the inspection requirement.

A lower threshold increases sensitivity to defects but may increase false positives. A higher threshold reduces false positives but may miss subtle defects.

## Anomaly Map Processing

The script extracts the anomaly map from each prediction batch:

```python
anomaly_maps = batch.anomaly_map
```

Since Anomalib predictions may be returned as tensors or arrays, the script includes helper functions to safely convert and extract data.

### Convert tensors to NumPy

```python
def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)
```

### Extract one item from a batch

```python
def extract_batch_item(array, index, batch_size):
    array = to_numpy(array)

    if batch_size == 1:
        return np.squeeze(array)

    if array.ndim >= 3:
        return np.squeeze(array[index])

    return np.squeeze(array)
```

After extraction, the anomaly map is resized to match the original image size:

```python
anomaly_map = cv2.resize(
    anomaly_map,
    (img_rgb.shape[1], img_rgb.shape[0]),
)
```

A Gaussian blur is applied to make the heatmap smoother for visualization:

```python
anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), 2.0)
```

The heatmap is then normalized to the range `0` to `1`:

```python
def normalize_heatmap(anomaly_map):
    heatmap = anomaly_map - anomaly_map.min()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap
```

## Image Loading

Images are loaded using OpenCV:

```python
img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
```

If an image cannot be read, the script skips it:

```python
if img_bgr is None:
    print(f"Cannot read image: {image_path}")
    continue
```

Since OpenCV loads images in BGR format, the image is converted to RGB before visualization with Matplotlib:

```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

## Console Output

For each valid image, the script prints:

- Image name
- Anomaly score
- OK/NG status

Example output format:

```text
============================================================
image : sample_image.png
score : 0.734521
status: NG
```

This makes it easy to inspect the numerical anomaly score and the final classification result.

## Visualization

The script visualizes each result using Matplotlib with three panels:

1. Original image
2. Anomaly heatmap
3. Heatmap overlay on the original image

The visualization layout is created with:

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
```

### Original Image

```python
axes[0].set_title(f"Original: {item['name']}")
axes[0].imshow(item["img"])
```

### Heatmap

```python
axes[1].set_title("Heatmap")
axes[1].imshow(item["heatmap"], cmap="jet", vmin=0, vmax=1)
```

### Overlay

```python
axes[2].set_title(
    f"Overlay | score={item['score']:.4f} | status={item['status']}"
)
axes[2].imshow(item["img"])
axes[2].imshow(item["heatmap"], cmap="jet", alpha=0.35, vmin=0, vmax=1)
```

The overlay helps show where the model identifies suspicious regions on the workpiece.

## Keyboard Navigation

The visualization window supports keyboard navigation:

|     Key     |        Action       |
|-------------|---------------------|
| Right arrow |   Show next image   | 
| Left arrow  | Show previous image |
|     Esc     |     Close window    |

The keyboard event handler is implemented as:

```python
def on_key(event):
    nonlocal idx

    if event.key == "right":
        idx = (idx + 1) % len(results)
        draw(idx)

    elif event.key == "left":
        idx = (idx - 1) % len(results)
        draw(idx)

    elif event.key == "escape":
        plt.close(fig)
```

This makes the inference demo easier to inspect when testing multiple images.

## How to Run Inference

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place your trained checkpoint file at:

```text
models/model.ckpt
```

Place test images in:

```text
data/test_images
```

Run inference with the default settings:

```bash
python inference/inference.py
```

Run inference with custom paths:

```bash
python inference/inference.py --ckpt models/model.ckpt --input data/test_images --threshold 0.5
```

Run inference on `.jpg` images:

```bash
python inference/inference.py --ckpt models/model.ckpt --input data/test_images --ext .jpg
```

The script will display the result window and print the score and OK/NG status for each image in the terminal.

## Notes

The inference script does not use machine-specific hard-coded paths. Checkpoint path, input folder, threshold, and image extension can be changed through command-line arguments.

The model checkpoint file is not included in this repository due to file size.

The inference script is designed for demonstration and inspection workflow understanding. It shows both the final OK/NG decision and the anomaly localization heatmap.