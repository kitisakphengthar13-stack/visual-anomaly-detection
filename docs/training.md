# Training Pipeline

This document explains the training pipeline used in this visual anomaly detection project.

The project uses PatchCore from Anomalib to train and evaluate a visual anomaly detection model on the MVTec AD Cable dataset.

---

## 1. Overview

The training script is located at:

```text
training/patchcore_training.py
```

The script performs the following steps:

1. Read command-line arguments
2. Load the MVTec AD dataset
3. Build a PatchCore anomaly detection model
4. Create an Anomalib training engine
5. Train the model
6. Evaluate the trained model on the test set

---

## 2. Command-Line Arguments

The training script uses command-line arguments so that dataset paths, output paths, and training settings do not need to be hard-coded.

Default placeholder values:

| Argument | Default | Description |
|---|---|---|
| `--data-root` | `path/to/mvtec_ad` | Root directory of the MVTec AD dataset |
| `--category` | `cable` | MVTec AD category to train and evaluate |
| `--output-dir` | `path/to/results` | Directory for training results and checkpoints |
| `--image-size` | `256` | Input image size reference |
| `--train-batch-size` | `32` | Training batch size |
| `--eval-batch-size` | `32` | Evaluation batch size |
| `--num-workers` | `4` | Number of dataloader workers |
| `--coreset-ratio` | `1.0` | PatchCore coreset sampling ratio |

Example command:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --category cable --output-dir path/to/results
```

---

## 3. Dataset

The project uses the MVTec AD dataset with the `cable` category.

The dataset is loaded using Anomalib's `MVTecAD` datamodule.

```python
from anomalib.data import MVTecAD
```

Expected dataset structure:

```text
path/to/mvtec_ad/
└── cable/
    ├── train/
    ├── test/
    └── ground_truth/
```

The dataset root can be changed using the `--data-root` argument:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad
```

The category can be changed using the `--category` argument:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --category cable
```

The MVTec AD dataset is not included in this repository. Download it separately and update the dataset path before running training.

---

## 4. Model

The model used in this project is PatchCore.

```python
from anomalib.models import Patchcore
```

PatchCore is used for:

- Image-level anomaly detection
- OK/NG classification
- Pixel-level anomaly localization
- Anomaly heatmap generation

The model configuration used in this project includes:

| Setting | Value |
|---|---|
| Backbone | `wide_resnet50_2` |
| Feature layers | `layer2`, `layer3` |
| Pretrained backbone | `True` |
| Coreset sampling ratio | `1.0` |
| Number of neighbors | `9` |

---

## 5. Training Engine

The training and evaluation process is handled by Anomalib's `Engine`.

```python
from anomalib.engine import Engine
```

The engine is configured with:

```text
accelerator = auto
devices = 1
```

This allows Anomalib to select an available supported accelerator depending on the environment.

---

## 6. Output Directory

Training results are saved under the output directory defined by:

```bash
--output-dir path/to/results
```

The output directory may contain:

- Model checkpoints
- Logs
- Evaluation outputs
- Anomalib-generated result files

These output files are not committed to GitHub because they can be large and environment-specific.

---

## 7. Training Process

The main training process follows this workflow:

```text
MVTec AD Cable Dataset
    ↓
Anomalib MVTecAD Datamodule
    ↓
PatchCore Model
    ↓
Anomalib Engine
    ↓
Training / Feature Memory Bank Construction
    ↓
Evaluation on Test Set
    ↓
Checkpoint and Result Outputs
```

The main engine calls are:

```python
engine.fit(
    model=model,
    datamodule=datamodule,
)

engine.test(
    model=model,
    datamodule=datamodule,
)
```

---

## 8. How to Run Training

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Prepare the MVTec AD dataset using this structure:

```text
path/to/mvtec_ad/
└── cable/
    ├── train/
    ├── test/
    └── ground_truth/
```

Run training with placeholder/default arguments:

```bash
python training/patchcore_training.py
```

Run training with custom dataset and output paths:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --category cable --output-dir path/to/results
```

Run training with a custom batch size:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --train-batch-size 32 --eval-batch-size 32
```

Run training with a custom coreset sampling ratio:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --coreset-ratio 1.0
```

---

## 9. Evaluation Results

The project evaluation produced the following results:

| Metric | Result |
|---|---:|
| Image AUROC | 99% |
| Image F1-score | 97% |
| Pixel AUROC | 98% |
| Pixel F1-score | 64% |

These values are reported as project evaluation results on the MVTec AD Cable category.

---

## 10. Notes

The training script uses placeholder paths. Replace all `path/to/...` values with actual local paths before running.

The MVTec AD dataset and model checkpoint files are not included in this repository.

Training outputs such as checkpoints, logs, and generated result folders are ignored by Git because they can be large and environment-specific.

The purpose of this training script is to demonstrate the PatchCore training and evaluation workflow for industrial visual anomaly detection using Anomalib.