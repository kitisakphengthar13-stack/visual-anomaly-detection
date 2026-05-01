# Training Pipeline

This document explains the training workflow for the PatchCore visual anomaly detection project.

The training script is:

```text
training/patchcore_training.py
```

## Overview

The script performs the following steps:

1. Read command-line arguments.
2. Load the MVTec AD dataset through Anomalib.
3. Build a PatchCore anomaly detection model.
4. Create an Anomalib training engine.
5. Train the model.
6. Evaluate the trained model on the test set.

## Command-Line Arguments

The training script uses command-line arguments so dataset paths, output paths, and training settings can be changed without editing source code.

| Argument | Default | Description |
|---|---|---|
| `--data-root` | `path/to/mvtec_ad` | Root directory containing MVTec AD categories |
| `--category` | `cable` | MVTec AD category to train and evaluate |
| `--output-dir` | `path/to/results` | Directory for Anomalib outputs, logs, and checkpoints |
| `--image-size` | `256` | Input image size reference documented for the project |
| `--train-batch-size` | `32` | Training batch size |
| `--eval-batch-size` | `32` | Evaluation batch size |
| `--num-workers` | `4` | Number of dataloader workers |
| `--coreset-ratio` | `1.0` | PatchCore coreset sampling ratio |

Example command:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --category cable --output-dir path/to/results
```

Note: `--image-size` is currently exposed as a documented argument but is not passed into the Anomalib datamodule by the script. This preserves the current training behavior.

## Dataset

The project uses the MVTec AD `cable` category. The dataset is loaded with Anomalib's `MVTecAD` datamodule:

```python
from anomalib.data import MVTecAD
```

Expected dataset structure:

```text
path/to/mvtec_ad/
`-- cable/
    |-- train/
    |-- test/
    `-- ground_truth/
```

The dataset root can be changed with:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad
```

The category can be changed with:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --category cable
```

The MVTec AD dataset is not included in this repository.

## Model

The model used in this project is PatchCore:

```python
from anomalib.models import Patchcore
```

Model settings:

| Setting | Value |
|---|---|
| Backbone | `wide_resnet50_2` |
| Feature layers | `layer2`, `layer3` |
| Pretrained backbone | `True` |
| Coreset sampling ratio | `1.0` |
| Number of neighbors | `9` |

## Training Engine

Training and evaluation are handled by Anomalib's `Engine`:

```python
from anomalib.engine import Engine
```

Engine settings:

```text
accelerator = auto
devices = 1
```

This allows Anomalib to select an available supported accelerator depending on the environment.

## Output Directory

Training results are saved under the directory passed to:

```bash
--output-dir path/to/results
```

The output directory may contain:

- Model checkpoints
- Logs
- Evaluation outputs
- Anomalib-generated result files

These generated files are not committed because they can be large and environment-specific.

## Training Workflow

```text
MVTec AD Cable dataset
    -> Anomalib MVTecAD datamodule
    -> PatchCore model
    -> Anomalib Engine
    -> Training / memory bank construction
    -> Evaluation on test set
    -> Checkpoint and result outputs
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

## How to Run Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training with default example paths:

```bash
python training/patchcore_training.py
```

Run training with local dataset and output paths:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --category cable --output-dir path/to/results
```

Run training with custom batch sizes:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --train-batch-size 32 --eval-batch-size 32
```

Run training with a custom coreset sampling ratio:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --coreset-ratio 1.0
```

## Evaluation Results

Project evaluation results on the MVTec AD Cable category:

| Metric | Result |
|---|---:|
| Image AUROC | 99% |
| Image F1-score | 97% |
| Pixel AUROC | 98% |
| Pixel F1-score | 64% |

## Reproducibility Notes

Results depend on the Anomalib version, PyTorch version, CUDA setup, dataset split, checkpoint, threshold, preprocessing, and environment. Record these details when comparing results across machines or runs.

## Notes

The script uses example path defaults such as `path/to/mvtec_ad`. Replace them with actual local paths before running.

The MVTec AD dataset and model checkpoints are not included in this repository. Training outputs such as checkpoints, logs, and generated result folders are ignored by Git because they can be large and environment-specific.
