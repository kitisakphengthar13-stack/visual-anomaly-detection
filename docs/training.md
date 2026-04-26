# Training Pipeline

This document explains the training pipeline used in this visual anomaly detection project.

The project uses PatchCore from Anomalib to train and evaluate a visual anomaly detection model on the MVTec AD Cable dataset.

## Overview

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
7. Print the test results

## Command-Line Arguments

The training script uses command-line arguments so that dataset paths, output paths, and training settings do not need to be hard-coded in the script.

Default values:

|       Argument       |   Default   |                  Description                   |
|----------------------|-------------|------------------------------------------------|
|    `--data-root`     | `data/mvtec`|     Root directory of the MVTec AD dataset     |
|    `--category`      |   `cable`   |     MVTec AD category to train and evaluate    |
|   `--output-dir`     |  `results`  | Directory for training outputs and checkpoints |
| `--train-batch-size` |    `12`     |               Training batch size              |
| `--eval-batch-size`  |    `12`     |              Evaluation batch size             |
|   `--num-workers`    |     `3`     |          Number of dataloader workers          |
|  `--coreset-ratio`   |    `1.0`    |        PatchCore coreset sampling ratio        |

Example command:

```bash
python training/patchcore_training.py --data-root data/mvtec --category cable --output-dir results
```

## Dataset

The project uses the MVTec AD dataset with the `cable` category.

In the training script, the dataset is loaded using Anomalib's `MVTecAD` datamodule.

```python
from anomalib.data import MVTecAD
```

The default dataset root is:

```text
data/mvtec
```

Expected dataset structure:

```text
data/
└── mvtec/
    └── cable/
        ├── train/
        ├── test/
        └── ground_truth/
```

The dataset root can be changed using the `--data-root` argument:

```bash
python training/patchcore_training.py --data-root path/to/MVTecAD
```

The category can be changed using the `--category` argument:

```bash
python training/patchcore_training.py --data-root data/mvtec --category cable
```

The dataset is not included in this repository. Please download the MVTec AD dataset from the official MVTec AD website and place it in your local dataset directory.

## Dataset Loading

The dataset is created in the `build_datamodule()` function.

```python
def build_datamodule(
    data_root: Path,
    category: str,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> MVTecAD:
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {data_root}\n"
            "Download the MVTec AD dataset and pass the dataset root with --data-root."
        )

    return MVTecAD(
        root=data_root,
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
```

The script checks whether the dataset root exists before training. If the dataset path is incorrect, it raises a clear error message.

## Model

The model used in this project is PatchCore.

```python
from anomalib.models import Patchcore
```

The model is created in the `build_model()` function.

```python
def build_model(coreset_ratio: float) -> Patchcore:
    return Patchcore(
        coreset_sampling_ratio=coreset_ratio,
    )
```

The default coreset sampling ratio is:

```text
1.0
```

It can be changed using:

```bash
python training/patchcore_training.py --coreset-ratio 1.0
```

PatchCore is an anomaly detection method commonly used for industrial visual inspection. It extracts visual features from normal samples and compares test images against the learned normal feature distribution.

In this project, PatchCore is used for:

- Image-level anomaly detection
- OK/NG classification
- Pixel-level anomaly localization
- Anomaly heatmap generation

## Training Engine

The training and evaluation process is handled by Anomalib's `Engine`.

```python
from anomalib.engine import Engine
```

The engine is created in the `build_engine()` function.

```python
def build_engine(output_dir: Path) -> Engine:
    return Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        default_root_dir=output_dir,
    )
```

The script automatically uses GPU acceleration if CUDA is available. Otherwise, it falls back to CPU.

```python
accelerator="gpu" if torch.cuda.is_available() else "cpu"
```

## Output Directory

By default, training results are saved to:

```text
results
```

The output directory may contain model checkpoints, logs, and evaluation outputs generated by Anomalib.

The output directory can be changed using the `--output-dir` argument:

```bash
python training/patchcore_training.py --output-dir results
```

These output files are not committed to GitHub because they can be large and environment-specific.

## Performance Settings

When CUDA is available, the script enables CUDA-related performance options:

```python
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
```

These settings help improve training and inference performance on supported GPU hardware.

## Warning and Log Filtering

The script suppresses unnecessary warnings and log messages to keep the console output clean.

Examples of filtered messages include:

- Triton FLOP counting warning
- Lightning dataloader worker warning
- Callback configuration warning
- Non-critical logs from Anomalib, Lightning, Torch, urllib3, and Hugging Face Hub

A custom `StderrFilter` is also used to remove the following repeated warning:

```text
triton not found; flop counting will not work for triton kernels
```

This filtering does not affect the training logic. It only makes the console output easier to read.

## Training Process

The main training process is executed in the `main()` function.

```python
args = parse_args()

data_root = Path(args.data_root)
output_dir = Path(args.output_dir)

datamodule = build_datamodule(
    data_root=data_root,
    category=args.category,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    num_workers=args.num_workers,
)

model = build_model(
    coreset_ratio=args.coreset_ratio,
)

engine = build_engine(
    output_dir=output_dir,
)

engine.fit(
    datamodule=datamodule,
    model=model,
)

test_results = engine.test(
    datamodule=datamodule,
    model=model,
)

print("Test Results:", test_results)
```

The training process consists of two main Anomalib engine calls.

### 1. Train the model

```python
engine.fit(
    datamodule=datamodule,
    model=model,
)
```

This step fits the PatchCore model using the training split of the MVTec AD Cable dataset.

### 2. Evaluate the model

```python
test_results = engine.test(
    datamodule=datamodule,
    model=model,
)
```

This step evaluates the trained model on the test split and returns test metrics.

## Test Results

The final test results are printed to the console:

```python
print("Test Results:", test_results)
```

The project evaluation produced the following results:

|     Metric     | Result |
|----------------|-------:|
|   Image AUROC  |   99%  |
| Image F1-score |   97%  |
|   Pixel AUROC  |   98%  |
| Pixel F1-score |   64%  |

## Result Interpretation

The model achieved strong image-level anomaly detection performance. This means it was able to separate normal and anomalous cable images effectively.

The pixel-level AUROC was also high, which means the anomaly map was able to rank abnormal pixels well across different thresholds.

However, the pixel-level F1-score was lower than the pixel-level AUROC. This is expected in many anomaly localization tasks because pixel-level F1 depends on the selected threshold and the exact overlap between the predicted anomaly mask and the ground-truth defect mask.

In practical industrial inspection, image-level OK/NG classification and pixel-level defect localization may require different threshold settings depending on the inspection objective.

## How to Run Training

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Prepare the MVTec AD dataset using this structure:

```text
data/
└── mvtec/
    └── cable/
        ├── train/
        ├── test/
        └── ground_truth/
```

Run training with default settings:

```bash
python training/patchcore_training.py
```

Run training with custom dataset and output paths:

```bash
python training/patchcore_training.py --data-root data/mvtec --category cable --output-dir results
```

Run training with custom batch size:

```bash
python training/patchcore_training.py --data-root data/mvtec --train-batch-size 12 --eval-batch-size 12
```

Run training with a custom coreset sampling ratio:

```bash
python training/patchcore_training.py --data-root data/mvtec --coreset-ratio 1.0
```

After training and testing, the script will print the test results in the terminal.

## Notes

The training script does not use machine-specific hard-coded paths. Dataset root, category, output directory, batch size, number of workers, and coreset sampling ratio can be changed through command-line arguments.

The MVTec AD dataset and model checkpoint files are not included in this repository.

Training outputs such as checkpoints, logs, and generated result folders are ignored by Git because they can be large and environment-specific.

The purpose of this training script is to demonstrate the PatchCore training and evaluation workflow for industrial visual anomaly detection using Anomalib.