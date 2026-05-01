# Visual Anomaly Detection with PatchCore

Visual anomaly detection project for industrial inspection using PatchCore with Anomalib on the MVTec AD Cable dataset.

The project demonstrates image-level OK/NG classification and pixel-level anomaly localization. Each input image receives an anomaly score, is classified with a threshold-based rule, and is visualized with an anomaly heatmap overlay.

## Demo

YouTube demo: [Watch the demo](https://www.youtube.com/watch?v=WkkjC8QWbuI)

## Project Scope

This repository is intended as a Computer Vision / AI engineering portfolio project. It documents the training, inference, evaluation, and visualization workflow for PatchCore anomaly detection.

The repository does not include the MVTec AD dataset, trained checkpoints, or generated training outputs because those files are large and environment-specific. It is not presented as a production-ready inspection system.

## Objective

The project focuses on:

- OK/NG image classification
- Pixel-level anomaly localization
- Threshold-based inspection decisions
- Heatmap visualization for defect interpretation
- Reproducible training and inference commands

## Dataset

- Dataset: MVTec AD
- Category: Cable
- Task: Visual anomaly detection and localization

The MVTec AD dataset is not included in this repository. Download it separately from the official MVTec AD source and provide the local dataset path when running training.

Expected dataset structure:

```text
path/to/mvtec_ad/
`-- cable/
    |-- train/
    |-- test/
    `-- ground_truth/
```

## Environment

The project depends on Python, PyTorch, Anomalib, OpenCV, NumPy, and Matplotlib. Exact compatibility depends on the Anomalib and PyTorch versions installed in your environment.

Recommended notes:

- Python: use a modern Python version supported by your selected Anomalib release.
- PyTorch/CUDA: install PyTorch with the CUDA build that matches your GPU driver, or use a CPU build if GPU acceleration is unavailable.
- Anomalib: APIs can change between versions, especially dataset, model, and engine interfaces. If an Anomalib upgrade breaks a script, check the installed Anomalib documentation for the matching API.
- OS: scripts are written with cross-platform `pathlib` paths and should work on Windows, Linux, or macOS when dependencies are installed correctly.

Install project-level dependencies:

```bash
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the PyTorch package from the official PyTorch installation selector before or instead of the generic `torch` package in `requirements.txt`.

## Configuration

The reference configuration is stored in:

```text
configs/patchcore_config.yaml
```

The Python scripts use command-line arguments directly. The YAML file documents the project settings and example local paths.

Model settings used by the scripts:

| Setting | Value |
|---|---|
| Method | PatchCore |
| Framework | Anomalib |
| Backbone | `wide_resnet50_2` |
| Feature layers | `layer2`, `layer3` |
| Coreset sampling ratio | `1.0` |
| Number of nearest neighbors | `9` |

## Path Examples

Use `.env.example` as a path reference for local setup. The scripts do not automatically load `.env` files, so pass these paths through command-line arguments.

Example paths:

```text
DATA_ROOT=path/to/mvtec_ad
CATEGORY=cable
TRAIN_OUTPUT_DIR=path/to/results
CHECKPOINT_PATH=path/to/model.ckpt
INPUT_PATH=path/to/input_image_or_folder
INFERENCE_OUTPUT_DIR=path/to/output_dir
THRESHOLD=0.5
```

## Training

Run training with the default example arguments:

```bash
python training/patchcore_training.py
```

Run training with local paths:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --category cable --output-dir path/to/results
```

Training command-line arguments:

| Argument | Default | Description |
|---|---|---|
| `--data-root` | `path/to/mvtec_ad` | Root directory containing MVTec AD categories |
| `--category` | `cable` | MVTec AD category used for training and evaluation |
| `--output-dir` | `path/to/results` | Directory for Anomalib outputs, logs, and checkpoints |
| `--image-size` | `256` | Input image size reference documented for the project |
| `--train-batch-size` | `32` | Training batch size |
| `--eval-batch-size` | `32` | Evaluation batch size |
| `--num-workers` | `4` | Number of dataloader workers |
| `--coreset-ratio` | `1.0` | PatchCore coreset sampling ratio |

Note: `--image-size` is currently exposed as a documented argument but is not passed into the Anomalib datamodule by the script. This preserves the current training behavior.

## Inference

Run inference with the default example arguments:

```bash
python inference/inference.py
```

Run inference with local paths:

```bash
python inference/inference.py --ckpt path/to/model.ckpt --input path/to/input_image_or_folder --output-dir path/to/output_dir --threshold 0.5
```

Inference command-line arguments:

| Argument | Default | Description |
|---|---|---|
| `--input` | `path/to/input_image_or_folder` | Input image or folder to inspect |
| `--ckpt` | `path/to/model.ckpt` | Trained PatchCore checkpoint path |
| `--output-dir` | `path/to/output_dir` | Directory for saved visualization outputs |
| `--threshold` | `0.5` | Anomaly score threshold for OK/NG classification |

Decision rule:

| Condition | Prediction |
|---|---|
| Anomaly score < 0.50 | OK |
| Anomaly score >= 0.50 | NG |

The threshold can be adjusted depending on the inspection requirement. A lower threshold increases sensitivity to defects but may increase false positives. A higher threshold reduces false positives but may miss subtle defects.

## Results

Project evaluation results on the MVTec AD Cable category:

| Metric | Result |
|---|---:|
| Image AUROC | 99% |
| Image F1-score | 97% |
| Pixel AUROC | 98% |
| Pixel F1-score | 64% |

These values are reported as project evaluation results for this repository and should not be treated as fixed benchmark numbers.

## Reproducibility

Results can vary across runs and environments. Differences may come from:

- Anomalib version and API behavior
- PyTorch and CUDA versions
- Dataset split and local dataset contents
- Trained checkpoint used for inference
- Threshold selection
- Image preprocessing and transforms applied by the installed Anomalib version
- Hardware, operating system, and dependency environment

When comparing results, record the dependency versions, dataset path/category, checkpoint path, threshold, and command used.

## Example Outputs

### OK Samples

![OK Sample Result 1](docs/assets/ok_sample_result_1.png)

![OK Sample Result 2](docs/assets/ok_sample_result_2.png)

### NG Samples

![NG Sample Result 1](docs/assets/ng_sample_result_1.png)

![NG Sample Result 2](docs/assets/ng_sample_result_2.png)

## Repository Structure

```text
visual_anomaly_detection/
|-- configs/
|   `-- patchcore_config.yaml
|-- docs/
|   |-- assets/
|   |   |-- ng_sample_result_1.png
|   |   |-- ng_sample_result_2.png
|   |   |-- ok_sample_result_1.png
|   |   `-- ok_sample_result_2.png
|   |-- inference.md
|   |-- results.md
|   `-- training.md
|-- inference/
|   `-- inference.py
|-- models/
|   `-- README.md
|-- training/
|   `-- patchcore_training.py
|-- .env.example
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Main Components

| Path | Purpose |
|---|---|
| `configs/patchcore_config.yaml` | Reference configuration for dataset, model, training, inference, and metric settings |
| `training/patchcore_training.py` | PatchCore training and test script using Anomalib |
| `inference/inference.py` | PatchCore inference and visualization script |
| `docs/training.md` | Training workflow documentation |
| `docs/inference.md` | Inference workflow documentation |
| `docs/results.md` | Evaluation results and interpretation notes |
| `docs/assets/` | Preserved sample output images |
| `models/README.md` | Checkpoint file instructions |
| `.env.example` | Example local path variables |
| `requirements.txt` | Python-level package requirements |

## Documentation

- [Training Guide](docs/training.md)
- [Inference Guide](docs/inference.md)
- [Results](docs/results.md)

## Notes

- Model checkpoint files are not included in this repository.
- The MVTec AD dataset is not included in this repository.
- Training outputs such as checkpoints, logs, and generated result folders are ignored by Git because they can be large and environment-specific.
- The demo video shows the inference and visualization workflow for this project.
