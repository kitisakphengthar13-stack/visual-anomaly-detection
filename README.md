# Visual Anomaly Detection with PatchCore

This project demonstrates a visual anomaly detection system for industrial inspection using PatchCore with Anomalib on the MVTec AD Cable dataset.

The system performs image-level OK/NG classification and pixel-level anomaly localization. Each test image is assigned an anomaly score, classified using a threshold-based decision rule, and visualized with an anomaly heatmap to show suspicious regions on the workpiece.

## Demo

YouTube Demo: [Watch the demo](https://www.youtube.com/watch?v=WkkjC8QWbuI)

## Objective

The objective of this project is to study how AI-based anomaly detection can be applied to industrial visual inspection, especially for:

- OK/NG classification
- Anomaly localization
- Threshold-based inspection decision
- Heatmap visualization for defect interpretation

## Dataset

- Dataset: MVTec AD
- Category: Cable
- Task: Visual anomaly detection and localization

## Model Configuration

- Method: PatchCore
- Framework: Anomalib
- Backbone: Wide ResNet-50-2
- Feature extraction layers: layer2, layer3
- Coreset sampling ratio: 1.0

## Results

The following results were obtained from the project evaluation on the MVTec AD Cable dataset:

|     Metric     | Result |
|----------------|-------:|
|   Image AUROC  |   99%  |
| Image F1-score |   97%  |
|   Pixel AUROC  |   98%  |
| Pixel F1-score |   64%  |

## OK/NG Classification

For demonstration, an anomaly score threshold of `0.5` was used to classify each test image.

|       Condition       | Prediction |
|-----------------------|------------|
| Anomaly score < 0.50  |     OK     |
| Anomaly score >= 0.50 |     NG     |

The threshold can be adjusted depending on the inspection requirement. A lower threshold increases sensitivity to defects but may also increase false positives. A higher threshold reduces false positives but may miss subtle defects.

## Visualization Output

For each test image, the system displays three outputs:

1. Original image
2. Anomaly heatmap
3. Heatmap overlay on the original image

These outputs help show not only whether a sample is classified as OK or NG, but also where the model identifies abnormal regions on the workpiece.

## Repository Structure

```text
visual_anomaly_detection/
│
├── configs/
│   └── patchcore_config.yaml
│
├── docs/
│   ├── images/
│   ├── inference.md
│   ├── training.md
│   └── results.md
│
├── inference/
│   └── inference.py
│
├── models/
│   └── README.md
│
├── training/
│   └── patchcore_training.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For CUDA support, install PyTorch according to your CUDA version from the official PyTorch installation guide.

### 2. Prepare the dataset

Download the MVTec AD dataset and place the Cable category in your dataset directory.

Default expected structure:

```text
data/
└── mvtec/
    └── cable/
        ├── train/
        ├── test/
        └── ground_truth/
```

The dataset is not included in this repository.

### 3. Train the PatchCore model

Run training with the default settings:

```bash
python training/patchcore_training.py
```

Run training with custom paths:

```bash
python training/patchcore_training.py --data-root data/mvtec --category cable --output-dir results
```

The training script saves outputs such as checkpoints and result files under the output directory.

### 4. Prepare the checkpoint for inference

After training, place or copy the trained checkpoint file to:

```text
models/model.ckpt
```

Model checkpoint files are not committed to this repository due to file size.

### 5. Run inference

Run inference with the default settings:

```bash
python inference/inference.py
```

Run inference with custom paths:

```bash
python inference/inference.py --ckpt models/model.ckpt --input data/test_images --threshold 0.5
```

For `.jpg` images:

```bash
python inference/inference.py --ckpt models/model.ckpt --input data/test_images --ext .jpg
```

The inference script classifies test images as OK or NG and generates visualization outputs including the original image, anomaly heatmap, and heatmap overlay.

## Technologies Used

- Python
- PyTorch
- PatchCore
- Anomalib
- OpenCV
- MVTec AD Dataset
- Wide ResNet-50-2

## Key Learning

This project helped me understand the practical workflow of visual anomaly detection for industrial inspection, including the difference between image-level anomaly detection, pixel-level anomaly localization, threshold-based OK/NG classification, and visualization for defect interpretation.

## Notes

Model checkpoint files are not included in this repository due to file size.

The MVTec AD dataset is not included in this repository. Please download it from the official MVTec AD website.

Training outputs such as checkpoints, logs, and generated result folders are ignored by Git because they can be large and environment-specific.

The demo video is provided to show the actual inference and visualization workflow.