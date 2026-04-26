"""
PatchCore Inference Script

This script performs visual anomaly detection using a trained PatchCore checkpoint.
It supports image-level OK/NG prediction and pixel-level anomaly visualization.

All paths are placeholders. Replace them with actual local paths before running.
"""

from pathlib import Path
import argparse
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PatchCore inference on an image or folder."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="path/to/input_image_or_folder",
        help="Path to an input image or folder.",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="path/to/model.ckpt",
        help="Path to the trained PatchCore checkpoint file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="path/to/output_dir",
        help="Directory for saving inference visualizations.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Anomaly score threshold for OK/NG decision.",
    )

    return parser.parse_args()


def to_numpy(value):
    """Convert torch tensor or array-like value to NumPy."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize anomaly heatmap to uint8 range 0-255."""
    heatmap = np.squeeze(to_numpy(heatmap)).astype(np.float32)

    min_val = float(np.min(heatmap))
    max_val = float(np.max(heatmap))

    if max_val - min_val < 1e-8:
        return np.zeros_like(heatmap, dtype=np.uint8)

    normalized = (heatmap - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Overlay anomaly heatmap on the original image."""
    heatmap_uint8 = normalize_heatmap(heatmap)
    heatmap_uint8 = cv2.resize(
        heatmap_uint8,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    return overlay


def save_result_visualization(
    image_path: Path,
    image: np.ndarray,
    heatmap: np.ndarray,
    score: float,
    label: str,
    output_dir: Path,
) -> None:
    """Save anomaly visualization result."""
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = overlay_heatmap(image, heatmap)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_rgb)
    plt.title(f"Prediction: {label} | Score: {score:.4f}")
    plt.axis("off")

    output_path = output_dir / f"{image_path.stem}_result.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved result: {output_path}")


def extract_prediction_item(prediction):
    """
    Extract the first prediction item from Anomalib output.

    Anomalib output structure may vary slightly by version.
    This helper keeps the script easier to adjust.
    """
    if isinstance(prediction, list):
        if len(prediction) == 0:
            raise RuntimeError("Empty prediction output.")
        return prediction[0]

    return prediction


def get_prediction_value(item, key: str, default=None):
    """Safely get prediction value from dict-like or object-like output."""
    if isinstance(item, dict):
        return item.get(key, default)

    return getattr(item, key, default)


def main() -> None:
    warnings.filterwarnings("ignore")

    args = parse_args()

    input_path = Path(args.input)
    ckpt_path = Path(args.ckpt)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    dataset = PredictDataset(
        path=input_path,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        accelerator="auto",
        devices=1,
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=ckpt_path,
    )

    if predictions is None:
        raise RuntimeError("No predictions returned from the model.")

    if not isinstance(predictions, list):
        predictions = [predictions]

    for index, prediction in enumerate(predictions):
        item = extract_prediction_item(prediction)

        image_path_value = get_prediction_value(item, "image_path", None)
        pred_score_value = get_prediction_value(item, "pred_score", None)
        anomaly_map_value = get_prediction_value(item, "anomaly_map", None)

        if image_path_value is None:
            print(f"[WARN] Missing image_path for prediction index {index}. Skipping.")
            continue

        if pred_score_value is None:
            print(f"[WARN] Missing pred_score for prediction index {index}. Skipping.")
            continue

        if anomaly_map_value is None:
            print(f"[WARN] Missing anomaly_map for prediction index {index}. Skipping.")
            continue

        image_path = Path(str(image_path_value))
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        score = float(np.squeeze(to_numpy(pred_score_value)))
        label = "NG" if score >= args.threshold else "OK"

        print(f"[RESULT] {image_path.name} | Score: {score:.4f} | Prediction: {label}")

        save_result_visualization(
            image_path=image_path,
            image=image,
            heatmap=anomaly_map_value,
            score=score,
            label=label,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()