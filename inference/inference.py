from pathlib import Path
import argparse
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


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


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_batch_item(array, index, batch_size):
    array = to_numpy(array)

    if batch_size == 1:
        return np.squeeze(array)

    if array.ndim >= 3:
        return np.squeeze(array[index])

    return np.squeeze(array)


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


def build_model():
    return Patchcore(
        coreset_sampling_ratio=1.0,
    )


def build_engine():
    return Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )


def normalize_heatmap(anomaly_map):
    heatmap = anomaly_map - anomaly_map.min()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def main():
    warnings.filterwarnings("ignore")
    warnings.filterwarnings(
        "ignore",
        message=r".*triton not found; flop counting will not work for triton kernels.*",
    )

    args = parse_args()

    ckpt_path = Path(args.ckpt)
    folder_path = Path(args.input)
    threshold = args.threshold
    image_extension = args.ext

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_path}\n"
            "Place your trained checkpoint in the models/ directory or pass it with --ckpt."
        )

    image_paths = load_image_paths(folder_path, image_extension)

    model = build_model()

    dataset = PredictDataset(
        path=folder_path,
    )

    engine = build_engine()

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(ckpt_path),
    )

    if predictions is None:
        raise RuntimeError("No predictions returned.")

    results = []

    for batch in predictions:
        batch_image_paths = batch.image_path

        if isinstance(batch_image_paths, (str, Path)):
            batch_image_paths = [batch_image_paths]

        batch_size = len(batch_image_paths)

        scores = batch.pred_score
        anomaly_maps = batch.anomaly_map

        scores_np = to_numpy(scores).reshape(-1)

        for i, image_path in enumerate(batch_image_paths):
            image_path = Path(image_path)

            score = float(scores_np[i])
            status = "NG" if score >= threshold else "OK"

            img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"Cannot read image: {image_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            anomaly_map = extract_batch_item(anomaly_maps, i, batch_size)
            anomaly_map = cv2.resize(
                anomaly_map,
                (img_rgb.shape[1], img_rgb.shape[0]),
            )
            anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), 2.0)

            heatmap = normalize_heatmap(anomaly_map)

            results.append(
                {
                    "name": image_path.name,
                    "img": img_rgb,
                    "heatmap": heatmap,
                    "score": score,
                    "status": status,
                }
            )

            print("=" * 60)
            print(f"image : {image_path.name}")
            print(f"score : {score:.6f}")
            print(f"status: {status}")

    if not results:
        raise RuntimeError("No valid images to display.")

    idx = 0
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    def draw(i):
        item = results[i]

        for ax in axes:
            ax.clear()
            ax.axis("off")

        axes[0].set_title(f"Original: {item['name']}")
        axes[0].imshow(item["img"])

        axes[1].set_title("Heatmap")
        axes[1].imshow(item["heatmap"], cmap="jet", vmin=0, vmax=1)

        axes[2].set_title(
            f"Overlay | score={item['score']:.4f} | status={item['status']}"
        )
        axes[2].imshow(item["img"])
        axes[2].imshow(item["heatmap"], cmap="jet", alpha=0.35, vmin=0, vmax=1)

        fig.suptitle(
            f"Image {i + 1}/{len(results)}  |  ← previous   → next   |   Esc = close"
        )
        fig.canvas.draw_idle()

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

    fig.canvas.mpl_connect("key_press_event", on_key)

    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()
    except Exception:
        pass

    draw(idx)
    plt.show()


if __name__ == "__main__":
    main()