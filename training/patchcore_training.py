"""
PatchCore Training Script

This script trains a PatchCore anomaly detection model using Anomalib
on the MVTec AD Cable dataset.

All paths are placeholders. Replace them with actual local paths before running.
"""

from multiprocessing import freeze_support
from pathlib import Path
import argparse
import warnings

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PatchCore on the MVTec AD Cable dataset."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="path/to/mvtec_ad",
        help="Path to the MVTec AD dataset root directory.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default="cable",
        help="MVTec AD category name.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="path/to/results",
        help="Directory for training results and checkpoints.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Input image size.",
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.1,
        help="PatchCore coreset sampling ratio.",
    )

    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")

    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    datamodule = MVTecAD(
        root=data_root,
        category=args.category,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=output_dir,
        accelerator="auto",
        devices=1,
    )

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    engine.test(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    freeze_support()
    main()