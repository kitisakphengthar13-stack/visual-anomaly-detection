import argparse
import io
import logging
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


class StderrFilter(io.TextIOBase):
    """Filter repeated non-critical stderr messages."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.buffer = ""

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)

        self.buffer += text

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)

            if "triton not found; flop counting will not work for triton kernels" in line:
                continue

            self.wrapped.write(line + "\n")

        return len(text)

    def flush(self):
        if self.buffer:
            if "triton not found; flop counting will not work for triton kernels" not in self.buffer:
                self.wrapped.write(self.buffer)
            self.buffer = ""

        self.wrapped.flush()

    def isatty(self):
        return getattr(self.wrapped, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self.wrapped, "encoding", "utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate PatchCore on the MVTec AD dataset."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/mvtec",
        help="Root directory of the MVTec AD dataset.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default="cable",
        help="MVTec AD category to use, for example cable, bottle, hazelnut, etc.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for training outputs, checkpoints, and evaluation results.",
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=12,
        help="Training batch size.",
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=12,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=1.0,
        help="PatchCore coreset sampling ratio.",
    )

    return parser.parse_args()


def configure_logging_and_warnings():
    warnings.filterwarnings("ignore")
    warnings.filterwarnings(
        "ignore",
        message=r".*triton not found; flop counting will not work for triton kernels.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The 'test_dataloader' does not have many workers.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The following callbacks returned in `LightningModule.configure_callbacks`.*",
    )

    for name in [
        "",
        "lightning",
        "lightning.pytorch",
        "lightning_fabric",
        "anomalib",
        "urllib3",
        "huggingface_hub",
        "torch",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)


def configure_torch():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


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


def build_model(coreset_ratio: float) -> Patchcore:
    return Patchcore(
        coreset_sampling_ratio=coreset_ratio,
    )


def build_engine(output_dir: Path) -> Engine:
    return Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        default_root_dir=output_dir,
    )


def main():
    sys.stderr = StderrFilter(sys.stderr)

    configure_logging_and_warnings()
    configure_torch()

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


if __name__ == "__main__":
    mp.freeze_support()
    main()