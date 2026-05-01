# Models

This directory is reserved for model checkpoint files used by the **Visual Anomaly Detection with PatchCore** project.

Actual model checkpoint files are not included in this repository due to file size limitations and environment-specific training outputs.

---

## Expected Checkpoint File

After training, Anomalib generates checkpoint files that can be used for inference.

Example checkpoint path:

```text
path/to/model.ckpt
```

This path should be replaced with the actual checkpoint path generated on your machine.

---

## Why Checkpoint Files Are Not Included

Model checkpoint files are not uploaded to this repository because:

- Checkpoint files can be large
- Training output paths may vary by machine and environment
- Checkpoints are generated after training
- This repository is intended to document the source code, configuration, workflow, and example outputs

Because of this, users should train the model or copy their own checkpoint file locally before running inference.

---

## Checkpoint Path Configuration

The checkpoint path can be provided when running inference:

```bash
python inference/inference.py --ckpt path/to/model.ckpt --input path/to/input_image_or_folder --output-dir path/to/output_dir
```

The checkpoint path is also referenced in:

```text
configs/patchcore_config.yaml
```

Example configuration:

```yaml
checkpoint:
  path: "path/to/model.ckpt"
```

Replace `path/to/model.ckpt` with the actual checkpoint file path used on your machine.

---

## Training Output

The checkpoint is generated during the training process.

Training script:

```text
training/patchcore_training.py
```

Example training command:

```bash
python training/patchcore_training.py --data-root path/to/mvtec_ad --category cable --output-dir path/to/results
```

The generated checkpoint will usually be stored inside the training output directory created by Anomalib.

---

## Notes

This directory intentionally does not contain actual checkpoint files.

Expected checkpoint type:

```text
.ckpt
```

Dataset files, checkpoints, logs, and generated training outputs are ignored by Git because they can be large and environment-specific.
