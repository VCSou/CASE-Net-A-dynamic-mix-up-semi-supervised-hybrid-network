# CASE-Net: Dynamic Mix-up Semi-supervised Hybrid Network for Histopathology Segmentation

This repository contains the current implementation of **CASE-Net**, a semi-supervised histopathology segmentation pipeline for early-stage colorectal cancer. The current codebase is no longer the old TransCH-Net-only version. It now includes:

- A CoNCH/TransUNet-style foundation-model segmentation baseline.
- An AdaMix semi-supervised training variant with pseudo labels and dynamic patch-level mix-up.
- A hybrid CASE-Net variant that adds a parallel SE-ResNeXt50 branch, gated fusion, CNN skip connections, stain jitter, optional EMA teacher, automatic resume, and optional SWA.

The main target classes are:

| ID | Class |
|---:|-------|
| 0 | Background |
| 1 | Low |
| 2 | High |
| 3 | MU |

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── generate_slices_and_annotations.py
│   ├── organize_patches_masks.py
│   ├── process_masks.py
│   └── resize_patches.py
└── src/
    ├── PFM_Seg_Models.py
    ├── build_conch_v1_5.py
    ├── conch_v1_5_config.py
    ├── 新的分割模型训练conch.py
    ├── 新的分割模型训练conch_AdaMix.py
    ├── 新的分割模型训练conch_AdaMix_SEResNeXt.py
    ├── inference_seresnext.py
    └── inference_seresnext_metrics_only.py
```

## Environment

Install the pinned packages first:

```bash
pip install -r requirements.txt
```

The current source code also imports the following packages, which may need to be installed if they are not already present in your environment:

```bash
pip install albumentations segmentation-models-pytorch einops einops-exts shapely
```

For WSI preprocessing with `.sdpc` slides, the `sdpc` Python package/runtime must also be available.

## Data Preparation

Training scripts expect a dataset root with this structure:

```text
DATASET_ROOT/
├── train/
│   ├── train_224patch/
│   └── train_224maskgray/
├── valid/
│   ├── valid_224patch/
│   └── valid_224maskgray/
├── test/
│   ├── test_224patch/
│   └── test_224maskgray/
└── unlabeled/
    └── unlabeled_224patch/        # optional, used by semi-supervised training
```

Image patches are RGB images. Mask files are grayscale label maps using class IDs `0, 1, 2, 3`. 

The preprocessing scripts in `scripts/` are utility scripts for the original SDPC/JSON workflow:

- `generate_slices_and_annotations.py` cuts SDPC whole-slide images into tiles and creates color masks from JSON polygon annotations.
- `organize_patches_masks.py` collects non-empty patch/mask pairs into flat folders.
- `process_masks.py` converts color masks to grayscale class-ID masks.
- `resize_patches.py` resizes image patches.

Several preprocessing utilities use path variables inside the script body, so set those paths before running them.

## Training

Run all commands from the repository root.

### 1. CASE-Net Hybrid AdaMix SE-ResNeXt Training

This is the current main training entry point:

```bash
python src/新的分割模型训练conch_AdaMix_SEResNeXt.py \
  --dataset_dir /path/to/DATASET_ROOT \
  --save_dir /path/to/runs/case_net \
  --pfm_weights_path /path/to/conch_v1_5_pytorch_model.bin \
  --epoch 50 \
  --num_folds 5 \
  --train_batch_size 16 \
  --eval_batch_size 8 \
  --device cuda:0 \
  --multi_gpu_devices cuda:0
```

Important options:

| Option | Purpose |
|--------|---------|
| `--pfm_weights_path` | Overrides the default local CoNCH/UNI/Virchow checkpoint path. Usually required outside the original server. |
| `--model_name` | Selects `Conch_v1_5`, `UNI`, `359999`, or `Virchow_v2`. Default: `Conch_v1_5`. |
| `--cnn_backbone` | Timm CNN branch backbone. Default: `seresnext50_32x4d`. |
| `--cnn_no_pretrained` | Disables ImageNet pretrained CNN weights. |
| `--cnn_checkpoint_path` | Loads a local CNN checkpoint. Useful on offline servers. |
| `--semi_teacher_mode` | `online` or `ema`. EMA uses more GPU memory. |
| `--unlabeled_image_dirs` | Comma-separated unlabeled image folders. Defaults to `DATASET_ROOT/unlabeled/unlabeled_224patch`. |
| `--use_swa` | Enables SWA in late training. |
| `--no_auto_resume` | Disables automatic resume from `save_dir/resume_checkpoint.pth`. |
| `--rerun_completed_folds` | Re-runs folds whose metrics already reached the target epoch. |

By default, the hybrid script enables semi-supervised training and AdaMix. It writes resumable checkpoints during training and skips compatible completed folds unless `--rerun_completed_folds` is set.

### 2. AdaMix-only Semi-supervised Training

```bash
python src/新的分割模型训练conch_AdaMix.py \
  --dataset_dir /path/to/DATASET_ROOT \
  --save_dir /path/to/runs/adamix \
  --pfm_weights_path /path/to/conch_v1_5_pytorch_model.bin \
  --epoch 50 \
  --num_folds 5
```

This version reuses the CoNCH segmentation model and adds AdaMix dynamic patch mixing, pseudo-label training, fold skipping, and automatic resume support.

### 3. CoNCH Segmentation Baseline

```bash
python src/新的分割模型训练conch.py \
  --dataset_dir /path/to/DATASET_ROOT \
  --save_dir /path/to/runs/conch_baseline \
  --pfm_weights_path /path/to/conch_v1_5_pytorch_model.bin \
  --epoch 50 \
  --num_folds 5
```

Add `--semi_supervised` to enable the baseline pseudo-label pathway without AdaMix.

## Training Outputs

Each fold is saved under:

```text
SAVE_DIR/fold_1/
SAVE_DIR/fold_2/
...
```

Typical outputs include:

- `best_model.pth`: best model by validation foreground Dice.
- `last.pth`: final epoch weights.
- `swa_model.pth`: saved only when `--use_swa` is enabled and SWA succeeds.
- `training_metrics_foldN.csv`: epoch-level train/validation metrics.
- `per_class_metrics_foldN.csv`: per-class metrics for train, validation, and best test evaluation.
- `test_results_foldN.csv` and `test_results_foldN.json`: fold test results.
- `model_info_foldN.json`, `model_summary_foldN.txt`, `trainable_parameters_foldN.csv`.
- `foldN_splits.json`: train/validation/test sample IDs for the fold.

Cross-validation summaries are written to:

```text
SAVE_DIR/cross_validation_fold_results.csv
SAVE_DIR/cross_validation_summary.csv
SAVE_DIR/cross_validation_summary.json
SAVE_DIR/test_results_fold{seed}.csv
```

## Inference and Evaluation

### Save Predictions and Metrics

```bash
python src/inference_seresnext.py \
  --inference_dir /path/to/test_224patch \
  --masks_dir /path/to/test_224maskgray \
  --model_weights /path/to/runs/case_net/fold_1/best_model.pth \
  --pfm_weights_path /path/to/conch_v1_5_pytorch_model.bin \
  --output_dir /path/to/inference_output \
  --device cuda:0 \
  --batch_size 8
```

Outputs:

```text
OUTPUT_DIR/
├── forcastmaskgray/          # grayscale class-ID predictions
├── forcastmask/              # RGB visualization masks
└── inference_metrics.csv
```

### Metrics Only

```bash
python src/inference_seresnext_metrics_only.py \
  --inference_dir /path/to/test_224patch \
  --masks_dir /path/to/test_224maskgray \
  --model_weights /path/to/runs/case_net/fold_1/best_model.pth \
  --pfm_weights_path /path/to/conch_v1_5_pytorch_model.bin \
  --output_dir /path/to/metrics_output \
  --device cuda:0
```

## Notes

- The default checkpoint paths inside the scripts point to the original training server. Use `--pfm_weights_path` and, if needed, `--cnn_checkpoint_path` on a new machine.
- The hybrid inference scripts import `新的分割模型训练conch_AdaMix_SEResNeXt.py` to reconstruct the architecture, so keep the training files together in `src/`.
- The default input size used by the training and inference code is `448`.
- Five-fold cross-validation uses samples from `train,valid` by default (`--cv_source_splits train,valid`) and evaluates every fold on the fixed `test` split.

## Contact

Xinyi Pan  
pan-xy23@mails.tsinghua.edu.cn  
Shenzhen International Graduate School, Tsinghua University
