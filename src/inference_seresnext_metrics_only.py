import argparse
import os

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from inference_seresnext import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_MASK_DIR,
    DEFAULT_MODEL_WEIGHTS,
    DEFAULT_OUTPUT_DIR,
    EvaluationDataset,
    build_model,
    evaluate,
    get_preprocessing,
    get_validation_augmentation,
    write_metrics_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Conch AdaMix SEResNeXt metrics-only inference"
    )
    parser.add_argument("--inference_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--masks_dir", type=str, default=DEFAULT_MASK_DIR)
    parser.add_argument("--model_weights", type=str, default=DEFAULT_MODEL_WEIGHTS)
    parser.add_argument("--pfm_weights_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="Conch_v1_5")
    parser.add_argument("--cnn_backbone", type=str, default="seresnext50_32x4d")
    parser.add_argument("--cnn_no_pretrained", action="store_true")
    parser.add_argument("--cnn_checkpoint_path", type=str, default="")
    parser.add_argument("--cnn_freeze_stages", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_csv = os.path.join(args.output_dir, "inference_metrics.csv")

    model = build_model(args, device)
    dataset = EvaluationDataset(
        args.inference_dir,
        args.masks_dir,
        classes=["back_ground", "low", "high", "mu"],
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    criterion = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
    loss, dice, iou, accuracy = evaluate(model, data_loader, criterion, device)
    write_metrics_csv(metrics_csv, loss, dice, iou, accuracy)

    print(f"Loss: {loss:.6f}")
    print(f"Dice: {dice:.6f}")
    print(f"IoU: {iou:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Saved metrics CSV: {metrics_csv}")


if __name__ == "__main__":
    main()
