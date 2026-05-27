import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base import modules as base_modules
from segmentation_models_pytorch.utils import base as smp_base
from segmentation_models_pytorch.utils import functional as smp_F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = SCRIPT_DIR / "新的分割模型训练conch_AdaMix_SEResNeXt.py"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

spec = importlib.util.spec_from_file_location("seresnext_training", TRAINING_SCRIPT)
seresnext_training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seresnext_training)
base = seresnext_training.base


CLASSES = ["back_ground", "low", "high", "mu"]
IMAGE_SIZE = 448
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

DEFAULT_MODEL_WEIGHTS = "/mnt/net_sda/pxydata/seresnext/try/fold_1/best_model.pth"
DEFAULT_IMAGE_DIR = "/data_sde/pxy/seg/examples/data/seg-code-transU/seg-code-transU/DATA2/test/test_224patch"
DEFAULT_MASK_DIR = "/data_sde/pxy/seg/examples/data/seg-code-transU/seg-code-transU/DATA2/test/test_maskforcast"
DEFAULT_OUTPUT_DIR = "/mnt/net_sda/pxydata/xiaogan_data/seresnext-internal"

PREDICTION_COLORS = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [0, 255, 0],
    3: [255, 0, 0],
}


class IoU(smp_base.Metric):
    __name__ = "iou_score"

    def __init__(
        self, eps=1e-7, threshold=0.5, activation="sigmoid", ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = base_modules.Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return smp_F.iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


def calculate_iou(pred, target, threshold=0.5):
    cal = IoU(threshold=threshold)
    iou_score = cal(pred, target)
    return iou_score.item()


def calculate_accuracy(pred, mask):
    if pred.ndim == 4:
        pred = torch.argmax(pred, dim=1)
    if mask.ndim == 4:
        mask = torch.argmax(mask, dim=1)

    assert pred.shape == mask.shape, "预测结果和真实标签的形状不一致"

    correct_pixels = torch.sum(pred == mask).item()
    total_pixels = mask.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy


def calculate_dice(pred, target, activation_fn=torch.sigmoid, threshold=0.5):
    if activation_fn:
        pred = activation_fn(pred)

    pred = (pred > threshold).float()
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (
        pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    )

    return dice.mean().item()


def get_validation_augmentation():
    return albu.Compose([albu.Resize(IMAGE_SIZE, IMAGE_SIZE)])


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing():
    transform = [
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(transform)


def find_mask_path(masks_dir, image_id):
    stem, _ = os.path.splitext(image_id)
    candidates = [
        os.path.join(masks_dir, stem + ".png"),
        os.path.join(masks_dir, stem + ".jpg"),
        os.path.join(masks_dir, stem + ".jpeg"),
        os.path.join(masks_dir, image_id),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


class EvaluationDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = sorted(
            image_id for image_id in os.listdir(images_dir) if image_id.lower().endswith(IMAGE_EXTENSIONS)
        )
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [find_mask_path(masks_dir, image_id) for image_id in self.ids]
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes] if classes else [0]
        self.augmentation = augmentation
        self.preprocessing = preprocessing if preprocessing else get_preprocessing()

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], 0)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {self.masks_fps[i]}")
        mask[mask == 99] = 1
        mask[mask == 199] = 2

        masks = [(mask == value) for value in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


class InferenceDataset(Dataset):
    def __init__(self, images_dir, augmentation=None, preprocessing=None):
        self.ids = sorted(
            image_id for image_id in os.listdir(images_dir) if image_id.lower().endswith(IMAGE_EXTENSIONS)
        )
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing if preprocessing else get_preprocessing()

    def __getitem__(self, i):
        image_id = self.ids[i]
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {self.images_fps[i]}")

        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        return image, image_id, np.array([height, width], dtype=np.int32)

    def __len__(self):
        return len(self.ids)


def make_training_args(args, deep_supervision=False):
    train_args = seresnext_training.build_parser().parse_args([])
    train_args.model_name = args.model_name
    train_args.pfm_weights_path = args.pfm_weights_path
    train_args.image_size = IMAGE_SIZE
    train_args.deep_supervision = deep_supervision
    train_args.cnn_backbone = args.cnn_backbone
    train_args.cnn_no_pretrained = args.cnn_no_pretrained
    train_args.cnn_checkpoint_path = args.cnn_checkpoint_path
    train_args.cnn_freeze_stages = args.cnn_freeze_stages
    return train_args


def normalize_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "state_dict_ema", "model_ema"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a state_dict-like object.")

    normalized = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def build_model(args, device):
    checkpoint = torch.load(args.model_weights, map_location="cpu")
    state_dict = normalize_state_dict(checkpoint)
    last_error = None

    for deep_supervision in (False, True):
        train_args = make_training_args(args, deep_supervision=deep_supervision)
        model, _ = seresnext_training.build_hybrid_model(train_args)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            last_error = exc
            continue
        model.to(device)
        model.eval()
        return model

    raise RuntimeError(f"Failed to load model checkpoint strictly: {last_error}")


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou, total_acc = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluation", unit="batch")

        for images, masks in progress_bar:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            logits = base.get_main_logits(outputs).float()
            loss = criterion(logits, masks)

            dice = calculate_dice(logits, masks)
            iou = calculate_iou(logits, masks)
            acc = calculate_accuracy(logits, masks)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_acc += acc

            progress_bar.set_postfix(loss=loss.item(), dice=dice, iou=iou, acc=acc)

    n = len(data_loader)
    if n == 0:
        raise ValueError("Evaluation data loader is empty.")
    return total_loss / n, total_dice / n, total_iou / n, total_acc / n


def colorize_mask(mask_gray):
    rgb_mask = np.zeros((*mask_gray.shape, 3), dtype=np.uint8)
    for class_value, color in PREDICTION_COLORS.items():
        rgb_mask[mask_gray == class_value] = color
    return rgb_mask


def save_predictions(model, images_dir, gray_dir, rgb_dir, batch_size, num_workers, device):
    os.makedirs(gray_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    dataset = InferenceDataset(
        images_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Saving masks", unit="batch")
        for images, image_ids, original_sizes in progress_bar:
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)
            logits = base.get_main_logits(outputs).float()
            masks = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

            for mask_gray, image_id, original_size in zip(masks, image_ids, original_sizes):
                height, width = int(original_size[0]), int(original_size[1])
                if mask_gray.shape != (height, width):
                    mask_gray = cv2.resize(
                        mask_gray,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                output_name = os.path.splitext(image_id)[0] + ".png"
                gray_path = os.path.join(gray_dir, output_name)
                rgb_path = os.path.join(rgb_dir, output_name)
                rgb_mask = colorize_mask(mask_gray)

                cv2.imwrite(gray_path, mask_gray)
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))


def write_metrics_csv(path, loss, dice, iou, accuracy):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Loss", f"{loss:.6f}"])
        writer.writerow(["Dice", f"{dice:.6f}"])
        writer.writerow(["IOU", f"{iou:.6f}"])
        writer.writerow(["Accuracy", f"{accuracy:.6f}"])


def main():
    parser = argparse.ArgumentParser(description="Conch AdaMix SEResNeXt segmentation inference")
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
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gray_dir = os.path.join(args.output_dir, "forcastmaskgray")
    rgb_dir = os.path.join(args.output_dir, "forcastmask")
    metrics_csv = os.path.join(args.output_dir, "inference_metrics.csv")
    os.makedirs(args.output_dir, exist_ok=True)

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
    save_predictions(
        model,
        args.inference_dir,
        gray_dir,
        rgb_dir,
        args.batch_size,
        args.num_workers,
        device,
    )

    print(f"Loss: {loss:.6f}")
    print(f"Dice: {dice:.6f}")
    print(f"IoU: {iou:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Saved metrics CSV: {metrics_csv}")
    print(f"Saved grayscale masks to: {gray_dir}")
    print(f"Saved RGB masks to: {rgb_dir}")


if __name__ == "__main__":
    main()
