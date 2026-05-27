import argparse
import copy
import csv
import json
import os
import random
import time

import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm


CLASS_NAMES = ["Background", "Low", "High", "MU"]
FOREGROUND_CLASS_IDS = [1, 2, 3]
DEFAULT_IMAGE_SIZE = 448


def build_parser():
    parser = argparse.ArgumentParser(description="TransCH-Net segmentation training")
    parser.add_argument("--dataset_dir", type=str, default="/data_sde/pxy/seg/examples/data/seg-code-transU/seg-code-transU/DATA2")
    parser.add_argument("--save_dir", type=str, default="try")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--model_name", default="Conch_v1_5", type=str)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--cv_source_splits", type=str, default="train,valid", help="Comma-separated split names used for cross-validation.")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--pfm_lr_mult", type=float, default=0.1)
    parser.add_argument("--class_weights", type=str, default="", help="Optional comma-separated CE weights, e.g. 0.5,1,1,1.2")
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--aux_loss_weight", type=float, default=0.4)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--decoder_dropout", type=float, default=0.1)
    parser.add_argument("--decoder_channels", type=str, default="128,64,32,16")
    parser.add_argument("--decoder_head_channels", type=int, default=256)
    parser.add_argument("--decoder_feature_noise_std", type=float, default=0.0, help="Optional Gaussian noise on PFM tokens during training.")
    parser.add_argument("--unfreeze_pfm_blocks", type=int, default=0, help="Fine-tune the last N PFM transformer blocks.")
    parser.add_argument("--pfm_weights_path", type=str, default="", help="Optional override for the selected PFM checkpoint path.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision training when --amp_forward is also set.")
    parser.add_argument("--amp_forward", action="store_true", help="Use fp16 autocast for model forward. Faster but less stable.")
    parser.add_argument("--no_scheduler", action="store_true")
    parser.add_argument("--deep_supervision", action="store_true", help="Enable auxiliary decoder losses. Uses more GPU memory.")
    parser.add_argument("--semi_supervised", action="store_true", help="Enable semi-supervised training with extra unlabeled patches.")
    parser.add_argument(
        "--unlabeled_image_dirs",
        type=str,
        default="",
        help="Comma-separated unlabeled image folders. Defaults to dataset_dir/unlabeled/unlabeled_224patch.",
    )
    parser.add_argument("--unlabeled_batch_size", type=int, default=0, help="Unlabeled batch size. 0 means half of train_batch_size.")
    parser.add_argument("--semi_start_epoch", type=int, default=5, help="Epoch to start pseudo-label training after supervised warmup.")
    parser.add_argument("--unsup_weight", type=float, default=0.5, help="Maximum weight for the unlabeled consistency loss.")
    parser.add_argument("--unsup_rampup_epochs", type=int, default=10, help="Ramp-up length for the unlabeled loss weight.")
    parser.add_argument("--unsup_confidence_threshold", type=float, default=0.75, help="Foreground pseudo-label confidence threshold.")
    parser.add_argument("--unsup_bg_confidence_threshold", type=float, default=0.98, help="Background pseudo-label confidence threshold.")
    parser.add_argument("--unsup_ignore_background", action="store_true", help="Ignore background pseudo-label pixels in unlabeled loss.")
    parser.add_argument("--unsup_dice_weight", type=float, default=1.0, help="Dice component weight for pseudo-label loss.")
    parser.add_argument("--unsup_ce_weight", type=float, default=1.0, help="CE component weight for pseudo-label loss.")
    parser.add_argument(
        "--semi_teacher_mode",
        type=str,
        default="online",
        choices=("online", "ema"),
        help="online uses the current model as teacher; ema uses a separate EMA teacher and needs more GPU memory.",
    )
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA teacher decay when --semi_teacher_mode ema is used.")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_training_augmentation(image_size):
    train_transform = [
        albu.Resize(image_size, image_size),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=15, shift_limit=0.1, p=0.8, border_mode=cv2.BORDER_CONSTANT),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.35),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.8,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
        albu.HueSaturationValue(p=0.5),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(image_size):
    return albu.Compose([albu.Resize(image_size, image_size)])


def get_unlabeled_geometric_augmentation(image_size):
    return albu.Compose(
        [
            albu.Resize(image_size, image_size),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.35, rotate_limit=15, shift_limit=0.08, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            albu.Perspective(p=0.25),
        ],
        additional_targets={"strong_image": "image"},
    )


def get_unlabeled_weak_color_augmentation():
    return albu.Compose([albu.RandomBrightnessContrast(p=0.15)])


def get_unlabeled_strong_color_augmentation():
    return albu.Compose(
        [
            albu.GaussNoise(p=0.25),
            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightnessContrast(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.8,
            ),
            albu.OneOf(
                [
                    albu.Sharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.5,
            ),
            albu.HueSaturationValue(p=0.5),
        ]
    )


def image_to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def mask_to_tensor(x, **kwargs):
    return x.astype("int64")


def get_preprocessing():
    transform = [
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albu.Lambda(image=image_to_tensor, mask=mask_to_tensor),
    ]
    return albu.Compose(transform)


def get_image_preprocessing():
    transform = [
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albu.Lambda(image=image_to_tensor),
    ]
    return albu.Compose(transform)


class HistologySegDataset(TorchDataset):
    image_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    gray_value_map = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 0,
        29: 1,
        76: 3,
        99: 1,
        150: 2,
        199: 2,
    }
    rgb_value_map = {
        (0, 0, 0): 0,
        (0, 0, 255): 1,
        (0, 255, 0): 2,
        (255, 0, 0): 3,
        (255, 165, 0): 0,
    }

    def __init__(self, images_dir=None, masks_dir=None, sample_records=None, augmentation=None, preprocessing=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        if sample_records is None:
            sample_records = self.collect_samples(images_dir, masks_dir)
        self.sample_records = list(sample_records)
        self.ids = [record["id"] for record in self.sample_records]
        self.images_fps = [record["image_path"] for record in self.sample_records]
        self.masks_fps = [record["mask_path"] for record in self.sample_records]
        self.augmentation = augmentation
        self.preprocessing = preprocessing or get_preprocessing()

    @classmethod
    def collect_samples(cls, images_dir, masks_dir, split_name=""):
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
        if not os.path.isdir(masks_dir):
            raise FileNotFoundError(f"Mask directory does not exist: {masks_dir}")

        records = []
        ids = sorted([name for name in os.listdir(images_dir) if name.lower().endswith(cls.image_exts)])
        for image_id in ids:
            image_path = os.path.join(images_dir, image_id)
            mask_path = cls.find_mask_path(masks_dir, image_id)
            records.append(
                {
                    "id": f"{split_name}/{image_id}" if split_name else image_id,
                    "image_id": image_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "split": split_name,
                }
            )
        return records

    @staticmethod
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

    def _load_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        if mask.ndim == 3:
            mask_rgb = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2RGB)
            mapped = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
            for color, class_id in self.rgb_value_map.items():
                mapped[np.all(mask_rgb == color, axis=-1)] = class_id
            return mapped

        if mask.max() <= len(CLASS_NAMES) - 1:
            return mask.astype(np.uint8)

        mapped = np.zeros(mask.shape, dtype=np.uint8)
        for raw_value, class_id in self.gray_value_map.items():
            mapped[mask == raw_value] = class_id
        return mapped

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self._load_mask(self.masks_fps[i])

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


class UnlabeledHistologyDataset(TorchDataset):
    image_exts = HistologySegDataset.image_exts

    def __init__(
        self,
        image_dirs,
        image_size,
        geometric_augmentation=None,
        weak_color_augmentation=None,
        strong_color_augmentation=None,
        preprocessing=None,
    ):
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        self.image_dirs = [path for path in image_dirs if path]
        self.image_size = image_size
        self.samples = self.collect_images(self.image_dirs)
        self.ids = [record["id"] for record in self.samples]
        self.images_fps = [record["image_path"] for record in self.samples]
        self.geometric_augmentation = geometric_augmentation or get_unlabeled_geometric_augmentation(image_size)
        self.weak_color_augmentation = weak_color_augmentation or get_unlabeled_weak_color_augmentation()
        self.strong_color_augmentation = strong_color_augmentation or get_unlabeled_strong_color_augmentation()
        self.preprocessing = preprocessing or get_image_preprocessing()

    @classmethod
    def collect_images(cls, image_dirs):
        records = []
        for image_dir in image_dirs:
            if not os.path.isdir(image_dir):
                continue
            for root, _, files in os.walk(image_dir):
                for file_name in sorted(files):
                    if not file_name.lower().endswith(cls.image_exts):
                        continue
                    image_path = os.path.join(root, file_name)
                    rel_id = os.path.relpath(image_path, image_dir)
                    records.append({"id": f"{os.path.basename(image_dir)}/{rel_id}", "image_path": image_path})
        records.sort(key=lambda item: item["image_path"])
        return records

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise FileNotFoundError(f"Failed to read unlabeled image: {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        weak_image = image.copy()
        strong_image = image.copy()

        if self.geometric_augmentation:
            sample = self.geometric_augmentation(image=weak_image, strong_image=strong_image)
            weak_image, strong_image = sample["image"], sample["strong_image"]

        if self.weak_color_augmentation:
            weak_image = self.weak_color_augmentation(image=weak_image)["image"]
        if self.strong_color_augmentation:
            strong_image = self.strong_color_augmentation(image=strong_image)["image"]

        if self.preprocessing:
            weak_image = self.preprocessing(image=weak_image)["image"]
            strong_image = self.preprocessing(image=strong_image)["image"]

        return weak_image, strong_image

    def __len__(self):
        return len(self.images_fps)


class CompositeSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        class_weights=None,
        dice_weight=1.0,
        ce_weight=0.5,
        aux_loss_weight=0.4,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.aux_loss_weight = aux_loss_weight
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        try:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        except TypeError:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def _resize_target(self, target, size):
        if target.shape[-2:] == size:
            return target
        target = target.unsqueeze(1).float()
        target = torch_F.interpolate(target, size=size, mode="nearest")
        return target.squeeze(1).long()

    def _single_loss(self, logits, target):
        target = self._resize_target(target, logits.shape[-2:])
        logits = logits.float()
        return self.dice_weight * self.dice_loss(logits, target) + self.ce_weight * self.ce_loss(logits, target)

    def forward(self, outputs, target):
        if isinstance(outputs, dict):
            main_logits = outputs["out"]
            loss = self._single_loss(main_logits, target)
            aux_outputs = outputs.get("aux", [])
            if aux_outputs:
                aux_loss = sum(self._single_loss(aux, target) for aux in aux_outputs) / len(aux_outputs)
                loss = loss + self.aux_loss_weight * aux_loss
            return loss
        return self._single_loss(outputs, target)


class PseudoLabelLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        dice_weight=1.0,
        ce_weight=1.0,
        confidence_threshold=0.75,
        bg_confidence_threshold=0.98,
        ignore_background=False,
        ignore_index=255,
        eps=1e-7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.confidence_threshold = confidence_threshold
        self.bg_confidence_threshold = bg_confidence_threshold
        self.ignore_background = ignore_background
        self.ignore_index = ignore_index
        self.eps = eps

    def make_pseudo_labels(self, teacher_logits):
        probs = torch.softmax(teacher_logits.float(), dim=1)
        confidence, pseudo = torch.max(probs, dim=1)
        thresholds = torch.full_like(confidence, self.confidence_threshold)
        thresholds = torch.where(pseudo == 0, torch.full_like(thresholds, self.bg_confidence_threshold), thresholds)
        valid = confidence >= thresholds
        if self.ignore_background:
            valid = valid & (pseudo != 0)
        pseudo = pseudo.long()
        pseudo = torch.where(valid, pseudo, torch.full_like(pseudo, self.ignore_index))
        return pseudo, confidence, valid

    def _dice_loss(self, logits, target, valid):
        if not valid.any():
            return logits.sum() * 0.0
        probs = torch.softmax(logits.float(), dim=1)
        safe_target = target.clamp(0, self.num_classes - 1)
        target_one_hot = torch_F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask
        dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (denominator + self.eps)
        valid_classes = target_one_hot.sum(dim=dims) > 0
        if not valid_classes.any():
            return logits.sum() * 0.0
        return dice_loss[valid_classes].mean()

    def forward(self, student_outputs, teacher_logits):
        student_logits = get_main_logits(student_outputs).float()
        pseudo, confidence, valid = self.make_pseudo_labels(teacher_logits)
        if student_logits.shape[-2:] != pseudo.shape[-2:]:
            student_logits = torch_F.interpolate(student_logits, size=pseudo.shape[-2:], mode="bilinear", align_corners=False)
        if not valid.any():
            zero = student_logits.sum() * 0.0
            stats = {
                "valid_ratio": 0.0,
                "mean_confidence": float(confidence.mean().detach().cpu().item()),
                "pseudo_foreground_ratio": 0.0,
            }
            return zero, stats
        ce_loss = torch_F.cross_entropy(student_logits, pseudo, ignore_index=self.ignore_index)
        dice_loss = self._dice_loss(student_logits, pseudo, valid)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        stats = {
            "valid_ratio": float(valid.float().mean().detach().cpu().item()),
            "mean_confidence": float(confidence.mean().detach().cpu().item()),
            "pseudo_foreground_ratio": float(((pseudo > 0) & valid).float().mean().detach().cpu().item()),
        }
        return loss, stats


def get_main_logits(outputs):
    return outputs["out"] if isinstance(outputs, dict) else outputs


def describe_bad_batch(logits, masks):
    with torch.no_grad():
        logits_float = logits.detach().float()
        mask_values = torch.unique(masks.detach()).cpu().tolist()
        return (
            f"logits_min={logits_float.min().item():.4f}, "
            f"logits_max={logits_float.max().item():.4f}, "
            f"logits_nan={torch.isnan(logits_float).any().item()}, "
            f"logits_inf={torch.isinf(logits_float).any().item()}, "
            f"mask_values={mask_values}"
        )


def gradients_are_finite(model):
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


def rampup_weight(current_epoch, start_epoch, rampup_epochs, max_weight):
    if max_weight <= 0 or current_epoch < start_epoch:
        return 0.0
    if rampup_epochs <= 0:
        return max_weight
    progress = min(max((current_epoch - start_epoch + 1) / float(rampup_epochs), 0.0), 1.0)
    return float(max_weight * progress)


def next_unlabeled_batch(unlabeled_iter, unlabeled_loader):
    try:
        batch = next(unlabeled_iter)
    except StopIteration:
        unlabeled_iter = iter(unlabeled_loader)
        batch = next(unlabeled_iter)
    return batch, unlabeled_iter


@torch.no_grad()
def update_ema_model(student_model, teacher_model, decay):
    student_state = student_model.state_dict()
    teacher_state = teacher_model.state_dict()
    for name, teacher_value in teacher_state.items():
        student_value = student_state[name].detach()
        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(student_value.to(teacher_value.device), alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value.to(teacher_value.device))


@torch.no_grad()
def teacher_forward(model, images, amp_enabled=False):
    was_training = model.training
    model.eval()
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        outputs = model(images)
        logits = get_main_logits(outputs).float().detach()
    if was_training:
        model.train()
    return logits


def prepare_target(masks):
    if masks.ndim == 4:
        if masks.shape[1] == len(CLASS_NAMES):
            masks = torch.argmax(masks, dim=1)
        else:
            masks = torch.argmax(masks, dim=-1)
    return masks.long()


@torch.no_grad()
def per_class_loss(logits, target, num_classes, class_weights=None, dice_weight=1.0, ce_weight=0.5, eps=1e-7):
    probs = torch.softmax(logits, dim=1)
    target_one_hot = torch_F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    dice_loss = 1.0 - dice

    ce_map = torch_F.cross_entropy(logits, target, weight=class_weights, reduction="none")
    ce_losses = []
    for class_id in range(num_classes):
        class_mask = target == class_id
        if class_mask.any():
            ce_losses.append(ce_map[class_mask].mean())
        else:
            ce_losses.append(torch.tensor(float("nan"), device=logits.device))
    ce_losses = torch.stack(ce_losses)
    ce_losses = torch.where(torch.isnan(ce_losses), torch.zeros_like(ce_losses), ce_losses)
    combined = dice_weight * dice_loss + ce_weight * ce_losses
    return combined.detach().cpu().numpy()


class SegmentationMetricsMeter:
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)
        self.loss_sum = 0.0
        self.sample_count = 0
        self.class_loss_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.class_loss_count = np.zeros(self.num_classes, dtype=np.float64)

    @torch.no_grad()
    def update(self, logits, target, loss_value, class_losses=None):
        batch_size = target.shape[0]
        pred = torch.argmax(logits, dim=1)
        target_cpu = target.detach().cpu().view(-1)
        pred_cpu = pred.detach().cpu().view(-1)
        valid = (target_cpu >= 0) & (target_cpu < self.num_classes)
        encoded = target_cpu[valid] * self.num_classes + pred_cpu[valid]
        bincount = torch.bincount(encoded, minlength=self.num_classes ** 2).double()
        self.confusion += bincount.reshape(self.num_classes, self.num_classes)
        self.loss_sum += float(loss_value) * batch_size
        self.sample_count += batch_size
        if class_losses is not None:
            class_losses = np.asarray(class_losses, dtype=np.float64)
            valid_losses = ~np.isnan(class_losses)
            self.class_loss_sum[valid_losses] += class_losses[valid_losses]
            self.class_loss_count[valid_losses] += 1

    def compute(self):
        cm = self.confusion
        tp = torch.diag(cm)
        pred_count = cm.sum(dim=0)
        target_count = cm.sum(dim=1)
        fp = pred_count - tp
        fn = target_count - tp
        total = cm.sum()

        iou = safe_divide(tp, tp + fp + fn)
        dice = safe_divide(2 * tp, 2 * tp + fp + fn)
        class_acc = safe_divide(tp, target_count)
        overall_acc = safe_divide(tp.sum(), total)

        class_losses = np.divide(
            self.class_loss_sum,
            self.class_loss_count,
            out=np.full_like(self.class_loss_sum, np.nan),
            where=self.class_loss_count > 0,
        )
        class_metrics = {}
        for idx, name in enumerate(self.class_names):
            class_metrics[name] = {
                "dice": tensor_to_float(dice[idx]),
                "iou": tensor_to_float(iou[idx]),
                "accuracy": tensor_to_float(class_acc[idx]),
                "loss": float(class_losses[idx]),
                "support_pixels": int(target_count[idx].item()),
                "pred_pixels": int(pred_count[idx].item()),
            }

        foreground = torch.tensor(FOREGROUND_CLASS_IDS, dtype=torch.long)
        return {
            "loss": self.loss_sum / max(self.sample_count, 1),
            "mean_dice": nanmean_tensor(dice),
            "mean_iou": nanmean_tensor(iou),
            "mean_accuracy": nanmean_tensor(class_acc),
            "mean_foreground_dice": nanmean_tensor(dice[foreground]),
            "mean_foreground_iou": nanmean_tensor(iou[foreground]),
            "overall_accuracy": tensor_to_float(overall_acc),
            "class_metrics": class_metrics,
        }


def safe_divide(numerator, denominator):
    result = numerator / torch.clamp(denominator, min=1.0)
    return torch.where(denominator > 0, result, torch.full_like(result, float("nan")))


def tensor_to_float(value):
    if torch.isnan(value):
        return float("nan")
    return float(value.item())


def nanmean_tensor(values):
    valid = values[~torch.isnan(values)]
    if valid.numel() == 0:
        return float("nan")
    return float(valid.mean().item())


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    grad_clip_norm=1.0,
    amp_forward=False,
    unlabeled_loader=None,
    unsup_criterion=None,
    unsup_weight=0.0,
    teacher_model=None,
    ema_decay=0.99,
):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    meter = SegmentationMetricsMeter(len(CLASS_NAMES), CLASS_NAMES)
    amp_enabled = scaler is not None and scaler.is_enabled() and amp_forward
    progress_bar = tqdm(train_loader, desc="Training", unit="batch")
    unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None and unsup_criterion is not None and unsup_weight > 0 else None
    unsup_loss_sum = 0.0
    unsup_valid_ratio_sum = 0.0
    unsup_confidence_sum = 0.0
    unsup_foreground_ratio_sum = 0.0
    unsup_batch_count = 0

    for images, masks in progress_bar:
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        masks = prepare_target(masks.to(device, non_blocking=True))

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
        logits_for_debug = get_main_logits(outputs)
        if not torch.isfinite(logits_for_debug.detach()).all():
            print(f"Skip non-finite model output before loss: {describe_bad_batch(logits_for_debug, masks)}")
            optimizer.zero_grad(set_to_none=True)
            continue

        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(outputs, masks)

        unsup_loss_value = 0.0
        if unlabeled_iter is not None:
            (weak_images, strong_images), unlabeled_iter = next_unlabeled_batch(unlabeled_iter, unlabeled_loader)
            weak_images = weak_images.to(device, dtype=torch.float32, non_blocking=True)
            strong_images = strong_images.to(device, dtype=torch.float32, non_blocking=True)
            teacher_source = teacher_model if teacher_model is not None else model
            teacher_logits = teacher_forward(teacher_source, weak_images, amp_enabled=amp_enabled)
            model.train()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                strong_outputs = model(strong_images)
            strong_logits_for_debug = get_main_logits(strong_outputs)
            if not torch.isfinite(strong_logits_for_debug.detach()).all():
                print(f"Skip unlabeled loss because student output is non-finite: {describe_bad_batch(strong_logits_for_debug, masks)}")
            else:
                with torch.cuda.amp.autocast(enabled=False):
                    unsup_loss, unsup_stats = unsup_criterion(strong_outputs, teacher_logits)
                    loss = loss + unsup_weight * unsup_loss
                unsup_loss_value = float(unsup_loss.detach().cpu().item())
                unsup_loss_sum += unsup_loss_value
                unsup_valid_ratio_sum += unsup_stats["valid_ratio"]
                unsup_confidence_sum += unsup_stats["mean_confidence"]
                unsup_foreground_ratio_sum += unsup_stats["pseudo_foreground_ratio"]
                unsup_batch_count += 1

        if not torch.isfinite(loss):
            print(f"Skip non-finite training loss: {describe_bad_batch(logits_for_debug, masks)}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not gradients_are_finite(model):
                print(f"Skip optimizer step because gradients are non-finite: {describe_bad_batch(logits_for_debug, masks)}")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if not gradients_are_finite(model):
                print(f"Skip optimizer step because gradients are non-finite: {describe_bad_batch(logits_for_debug, masks)}")
                optimizer.zero_grad(set_to_none=True)
                continue
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        if teacher_model is not None:
            update_ema_model(model, teacher_model, ema_decay)

        logits = logits_for_debug.detach()
        class_losses = per_class_loss(
            logits,
            masks,
            len(CLASS_NAMES),
            class_weights=criterion.ce_loss.weight,
            dice_weight=criterion.dice_weight,
            ce_weight=criterion.ce_weight,
        )
        meter.update(logits, masks, loss.item(), class_losses)
        running = meter.compute()
        progress_bar.set_postfix(
            loss=f"{running['loss']:.4f}",
            dice=f"{running['mean_foreground_dice']:.4f}",
            iou=f"{running['mean_foreground_iou']:.4f}",
            unsup=f"{unsup_loss_value:.4f}",
        )

    metrics = meter.compute()
    if unsup_batch_count > 0:
        metrics["unsup_loss"] = unsup_loss_sum / unsup_batch_count
        metrics["unsup_valid_ratio"] = unsup_valid_ratio_sum / unsup_batch_count
        metrics["unsup_confidence"] = unsup_confidence_sum / unsup_batch_count
        metrics["unsup_pseudo_foreground_ratio"] = unsup_foreground_ratio_sum / unsup_batch_count
    else:
        metrics["unsup_loss"] = 0.0
        metrics["unsup_valid_ratio"] = 0.0
        metrics["unsup_confidence"] = 0.0
        metrics["unsup_pseudo_foreground_ratio"] = 0.0
    metrics["unsup_weight"] = float(unsup_weight)
    return metrics


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, desc="Validation"):
    model.eval()
    meter = SegmentationMetricsMeter(len(CLASS_NAMES), CLASS_NAMES)
    progress_bar = tqdm(data_loader, desc=desc, unit="batch")

    for images, masks in progress_bar:
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        masks = prepare_target(masks.to(device, non_blocking=True))
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(images)
            logits = get_main_logits(outputs).float()
            loss = criterion(logits, masks)
        class_losses = per_class_loss(
            logits,
            masks,
            len(CLASS_NAMES),
            class_weights=criterion.ce_loss.weight,
            dice_weight=criterion.dice_weight,
            ce_weight=criterion.ce_weight,
        )
        meter.update(logits, masks, loss.item(), class_losses)
        running = meter.compute()
        progress_bar.set_postfix(
            loss=f"{running['loss']:.4f}",
            dice=f"{running['mean_foreground_dice']:.4f}",
            iou=f"{running['mean_foreground_iou']:.4f}",
            acc=f"{running['overall_accuracy']:.4f}",
        )

    return meter.compute()


def parse_class_weights(class_weights, device):
    if not class_weights:
        return None
    values = [float(x.strip()) for x in class_weights.split(",") if x.strip()]
    if len(values) != len(CLASS_NAMES):
        raise ValueError(f"--class_weights must contain {len(CLASS_NAMES)} values.")
    return torch.tensor(values, dtype=torch.float32, device=device)


def parse_int_tuple(value, expected_len=None):
    values = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated integers, got {value}.")
    return values


def build_model(args):
    from PFM_Seg_Models import PFM_Seg_Model

    model_configs = {
        "UNI": ("UNI", "/home/pxy/anaconda3/bin/pytorch_model1.bin", 1024),
        "359999": ("UNI", "/home/wyz/code/Extra_features/ckpt/Cervix-weight/training_359999/teacher_checkpoint-UNI-format.pth", 1024),
        "Virchow_v2": ("Virchow_v2", "/home/wyz/code/Extra_features/ckpt/virchow2/Virchow_2_weights/pytorch_model.bin", 1280),
        "Conch_v1_5": ("Conch_v1_5", "/home/pxy/anaconda3/bin/Conch_1_5_weights/conch_v1_5_pytorch_model.bin", 1024),
    }
    pfm_name, weights_path, emb_dim = model_configs.get(
        args.model_name,
        ("UNI", "/home/wyz/code/Extra_features/ckpt/UNI/pytorch_model.bin", 1024),
    )
    if args.pfm_weights_path:
        weights_path = args.pfm_weights_path

    model = PFM_Seg_Model(
        PFM_name=pfm_name,
        PFM_weights_path=weights_path,
        emb_dim=emb_dim,
        frozen_PFM=True,
        img_size=args.image_size,
        num_classes=len(CLASS_NAMES),
        deep_supervision=args.deep_supervision,
        decoder_dropout=args.decoder_dropout,
        decoder_channels=parse_int_tuple(args.decoder_channels, expected_len=4),
        decoder_head_channels=args.decoder_head_channels,
        feature_noise_std=args.decoder_feature_noise_std,
    )
    unfreezed_names = model.unfreeze_last_blocks(args.unfreeze_pfm_blocks)
    return model, {
        "requested_model_name": args.model_name,
        "pfm_name": pfm_name,
        "pfm_weights_path": weights_path,
        "emb_dim": emb_dim,
        "unfreezed_pfm_parameters": unfreezed_names,
    }


def build_optimizer(model, args):
    decoder_params = []
    pfm_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("transformer"):
            pfm_params.append(param)
        else:
            decoder_params.append(param)

    param_groups = []
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": args.lr})
    if pfm_params:
        param_groups.append({"params": pfm_params, "lr": args.lr * args.pfm_lr_mult})
    return AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def summarize_model(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    state_dict_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "frozen_parameters": int(frozen_params),
        "trainable_ratio": trainable_params / max(total_params, 1),
        "state_dict_size_mb": state_dict_bytes / (1024 ** 2),
        "trainable_parameter_size_mb": trainable_bytes / (1024 ** 2),
    }


def write_model_files(save_dir, args, model, model_meta, fold_idx=None):
    summary = summarize_model(model)
    info = {
        "class_names": CLASS_NAMES,
        "foreground_classes_for_mean_metrics": [CLASS_NAMES[i] for i in FOREGROUND_CLASS_IDS],
        "args": vars(args),
        "fold": fold_idx,
        "model": model_meta,
        "summary": summary,
    }
    fold_suffix = f"fold{fold_idx}" if fold_idx is not None else f"fold{args.seed}"
    with open(os.path.join(save_dir, f"model_info_{fold_suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_dir, f"trainable_parameters_{fold_suffix}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "shape", "num_parameters", "requires_grad"])
        for name, param in model.named_parameters():
            writer.writerow([name, list(param.shape), int(param.numel()), bool(param.requires_grad)])

    with open(os.path.join(save_dir, f"model_summary_{fold_suffix}.txt"), "w", encoding="utf-8") as f:
        f.write("TransCH-Net model summary\n")
        if fold_idx is not None:
            f.write(f"fold: {fold_idx}\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\nTrainable parameters:\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"{name}: shape={list(param.shape)}, params={param.numel()}\n")


def flatten_metrics(prefix, metrics):
    row = {
        f"{prefix}_loss": metrics["loss"],
        f"{prefix}_mean_dice": metrics["mean_dice"],
        f"{prefix}_mean_iou": metrics["mean_iou"],
        f"{prefix}_mean_accuracy": metrics["mean_accuracy"],
        f"{prefix}_mean_foreground_dice": metrics["mean_foreground_dice"],
        f"{prefix}_mean_foreground_iou": metrics["mean_foreground_iou"],
        f"{prefix}_overall_accuracy": metrics["overall_accuracy"],
    }
    for class_name, class_metrics in metrics["class_metrics"].items():
        for key, value in class_metrics.items():
            row[f"{prefix}_{class_name}_{key}"] = value
    for key in ["unsup_loss", "unsup_weight", "unsup_valid_ratio", "unsup_confidence", "unsup_pseudo_foreground_ratio"]:
        if key in metrics:
            row[f"{prefix}_{key}"] = metrics[key]
    return row


def append_dict_csv(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_per_class_csv(path, epoch, split, metrics):
    write_header = not os.path.exists(path)
    fieldnames = ["epoch", "split", "class", "loss", "dice", "iou", "accuracy", "support_pixels", "pred_pixels"]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for class_name, class_metrics in metrics["class_metrics"].items():
            row = {"epoch": epoch, "split": split, "class": class_name}
            row.update(class_metrics)
            writer.writerow(row)


def print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics, elapsed, lr, fold_idx=None):
    fold_text = f"Fold [{fold_idx}] " if fold_idx is not None else ""
    semi_text = ""
    if train_metrics.get("unsup_weight", 0.0) > 0:
        semi_text = (
            f" unsup_weight={train_metrics['unsup_weight']:.3f}"
            f" unsup_loss={train_metrics['unsup_loss']:.4f}"
            f" pseudo_valid={train_metrics['unsup_valid_ratio']:.3f}"
        )
    print(
        f"{fold_text}Epoch [{epoch}/{num_epochs}] "
        f"lr={lr:.6g} "
        f"train_loss={train_metrics['loss']:.4f} "
        f"train_fg_dice={train_metrics['mean_foreground_dice']:.4f} "
        f"train_fg_iou={train_metrics['mean_foreground_iou']:.4f} "
        f"val_loss={val_metrics['loss']:.4f} "
        f"val_fg_dice={val_metrics['mean_foreground_dice']:.4f} "
        f"val_fg_iou={val_metrics['mean_foreground_iou']:.4f} "
        f"val_acc={val_metrics['overall_accuracy']:.4f} "
        f"time={elapsed:.2f}s"
        f"{semi_text}"
    )


def current_lr(optimizer):
    return min(group["lr"] for group in optimizer.param_groups)


def split_name_to_dirs(dataset_dir, split_name):
    split_name = split_name.strip()
    if split_name == "train":
        return (
            os.path.join(dataset_dir, "train/train_224patch"),
            os.path.join(dataset_dir, "train/train_224maskgray"),
        )
    if split_name in ("valid", "val"):
        return (
            os.path.join(dataset_dir, "valid/valid_224patch"),
            os.path.join(dataset_dir, "valid/valid_224maskgray"),
        )
    if split_name == "test":
        return (
            os.path.join(dataset_dir, "test/test_224patch"),
            os.path.join(dataset_dir, "test/test_224maskgray"),
        )
    return (
        os.path.join(dataset_dir, f"{split_name}/{split_name}_224patch"),
        os.path.join(dataset_dir, f"{split_name}/{split_name}_224maskgray"),
    )


def default_unlabeled_image_dir(dataset_dir):
    return os.path.join(dataset_dir, "unlabeled/unlabeled_224patch")


def parse_unlabeled_dirs(args):
    if args.unlabeled_image_dirs:
        return [path.strip() for path in args.unlabeled_image_dirs.split(",") if path.strip()]
    return [default_unlabeled_image_dir(args.dataset_dir)]


def build_unlabeled_loader(args, device):
    if not args.semi_supervised:
        return None, []
    image_dirs = parse_unlabeled_dirs(args)
    dataset = UnlabeledHistologyDataset(
        image_dirs=image_dirs,
        image_size=args.image_size,
        geometric_augmentation=get_unlabeled_geometric_augmentation(args.image_size),
        weak_color_augmentation=get_unlabeled_weak_color_augmentation(),
        strong_color_augmentation=get_unlabeled_strong_color_augmentation(),
        preprocessing=get_image_preprocessing(),
    )
    if len(dataset) == 0:
        raise ValueError(
            "Semi-supervised training is enabled, but no unlabeled images were found. "
            f"Checked: {image_dirs}"
        )
    batch_size = args.unlabeled_batch_size if args.unlabeled_batch_size > 0 else max(1, args.train_batch_size // 2)
    batch_size = min(batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= batch_size * 2,
    )
    return loader, image_dirs


def collect_cv_samples(args):
    split_names = [name.strip() for name in args.cv_source_splits.split(",") if name.strip()]
    records = []
    for split_name in split_names:
        images_dir, masks_dir = split_name_to_dirs(args.dataset_dir, split_name)
        split_records = HistologySegDataset.collect_samples(images_dir, masks_dir, split_name=split_name)
        records.extend(split_records)
    if not records:
        raise ValueError("No samples were found for cross-validation.")
    return records


def make_folds(num_samples, num_folds, seed):
    if num_folds < 2:
        raise ValueError("--num_folds must be at least 2 for cross-validation.")
    if num_samples < num_folds:
        raise ValueError(f"num_folds={num_folds} is larger than the number of samples={num_samples}.")
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_samples)
    return [fold.astype(np.int64) for fold in np.array_split(indices, num_folds)]


def records_from_indices(records, indices):
    return [records[int(idx)] for idx in indices]


def build_loader(sample_records, image_size, batch_size, num_workers, pin_memory, shuffle, is_train):
    dataset = HistologySegDataset(
        sample_records=sample_records,
        augmentation=get_training_augmentation(image_size) if is_train else get_validation_augmentation(image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def write_fold_split_file(save_dir, fold_idx, train_records, val_records, test_records):
    split_path = os.path.join(save_dir, f"fold{fold_idx}_splits.json")
    payload = {
        "fold": fold_idx,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "train_ids": [record["id"] for record in train_records],
        "val_ids": [record["id"] for record in val_records],
        "test_ids": [record["id"] for record in test_records],
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_fold(args, fold_idx, train_records, val_records, test_records, device):
    fold_dir = os.path.join(args.save_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    write_fold_split_file(fold_dir, fold_idx, train_records, val_records, test_records)

    pin_memory = device.type == "cuda"
    train_loader = build_loader(
        train_records,
        args.image_size,
        args.train_batch_size,
        args.num_workers,
        pin_memory,
        shuffle=True,
        is_train=True,
    )
    val_loader = build_loader(
        val_records,
        args.image_size,
        args.eval_batch_size,
        args.num_workers,
        pin_memory,
        shuffle=False,
        is_train=False,
    )
    test_loader = build_loader(
        test_records,
        args.image_size,
        args.eval_batch_size,
        args.num_workers,
        pin_memory,
        shuffle=False,
        is_train=False,
    )

    model, model_meta = build_model(args)
    model.to(device)
    write_model_files(fold_dir, args, model, model_meta, fold_idx=fold_idx)
    print(f"Fold [{fold_idx}] model summary:")
    print(json.dumps(summarize_model(model), indent=2))

    unlabeled_loader, unlabeled_dirs = build_unlabeled_loader(args, device)
    if unlabeled_loader is not None:
        print(
            f"Fold [{fold_idx}] semi-supervised training enabled: "
            f"unlabeled_images={len(unlabeled_loader.dataset)} batch_size={unlabeled_loader.batch_size} dirs={unlabeled_dirs}"
        )

    class_weights = parse_class_weights(args.class_weights, device)
    criterion = CompositeSegmentationLoss(
        num_classes=len(CLASS_NAMES),
        class_weights=class_weights,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        aux_loss_weight=args.aux_loss_weight,
        label_smoothing=args.label_smoothing,
    )
    unsup_criterion = None
    if args.semi_supervised:
        unsup_criterion = PseudoLabelLoss(
            num_classes=len(CLASS_NAMES),
            dice_weight=args.unsup_dice_weight,
            ce_weight=args.unsup_ce_weight,
            confidence_threshold=args.unsup_confidence_threshold,
            bg_confidence_threshold=args.unsup_bg_confidence_threshold,
            ignore_background=args.unsup_ignore_background,
        )
    optimizer = build_optimizer(model, args)
    scheduler = None if args.no_scheduler else CosineAnnealingLR(optimizer, T_max=max(args.epoch, 1), eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_forward and device.type == "cuda")
    if args.amp and not args.amp_forward:
        print("AMP flag detected, but fp16 forward is disabled for stability. Add --amp_forward only after fp32 training is stable.")
    teacher_model = None
    if args.semi_supervised and args.semi_teacher_mode == "ema":
        teacher_model = copy.deepcopy(model).to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    epoch_metrics_file = os.path.join(fold_dir, f"training_metrics_fold{fold_idx}.csv")
    per_class_file = os.path.join(fold_dir, f"per_class_metrics_fold{fold_idx}.csv")
    best_val_dice = -1.0
    best_model_path = os.path.join(fold_dir, "best_model.pth")
    last_model_path = os.path.join(fold_dir, "last.pth")

    for epoch in range(1, args.epoch + 1):
        start_time = time.time()
        lr = current_lr(optimizer)
        epoch_unsup_weight = rampup_weight(
            epoch,
            args.semi_start_epoch,
            args.unsup_rampup_epochs,
            args.unsup_weight if args.semi_supervised else 0.0,
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            grad_clip_norm=args.grad_clip_norm,
            amp_forward=args.amp_forward,
            unlabeled_loader=unlabeled_loader,
            unsup_criterion=unsup_criterion,
            unsup_weight=epoch_unsup_weight,
            teacher_model=teacher_model,
            ema_decay=args.ema_decay,
        )
        eval_model = teacher_model if teacher_model is not None else model
        val_metrics = evaluate(eval_model, val_loader, criterion, device, desc=f"Validation fold {fold_idx}")
        elapsed = time.time() - start_time

        row = {"fold": fold_idx, "epoch": epoch, "lr": lr, "time_seconds": elapsed}
        row.update(flatten_metrics("train", train_metrics))
        row.update(flatten_metrics("val", val_metrics))
        append_dict_csv(epoch_metrics_file, row)
        append_per_class_csv(per_class_file, epoch, "train", train_metrics)
        append_per_class_csv(per_class_file, epoch, "val", val_metrics)
        print_epoch_summary(epoch, args.epoch, train_metrics, val_metrics, elapsed, lr, fold_idx=fold_idx)

        if val_metrics["mean_foreground_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_foreground_dice"]
            model_to_save = teacher_model if teacher_model is not None else model
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f"Fold [{fold_idx}] best model saved at epoch {epoch} with foreground Dice: {best_val_dice:.4f}")

        model_to_save = teacher_model if teacher_model is not None else model
        torch.save(model_to_save.state_dict(), last_model_path)
        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, desc=f"Test fold {fold_idx}")

    test_row = {"fold": fold_idx, "best_val_mean_foreground_dice": best_val_dice}
    test_row.update(flatten_metrics("test", test_metrics))
    test_file = os.path.join(fold_dir, f"test_results_fold{fold_idx}.csv")
    append_dict_csv(test_file, test_row)
    append_per_class_csv(per_class_file, "best", "test", test_metrics)

    with open(os.path.join(fold_dir, f"test_results_fold{fold_idx}.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print(
        f"Fold [{fold_idx}] Test Results - Loss: {test_metrics['loss']:.4f}, "
        f"Foreground Dice: {test_metrics['mean_foreground_dice']:.4f}, "
        f"Foreground IoU: {test_metrics['mean_foreground_iou']:.4f}, "
        f"Pixel Accuracy: {test_metrics['overall_accuracy']:.4f}"
    )
    return test_row, test_metrics


def numeric_mean_std(values):
    arr = np.asarray([value for value in values if value == value], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def write_cross_validation_summary(save_dir, fold_rows, fold_metrics):
    fold_results_path = os.path.join(save_dir, "cross_validation_fold_results.csv")
    with open(fold_results_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(fold_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_rows)

    summary = {}
    metric_keys = [
        "loss",
        "mean_dice",
        "mean_iou",
        "mean_accuracy",
        "mean_foreground_dice",
        "mean_foreground_iou",
        "overall_accuracy",
    ]
    for key in metric_keys:
        mean_value, std_value = numeric_mean_std([metrics[key] for metrics in fold_metrics])
        summary[f"test_{key}_mean"] = mean_value
        summary[f"test_{key}_std"] = std_value

    per_class_summary = {}
    for class_name in CLASS_NAMES:
        per_class_summary[class_name] = {}
        for key in ["loss", "dice", "iou", "accuracy"]:
            mean_value, std_value = numeric_mean_std(
                [metrics["class_metrics"][class_name][key] for metrics in fold_metrics]
            )
            per_class_summary[class_name][f"{key}_mean"] = mean_value
            per_class_summary[class_name][f"{key}_std"] = std_value

    summary_payload = {
        "num_folds": len(fold_rows),
        "summary": summary,
        "per_class_summary": per_class_summary,
        "fold_results": fold_rows,
    }
    with open(os.path.join(save_dir, "cross_validation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    summary_csv_path = os.path.join(save_dir, "cross_validation_summary.csv")
    flat_summary = dict(summary)
    for class_name, class_values in per_class_summary.items():
        for key, value in class_values.items():
            flat_summary[f"test_{class_name}_{key}"] = value
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_summary.keys()))
        writer.writeheader()
        writer.writerow(flat_summary)

    print("Five-fold cross-validation summary:")
    print(
        f"Dice {summary['test_mean_foreground_dice_mean']:.4f} +/- {summary['test_mean_foreground_dice_std']:.4f}, "
        f"IoU {summary['test_mean_foreground_iou_mean']:.4f} +/- {summary['test_mean_foreground_iou_std']:.4f}, "
        f"Accuracy {summary['test_overall_accuracy_mean']:.4f} +/- {summary['test_overall_accuracy_std']:.4f}"
    )
    return summary, per_class_summary


def write_legacy_final_results(save_dir, seed, summary, per_class_summary):
    results_file = os.path.join(save_dir, f"test_results_fold{seed}.csv")
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Std"])
        writer.writerow(["Test Loss", summary["test_loss_mean"], summary["test_loss_std"]])
        writer.writerow(["Test Dice", summary["test_mean_foreground_dice_mean"], summary["test_mean_foreground_dice_std"]])
        writer.writerow(["Test IoU", summary["test_mean_foreground_iou_mean"], summary["test_mean_foreground_iou_std"]])
        writer.writerow(["Test Pixel Accuracy", summary["test_overall_accuracy_mean"], summary["test_overall_accuracy_std"]])
        writer.writerow([])
        writer.writerow(["Class", "Loss Mean", "Loss Std", "Dice Mean", "Dice Std", "IoU Mean", "IoU Std", "Accuracy Mean", "Accuracy Std"])
        for class_name, class_summary in per_class_summary.items():
            writer.writerow(
                [
                    class_name,
                    class_summary["loss_mean"],
                    class_summary["loss_std"],
                    class_summary["dice_mean"],
                    class_summary["dice_std"],
                    class_summary["iou_mean"],
                    class_summary["iou_std"],
                    class_summary["accuracy_mean"],
                    class_summary["accuracy_std"],
                ]
            )


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cv_records = collect_cv_samples(args)
    test_images_dir, test_masks_dir = split_name_to_dirs(args.dataset_dir, "test")
    test_records = HistologySegDataset.collect_samples(test_images_dir, test_masks_dir, split_name="test")
    folds = make_folds(len(cv_records), args.num_folds, args.seed)

    fold_rows = []
    fold_metrics = []
    for fold_idx, val_indices in enumerate(folds, start=1):
        val_index_set = set(int(idx) for idx in val_indices)
        train_indices = [idx for idx in range(len(cv_records)) if idx not in val_index_set]
        train_records = records_from_indices(cv_records, train_indices)
        val_records = records_from_indices(cv_records, val_indices)

        print(
            f"Starting fold [{fold_idx}/{args.num_folds}] "
            f"train={len(train_records)} val={len(val_records)} test={len(test_records)}"
        )
        fold_row, metrics = run_fold(args, fold_idx, train_records, val_records, test_records, device)
        fold_rows.append(fold_row)
        fold_metrics.append(metrics)

    summary, per_class_summary = write_cross_validation_summary(args.save_dir, fold_rows, fold_metrics)
    write_legacy_final_results(args.save_dir, args.seed, summary, per_class_summary)
    print(f"Logs saved to {args.save_dir}")


if __name__ == "__main__":
    main()
