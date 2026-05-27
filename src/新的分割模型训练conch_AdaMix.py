import copy
import csv
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "新的分割模型训练conch.py"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

spec = importlib.util.spec_from_file_location("base_conch_training", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class UnlabeledImageDataset(TorchDataset):
    image_exts = base.HistologySegDataset.image_exts

    def __init__(self, image_dirs, image_size, augmentation=None, preprocessing=None):
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        self.image_dirs = [path for path in image_dirs if path]
        self.image_size = image_size
        self.samples = self.collect_images(self.image_dirs)
        self.ids = [record["id"] for record in self.samples]
        self.images_fps = [record["image_path"] for record in self.samples]
        self.augmentation = augmentation or base.get_training_augmentation(image_size)
        self.preprocessing = preprocessing or base.get_image_preprocessing()

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
        if self.augmentation:
            image = self.augmentation(image=image)["image"]
        if self.preprocessing:
            image = self.preprocessing(image=image)["image"]
        return image

    def __len__(self):
        return len(self.images_fps)


class AdaptiveMix2D(torch.nn.Module):
    def __init__(
        self,
        image_size,
        patch_divisor=14,
        topk=16,
        p=0.5,
        total_steps=0,
        self_paced=True,
        device="cuda",
    ):
        super().__init__()
        if image_size % patch_divisor != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by adamix_patch_divisor={patch_divisor}."
            )
        self.image_size = image_size
        self.patch_divisor = patch_divisor
        self.patch_size = image_size // patch_divisor
        self.topk = topk
        self.p = p
        self.total_steps = total_steps
        self.self_paced = self_paced
        self.device = device
        self.unfold = torch.nn.Unfold(
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
        ).to(device)
        self.fold = torch.nn.Fold(
            output_size=(image_size, image_size),
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
        ).to(device)

    @staticmethod
    def sigmoid_rampup(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def increase_age(self, cur_step):
        return self.sigmoid_rampup(cur_step, self.total_steps)

    @staticmethod
    def dice_loss_per_sample(logits, target, num_classes, class_weights=None):
        probs = torch.softmax(logits.float(), dim=1)
        target = target.long()
        target_one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        dims = (2, 3)
        intersection = (probs * target_one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
        dice = 1.0 - (2.0 * intersection + 1e-7) / (denominator + 1e-7)
        valid_classes = target_one_hot.sum(dim=dims) > 0
        dice_sum = torch.zeros(logits.shape[0], device=logits.device)
        for idx in range(logits.shape[0]):
            if valid_classes[idx].any():
                dice_sum[idx] = dice[idx][valid_classes[idx]].mean()
        ce = F.cross_entropy(
            logits.float(),
            target,
            weight=class_weights,
            reduction="none",
        ).mean(dim=(1, 2))
        return dice_sum, ce

    @staticmethod
    def proxy_loss(logits, target, num_classes, class_weights=None, dice_weight=1.0, ce_weight=0.5):
        dice, ce = AdaptiveMix2D.dice_loss_per_sample(logits, target, num_classes, class_weights=class_weights)
        return dice_weight * dice + ce_weight * ce

    def patch_scores(self, confidence_map):
        conf = confidence_map.unsqueeze(1).float()
        conf_unfold = self.unfold(conf)
        conf_unfold = conf_unfold.view(conf.shape[0], 1, self.patch_size, self.patch_size, -1)
        return conf_unfold.mean(dim=(1, 2, 3))

    def mix(self, oimage, aimage, olabel, alabel, oconf, aconf, proxy_loss, cur_step):
        if torch.rand(1, device=oimage.device).item() >= self.p:
            return oimage, olabel, oconf

        age = self.increase_age(cur_step) if self.self_paced else 1.0
        proxy_loss = proxy_loss.detach().float().view(-1)
        sp_mask = proxy_loss < age
        sp_weight = torch.clamp(1.0 - proxy_loss / (age + 1e-5), min=0.0, max=1.0)

        B, C, H, W = oimage.shape
        oimage_u = self.unfold(oimage).view(B, C, self.patch_size, self.patch_size, -1)
        aimage_u = self.unfold(aimage).view(B, C, self.patch_size, self.patch_size, -1)
        olabel_u = self.unfold(olabel.unsqueeze(1).float()).view(B, 1, self.patch_size, self.patch_size, -1)
        alabel_u = self.unfold(alabel.unsqueeze(1).float()).view(B, 1, self.patch_size, self.patch_size, -1)
        oconf_u = self.unfold(oconf.unsqueeze(1).float()).view(B, 1, self.patch_size, self.patch_size, -1)
        aconf_u = self.unfold(aconf.unsqueeze(1).float()).view(B, 1, self.patch_size, self.patch_size, -1)

        oconf_scores = self.patch_scores(oconf)
        aconf_scores = self.patch_scores(aconf)

        for i in range(B):
            n = int(min(self.topk, max(0, round(self.topk * float(sp_weight[i].item())))))
            if n <= 0:
                continue
            if bool(sp_mask[i].item()):
                src_idx = torch.argsort(oconf_scores[i], descending=True)[:n]
                dst_idx = torch.argsort(aconf_scores[i], descending=False)[:n]
            else:
                src_idx = torch.argsort(oconf_scores[i], descending=False)[:n]
                dst_idx = torch.argsort(aconf_scores[i], descending=True)[:n]
            oimage_u[i, :, :, :, src_idx] = aimage_u[i, :, :, :, dst_idx]
            olabel_u[i, :, :, :, src_idx] = alabel_u[i, :, :, :, dst_idx]
            oconf_u[i, :, :, :, src_idx] = aconf_u[i, :, :, :, dst_idx]

        mixed_image = self.fold(oimage_u.view(B, C * self.patch_size * self.patch_size, -1))
        mixed_label = self.fold(olabel_u.view(B, 1 * self.patch_size * self.patch_size, -1)).squeeze(1).long()
        mixed_conf = self.fold(oconf_u.view(B, 1 * self.patch_size * self.patch_size, -1)).squeeze(1)
        return mixed_image, mixed_label, mixed_conf


def build_parser():
    parser = base.build_parser()
    parser.description = "TransCH-Net + Conch AdaMix semi-supervised segmentation training"
    parser.set_defaults(semi_supervised=True, epoch=50)
    parser.add_argument("--adamix_patch_divisor", type=int, default=14, help="Image-size divisor for AdaMix patch regions. 448/14 = 32px patches.")
    parser.add_argument("--adamix_topk", type=int, default=16, help="Maximum number of AdaMix patch swaps per image.")
    parser.add_argument("--adamix_prob", type=float, default=0.5, help="Probability of applying AdaMix on a batch.")
    parser.add_argument("--adamix_no_self_paced", action="store_true", help="Disable the self-paced curriculum in AdaMix.")
    parser.add_argument("--adamix_use_labeled", action="store_true", help="Apply AdaMix to labeled batches as well as unlabeled batches.")
    parser.add_argument("--rerun_completed_folds", action="store_true", help="Do not skip folds that already finished the requested number of epochs.")
    parser.add_argument("--no_auto_resume", action="store_true", help="Disable automatic resume from save_dir/resume_checkpoint.pth.")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Optional explicit checkpoint path. Defaults to save_dir/resume_checkpoint.pth.")
    parser.add_argument("--resume_save_every", type=int, default=100, help="Save a resumable checkpoint every N training batches.")
    return parser


def set_seed(seed):
    base.set_seed(seed)


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def meter_state_dict(meter):
    return {
        "confusion": meter.confusion.clone(),
        "loss_sum": meter.loss_sum,
        "sample_count": meter.sample_count,
        "class_loss_sum": meter.class_loss_sum.copy(),
        "class_loss_count": meter.class_loss_count.copy(),
    }


def load_meter_state(meter, state):
    if not state:
        return
    meter.confusion = state["confusion"].clone()
    meter.loss_sum = float(state["loss_sum"])
    meter.sample_count = int(state["sample_count"])
    meter.class_loss_sum = np.asarray(state["class_loss_sum"], dtype=np.float64)
    meter.class_loss_count = np.asarray(state["class_loss_count"], dtype=np.float64)


def atomic_torch_save(payload, path):
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def default_resume_path(args):
    return args.resume_checkpoint or os.path.join(args.save_dir, "resume_checkpoint.pth")


def build_resumable_loader(records, args, pin_memory, shuffle_seed, start_batch):
    dataset = base.HistologySegDataset(
        sample_records=records,
        augmentation=base.get_training_augmentation(args.image_size),
    )
    indices = list(range(len(dataset)))
    rng = np.random.RandomState(shuffle_seed)
    rng.shuffle(indices)
    offset = min(max(int(start_batch), 0) * args.train_batch_size, len(indices))
    subset = Subset(dataset, indices[offset:])
    loader = DataLoader(
        subset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader, len(indices), offset


def resume_state_matches(checkpoint, args, fold_idx):
    if not checkpoint:
        return False
    if "model" not in checkpoint or "optimizer" not in checkpoint:
        return False
    if int(checkpoint.get("fold_idx", -1)) != int(fold_idx):
        return False
    saved_args = checkpoint.get("args", {})
    keys = [
        "dataset_dir",
        "image_size",
        "train_batch_size",
        "num_folds",
        "seed",
        "model_name",
        "adamix_patch_divisor",
    ]
    for key in keys:
        if key in saved_args and hasattr(args, key):
            if str(saved_args[key]) != str(getattr(args, key)):
                return False
    return True


def save_resume_checkpoint(
    path,
    args,
    fold_idx,
    epoch,
    next_batch,
    model,
    optimizer,
    scheduler,
    scaler,
    teacher_model,
    meter,
    train_stats,
    best_val_dice,
    completed_folds,
):
    payload = {
        "version": 1,
        "args": vars(args),
        "fold_idx": fold_idx,
        "epoch": epoch,
        "next_batch": next_batch,
        "best_val_dice": best_val_dice,
        "completed_folds": list(completed_folds),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "teacher_model": teacher_model.state_dict() if teacher_model is not None else None,
        "meter": meter_state_dict(meter) if meter is not None else None,
        "train_stats": train_stats,
        "rng_state": get_rng_state(),
    }
    atomic_torch_save(payload, path)


def save_light_resume_checkpoint(path, args, fold_idx, completed_folds):
    payload = {
        "version": 1,
        "args": vars(args),
        "fold_idx": fold_idx,
        "epoch": 1,
        "next_batch": 0,
        "best_val_dice": -1.0,
        "completed_folds": list(completed_folds),
        "rng_state": get_rng_state(),
    }
    atomic_torch_save(payload, path)


def pair_permutation(batch_size, device):
    if batch_size <= 1:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    perm = torch.randperm(batch_size, device=device)
    if torch.all(perm == torch.arange(batch_size, device=device)):
        perm = torch.roll(perm, shifts=1)
    return perm


def get_unlabeled_loader(args, device):
    if not args.semi_supervised:
        return None, []
    image_dirs = base.parse_unlabeled_dirs(args)
    dataset = UnlabeledImageDataset(
        image_dirs=image_dirs,
        image_size=args.image_size,
        augmentation=base.get_training_augmentation(args.image_size),
        preprocessing=base.get_image_preprocessing(),
    )
    if len(dataset) == 0:
        raise ValueError(
            "AdaMix semi-supervised training is enabled, but no unlabeled images were found. "
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


def prepare_labels(masks):
    return base.prepare_target(masks)


def fold_reached_epoch(save_dir, fold_idx, target_epoch):
    metrics_file = os.path.join(save_dir, f"fold_{fold_idx}", f"training_metrics_fold{fold_idx}.csv")
    if not os.path.exists(metrics_file):
        return False, 0
    max_epoch = 0
    with open(metrics_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_epoch = max(max_epoch, int(float(row.get("epoch", 0))))
            except (TypeError, ValueError):
                continue
    return max_epoch >= target_epoch, max_epoch


def load_existing_test_result(save_dir, fold_idx):
    fold_dir = os.path.join(save_dir, f"fold_{fold_idx}")
    json_path = os.path.join(fold_dir, f"test_results_fold{fold_idx}.json")
    csv_path = os.path.join(fold_dir, f"test_results_fold{fold_idx}.csv")
    if not os.path.exists(json_path) or not os.path.exists(csv_path):
        return None, None
    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    row = {}
    for key, value in rows[-1].items():
        try:
            row[key] = float(value)
        except (TypeError, ValueError):
            row[key] = value
    if "fold" in row:
        row["fold"] = int(float(row["fold"]))
    return row, metrics


def evaluate_completed_fold(args, fold_idx, test_records, device):
    existing_row, existing_metrics = load_existing_test_result(args.save_dir, fold_idx)
    if existing_row is not None and existing_metrics is not None:
        print(f"Fold [{fold_idx}] already has test results; loading existing result.")
        return existing_row, existing_metrics

    fold_dir = os.path.join(args.save_dir, f"fold_{fold_idx}")
    best_model_path = os.path.join(fold_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Fold [{fold_idx}] reached epoch {args.epoch}, but best_model.pth was not found: {best_model_path}"
        )

    pin_memory = device.type == "cuda"
    test_loader = base.build_loader(
        test_records,
        args.image_size,
        args.eval_batch_size,
        args.num_workers,
        pin_memory,
        shuffle=False,
        is_train=False,
    )
    model, _ = base.build_model(args)
    model.to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    class_weights = base.parse_class_weights(args.class_weights, device)
    criterion = base.CompositeSegmentationLoss(
        num_classes=len(base.CLASS_NAMES),
        class_weights=class_weights,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        aux_loss_weight=args.aux_loss_weight,
        label_smoothing=args.label_smoothing,
    )
    test_metrics = base.evaluate(model, test_loader, criterion, device, desc=f"Test fold {fold_idx}")

    best_val_dice = -1.0
    metrics_file = os.path.join(fold_dir, f"training_metrics_fold{fold_idx}.csv")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    best_val_dice = max(best_val_dice, float(row.get("val_mean_foreground_dice", -1.0)))
                except (TypeError, ValueError):
                    continue

    test_row = {"fold": fold_idx, "best_val_mean_foreground_dice": best_val_dice}
    test_row.update(base.flatten_metrics("test", test_metrics))
    test_file = os.path.join(fold_dir, f"test_results_fold{fold_idx}.csv")
    base.append_dict_csv(test_file, test_row)
    per_class_file = os.path.join(fold_dir, f"per_class_metrics_fold{fold_idx}.csv")
    base.append_per_class_csv(per_class_file, "best", "test", test_metrics)
    with open(os.path.join(fold_dir, f"test_results_fold{fold_idx}.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    return test_row, test_metrics


def masked_multiclass_loss(
    outputs,
    target,
    confidence_map,
    class_weights=None,
    dice_weight=1.0,
    ce_weight=0.5,
    fg_threshold=0.75,
    bg_threshold=0.98,
    ignore_background=False,
):
    logits = base.get_main_logits(outputs).float()
    target = target.long()
    confidence_map = confidence_map.float()
    valid = confidence_map >= fg_threshold
    valid = valid | ((target == 0) & (confidence_map >= bg_threshold))
    if ignore_background:
        valid = valid & (target != 0)

    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

    if not valid.any():
        zero = logits.sum() * 0.0
        return zero, {
            "valid_ratio": 0.0,
            "mean_confidence": float(confidence_map.mean().detach().cpu().item()),
            "pseudo_foreground_ratio": 0.0,
        }

    masked_target = target.clone()
    masked_target[~valid] = 255
    ce = F.cross_entropy(logits, masked_target, weight=class_weights, ignore_index=255)

    probs = torch.softmax(logits.float(), dim=1)
    target_one_hot = F.one_hot(target.clamp(0, logits.shape[1] - 1), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    valid_mask = valid.unsqueeze(1).float()
    probs = probs * valid_mask
    target_one_hot = target_one_hot * valid_mask
    dims = (2, 3)
    intersection = (probs * target_one_hot).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice = 1.0 - (2.0 * intersection + 1e-7) / (denominator + 1e-7)
    valid_classes = target_one_hot.sum(dim=dims) > 0
    dice_values = []
    for idx in range(logits.shape[0]):
        if valid_classes[idx].any():
            dice_values.append(dice[idx][valid_classes[idx]].mean())
        else:
            dice_values.append(torch.tensor(0.0, device=logits.device))
    dice_loss = torch.stack(dice_values).mean()

    loss = dice_weight * dice_loss + ce_weight * ce
    stats = {
        "valid_ratio": float(valid.float().mean().detach().cpu().item()),
        "mean_confidence": float(confidence_map.mean().detach().cpu().item()),
        "pseudo_foreground_ratio": float(((target > 0) & valid).float().mean().detach().cpu().item()),
    }
    return loss, stats


def train_one_epoch_adamix(
    model,
    train_loader,
    unlabeled_loader,
    criterion,
    optimizer,
    device,
    adamix,
    scaler=None,
    grad_clip_norm=1.0,
    amp_forward=False,
    unsup_weight=0.0,
    teacher_model=None,
    ema_decay=0.99,
    class_weights=None,
    fg_threshold=0.75,
    bg_threshold=0.98,
    ignore_background=False,
    use_labeled_adamix=True,
    epoch=1,
    start_batch=0,
    resume_meter_state=None,
    resume_train_stats=None,
    checkpoint_context=None,
):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    meter = base.SegmentationMetricsMeter(len(base.CLASS_NAMES), base.CLASS_NAMES)
    load_meter_state(meter, resume_meter_state)
    amp_enabled = scaler is not None and scaler.is_enabled() and amp_forward
    progress_bar = tqdm(train_loader, desc="Training AdaMix", unit="batch")
    unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None and unsup_weight > 0 else None

    if resume_train_stats:
        unsup_loss_sum = float(resume_train_stats.get("unsup_loss_sum", 0.0))
        unsup_valid_ratio_sum = float(resume_train_stats.get("unsup_valid_ratio_sum", 0.0))
        unsup_confidence_sum = float(resume_train_stats.get("unsup_confidence_sum", 0.0))
        unsup_foreground_ratio_sum = float(resume_train_stats.get("unsup_foreground_ratio_sum", 0.0))
        unsup_batch_count = int(resume_train_stats.get("unsup_batch_count", 0))
    else:
        unsup_loss_sum = 0.0
        unsup_valid_ratio_sum = 0.0
        unsup_confidence_sum = 0.0
        unsup_foreground_ratio_sum = 0.0
        unsup_batch_count = 0

    save_every = 0
    if checkpoint_context is not None:
        save_every = int(checkpoint_context.get("save_every", 0) or 0)

    for local_step, (images, masks) in enumerate(progress_bar):
        step = start_batch + local_step
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        masks = prepare_labels(masks.to(device, non_blocking=True))

        optimizer.zero_grad(set_to_none=True)

        if use_labeled_adamix:
            labeled_perm = pair_permutation(images.shape[0], images.device)
            labeled_aux_images = images[labeled_perm]
            labeled_aux_masks = masks[labeled_perm]
            with torch.no_grad():
                labeled_student_logits = base.teacher_forward(model, images, amp_enabled=amp_enabled)
                labeled_student_aux_logits = base.teacher_forward(model, labeled_aux_images, amp_enabled=amp_enabled)
                labeled_o_conf = torch.softmax(labeled_student_logits.float(), dim=1).max(dim=1)[0]
                labeled_a_conf = torch.softmax(labeled_student_aux_logits.float(), dim=1).max(dim=1)[0]
                labeled_proxy = 0.5 * (
                    AdaptiveMix2D.proxy_loss(
                        labeled_student_logits,
                        masks,
                        len(base.CLASS_NAMES),
                        class_weights=class_weights,
                        dice_weight=criterion.dice_weight,
                        ce_weight=criterion.ce_weight,
                    )
                    + AdaptiveMix2D.proxy_loss(
                        labeled_student_aux_logits,
                        labeled_aux_masks,
                        len(base.CLASS_NAMES),
                        class_weights=class_weights,
                        dice_weight=criterion.dice_weight,
                        ce_weight=criterion.ce_weight,
                    )
                )
            mixed_l_images, mixed_l_masks, mixed_l_conf = adamix.mix(
                images,
                labeled_aux_images,
                masks,
                labeled_aux_masks,
                labeled_o_conf,
                labeled_a_conf,
                labeled_proxy,
                cur_step=step + len(progress_bar) * 0,
            )
        else:
            mixed_l_images, mixed_l_masks, mixed_l_conf = images, masks, torch.ones_like(masks, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            labeled_outputs = model(mixed_l_images)
        labeled_logits_for_debug = base.get_main_logits(labeled_outputs)
        if not torch.isfinite(labeled_logits_for_debug.detach()).all():
            print(f"Skip non-finite labeled output before loss: {base.describe_bad_batch(labeled_logits_for_debug, mixed_l_masks)}")
            optimizer.zero_grad(set_to_none=True)
            continue
        with torch.cuda.amp.autocast(enabled=False):
            loss_s = criterion(labeled_outputs, mixed_l_masks)

        loss_u = torch.tensor(0.0, device=device)
        unsup_loss_value = 0.0
        if unlabeled_iter is not None:
            try:
                uimages = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                uimages = next(unlabeled_iter)
            uimages = uimages.to(device, dtype=torch.float32, non_blocking=True)
            u_perm = pair_permutation(uimages.shape[0], uimages.device)
            uaux_images = uimages[u_perm]

            teacher_source = teacher_model if teacher_model is not None else model
            with torch.no_grad():
                teacher_logits_u = base.teacher_forward(teacher_source, uimages, amp_enabled=amp_enabled)
                teacher_logits_ua = base.teacher_forward(teacher_source, uaux_images, amp_enabled=amp_enabled)
                pseudo_u = torch.argmax(torch.softmax(teacher_logits_u.float(), dim=1), dim=1)
                pseudo_ua = torch.argmax(torch.softmax(teacher_logits_ua.float(), dim=1), dim=1)

                if teacher_model is None:
                    student_logits_u = teacher_logits_u
                    student_logits_ua = teacher_logits_ua
                else:
                    student_logits_u = base.teacher_forward(model, uimages, amp_enabled=amp_enabled)
                    student_logits_ua = base.teacher_forward(model, uaux_images, amp_enabled=amp_enabled)
                u_o_conf = torch.softmax(student_logits_u.float(), dim=1).max(dim=1)[0]
                u_a_conf = torch.softmax(student_logits_ua.float(), dim=1).max(dim=1)[0]
                u_proxy = 0.5 * (
                    AdaptiveMix2D.proxy_loss(
                        student_logits_u,
                        pseudo_u,
                        len(base.CLASS_NAMES),
                        class_weights=class_weights,
                        dice_weight=criterion.dice_weight,
                        ce_weight=criterion.ce_weight,
                    )
                    + AdaptiveMix2D.proxy_loss(
                        student_logits_ua,
                        pseudo_ua,
                        len(base.CLASS_NAMES),
                        class_weights=class_weights,
                        dice_weight=criterion.dice_weight,
                        ce_weight=criterion.ce_weight,
                    )
                )

            mixed_u_images, mixed_u_pseudo, mixed_u_conf = adamix.mix(
                uimages,
                uaux_images,
                pseudo_u,
                pseudo_ua,
                u_o_conf,
                u_a_conf,
                u_proxy,
                cur_step=step + len(progress_bar),
            )
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                unlabeled_outputs = model(mixed_u_images)
            unlabeled_logits_for_debug = base.get_main_logits(unlabeled_outputs)
            if not torch.isfinite(unlabeled_logits_for_debug.detach()).all():
                print(
                    f"Skip unlabeled AdaMix loss because student output is non-finite: "
                    f"{base.describe_bad_batch(unlabeled_logits_for_debug, mixed_u_pseudo)}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            with torch.cuda.amp.autocast(enabled=False):
                loss_u, unsup_stats = masked_multiclass_loss(
                    unlabeled_outputs,
                    mixed_u_pseudo,
                    mixed_u_conf,
                    class_weights=class_weights,
                    dice_weight=criterion.dice_weight,
                    ce_weight=criterion.ce_weight,
                    fg_threshold=fg_threshold,
                    bg_threshold=bg_threshold,
                    ignore_background=ignore_background,
                )
            unsup_loss_value = float(loss_u.detach().cpu().item())
            unsup_loss_sum += unsup_loss_value
            unsup_valid_ratio_sum += unsup_stats["valid_ratio"]
            unsup_confidence_sum += unsup_stats["mean_confidence"]
            unsup_foreground_ratio_sum += unsup_stats["pseudo_foreground_ratio"]
            unsup_batch_count += 1

        loss = loss_s + unsup_weight * loss_u

        if not torch.isfinite(loss):
            print(f"Skip non-finite training loss: {base.describe_bad_batch(labeled_logits_for_debug, mixed_l_masks)}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not base.gradients_are_finite(model):
                print(f"Skip optimizer step because gradients are non-finite: {base.describe_bad_batch(labeled_logits_for_debug, mixed_l_masks)}")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if not base.gradients_are_finite(model):
                print(f"Skip optimizer step because gradients are non-finite: {base.describe_bad_batch(labeled_logits_for_debug, mixed_l_masks)}")
                optimizer.zero_grad(set_to_none=True)
                continue
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if teacher_model is not None:
            base.update_ema_model(model, teacher_model, ema_decay)

        logits = labeled_logits_for_debug.detach()
        class_losses = base.per_class_loss(
            logits,
            mixed_l_masks,
            len(base.CLASS_NAMES),
            class_weights=class_weights,
            dice_weight=criterion.dice_weight,
            ce_weight=criterion.ce_weight,
        )
        meter.update(logits, mixed_l_masks, loss.item(), class_losses)
        running = meter.compute()
        progress_bar.set_postfix(
            loss=f"{running['loss']:.4f}",
            dice=f"{running['mean_foreground_dice']:.4f}",
            iou=f"{running['mean_foreground_iou']:.4f}",
            unsup=f"{unsup_loss_value:.4f}",
        )

        if save_every > 0 and checkpoint_context is not None and (step + 1) % save_every == 0:
            train_stats = {
                "unsup_loss_sum": unsup_loss_sum,
                "unsup_valid_ratio_sum": unsup_valid_ratio_sum,
                "unsup_confidence_sum": unsup_confidence_sum,
                "unsup_foreground_ratio_sum": unsup_foreground_ratio_sum,
                "unsup_batch_count": unsup_batch_count,
            }
            save_resume_checkpoint(
                checkpoint_context["path"],
                checkpoint_context["args"],
                checkpoint_context["fold_idx"],
                epoch,
                step + 1,
                model,
                optimizer,
                checkpoint_context.get("scheduler"),
                scaler,
                teacher_model,
                meter,
                train_stats,
                checkpoint_context.get("best_val_dice", -1.0),
                checkpoint_context.get("completed_folds", []),
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


def run_fold(args, fold_idx, train_records, val_records, test_records, device, resume_checkpoint=None, completed_folds=None):
    completed_folds = completed_folds or []
    fold_dir = os.path.join(args.save_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    base.write_fold_split_file(fold_dir, fold_idx, train_records, val_records, test_records)

    pin_memory = device.type == "cuda"
    resume_epoch = int(resume_checkpoint.get("epoch", 1)) if resume_checkpoint else 1
    resume_next_batch = int(resume_checkpoint.get("next_batch", 0)) if resume_checkpoint else 0
    if resume_checkpoint and not resume_state_matches(resume_checkpoint, args, fold_idx):
        print(f"Ignore incompatible resume checkpoint for fold {fold_idx}.")
        resume_checkpoint = None
        resume_epoch = 1
        resume_next_batch = 0

    val_loader = base.build_loader(
        val_records,
        args.image_size,
        args.eval_batch_size,
        args.num_workers,
        pin_memory,
        shuffle=False,
        is_train=False,
    )
    test_loader = base.build_loader(
        test_records,
        args.image_size,
        args.eval_batch_size,
        args.num_workers,
        pin_memory,
        shuffle=False,
        is_train=False,
    )
    unlabeled_loader, unlabeled_dirs = get_unlabeled_loader(args, device)

    model, model_meta = base.build_model(args)
    model.to(device)
    model_meta["semi_supervised_variant"] = "AdaMix"
    model_meta["adamix_patch_divisor"] = args.adamix_patch_divisor
    model_meta["adamix_topk"] = args.adamix_topk
    model_meta["adamix_prob"] = args.adamix_prob
    model_meta["adamix_use_labeled"] = args.adamix_use_labeled
    base.write_model_files(fold_dir, args, model, model_meta, fold_idx=fold_idx)
    print(f"Fold [{fold_idx}] AdaMix model summary:")
    print(json.dumps(base.summarize_model(model), indent=2))
    if unlabeled_loader is not None:
        print(
            f"Fold [{fold_idx}] AdaMix semi-supervised training enabled: "
            f"unlabeled_images={len(unlabeled_loader.dataset)} batch_size={unlabeled_loader.batch_size} dirs={unlabeled_dirs}"
        )

    class_weights = base.parse_class_weights(args.class_weights, device)
    criterion = base.CompositeSegmentationLoss(
        num_classes=len(base.CLASS_NAMES),
        class_weights=class_weights,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        aux_loss_weight=args.aux_loss_weight,
        label_smoothing=args.label_smoothing,
    )
    optimizer = base.build_optimizer(model, args)
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

    full_train_batches = max(int(np.ceil(len(train_records) / float(args.train_batch_size))), 1)
    adamix = AdaptiveMix2D(
        image_size=args.image_size,
        patch_divisor=args.adamix_patch_divisor,
        topk=args.adamix_topk,
        p=args.adamix_prob,
        total_steps=max(args.epoch * full_train_batches, 1),
        self_paced=not args.adamix_no_self_paced,
        device=device,
    )

    epoch_metrics_file = os.path.join(fold_dir, f"training_metrics_fold{fold_idx}.csv")
    per_class_file = os.path.join(fold_dir, f"per_class_metrics_fold{fold_idx}.csv")
    best_val_dice = -1.0
    best_model_path = os.path.join(fold_dir, "best_model.pth")
    last_model_path = os.path.join(fold_dir, "last.pth")
    checkpoint_path = default_resume_path(args)
    resume_meter_state = None
    resume_train_stats = None

    if resume_checkpoint:
        print(
            f"Resume fold {fold_idx} from epoch {resume_epoch}, "
            f"next_batch={resume_next_batch}, checkpoint={checkpoint_path}"
        )
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if scheduler is not None and resume_checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        if scaler is not None and resume_checkpoint.get("scaler") is not None:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        if teacher_model is not None and resume_checkpoint.get("teacher_model") is not None:
            teacher_model.load_state_dict(resume_checkpoint["teacher_model"])
        best_val_dice = float(resume_checkpoint.get("best_val_dice", best_val_dice))
        resume_meter_state = resume_checkpoint.get("meter")
        resume_train_stats = resume_checkpoint.get("train_stats")
        set_rng_state(resume_checkpoint.get("rng_state"))

    for epoch in range(resume_epoch, args.epoch + 1):
        start_time = time.time()
        lr = base.current_lr(optimizer)
        epoch_unsup_weight = base.rampup_weight(
            epoch,
            args.semi_start_epoch,
            args.unsup_rampup_epochs,
            args.unsup_weight if args.semi_supervised else 0.0,
        )
        start_batch = resume_next_batch if resume_checkpoint and epoch == resume_epoch else 0
        train_loader, train_count, train_offset = build_resumable_loader(
            train_records,
            args,
            pin_memory,
            shuffle_seed=args.seed + fold_idx * 100000 + epoch,
            start_batch=start_batch,
        )
        if train_offset >= train_count and resume_meter_state is not None:
            print(f"Fold [{fold_idx}] epoch {epoch} already finished in checkpoint; continuing with validation.")
            meter = base.SegmentationMetricsMeter(len(base.CLASS_NAMES), base.CLASS_NAMES)
            load_meter_state(meter, resume_meter_state)
            train_metrics = meter.compute()
            stats = resume_train_stats or {}
            unsup_batch_count = max(int(stats.get("unsup_batch_count", 0)), 1)
            train_metrics["unsup_loss"] = float(stats.get("unsup_loss_sum", 0.0)) / unsup_batch_count
            train_metrics["unsup_valid_ratio"] = float(stats.get("unsup_valid_ratio_sum", 0.0)) / unsup_batch_count
            train_metrics["unsup_confidence"] = float(stats.get("unsup_confidence_sum", 0.0)) / unsup_batch_count
            train_metrics["unsup_pseudo_foreground_ratio"] = float(stats.get("unsup_foreground_ratio_sum", 0.0)) / unsup_batch_count
            train_metrics["unsup_weight"] = float(epoch_unsup_weight)
        else:
            checkpoint_context = {
                "path": checkpoint_path,
                "args": args,
                "fold_idx": fold_idx,
                "scheduler": scheduler,
                "save_every": args.resume_save_every,
                "best_val_dice": best_val_dice,
                "completed_folds": completed_folds,
            }
            train_metrics = train_one_epoch_adamix(
                model,
                train_loader,
                unlabeled_loader,
                criterion,
                optimizer,
                device,
                adamix,
                scaler=scaler,
                grad_clip_norm=args.grad_clip_norm,
                amp_forward=args.amp_forward,
                unsup_weight=epoch_unsup_weight,
                teacher_model=teacher_model,
                ema_decay=args.ema_decay,
                class_weights=class_weights,
                fg_threshold=args.unsup_confidence_threshold,
                bg_threshold=args.unsup_bg_confidence_threshold,
                ignore_background=args.unsup_ignore_background,
                use_labeled_adamix=args.adamix_use_labeled,
                epoch=epoch,
                start_batch=start_batch,
                resume_meter_state=resume_meter_state if epoch == resume_epoch else None,
                resume_train_stats=resume_train_stats if epoch == resume_epoch else None,
                checkpoint_context=checkpoint_context,
            )
        resume_checkpoint = None
        resume_meter_state = None
        resume_train_stats = None
        resume_next_batch = 0
        eval_model = teacher_model if teacher_model is not None else model
        val_metrics = base.evaluate(eval_model, val_loader, criterion, device, desc=f"Validation fold {fold_idx}")
        elapsed = time.time() - start_time

        row = {"fold": fold_idx, "epoch": epoch, "lr": lr, "time_seconds": elapsed}
        row.update(base.flatten_metrics("train", train_metrics))
        row.update(base.flatten_metrics("val", val_metrics))
        base.append_dict_csv(epoch_metrics_file, row)
        base.append_per_class_csv(per_class_file, epoch, "train", train_metrics)
        base.append_per_class_csv(per_class_file, epoch, "val", val_metrics)
        base.print_epoch_summary(epoch, args.epoch, train_metrics, val_metrics, elapsed, lr, fold_idx=fold_idx)

        if val_metrics["mean_foreground_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_foreground_dice"]
            model_to_save = teacher_model if teacher_model is not None else model
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f"Fold [{fold_idx}] best AdaMix model saved at epoch {epoch} with foreground Dice: {best_val_dice:.4f}")

        model_to_save = teacher_model if teacher_model is not None else model
        torch.save(model_to_save.state_dict(), last_model_path)
        if scheduler is not None:
            scheduler.step()
        save_resume_checkpoint(
            checkpoint_path,
            args,
            fold_idx,
            epoch + 1,
            0,
            model,
            optimizer,
            scheduler,
            scaler,
            teacher_model,
            None,
            None,
            best_val_dice,
            completed_folds,
        )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = base.evaluate(model, test_loader, criterion, device, desc=f"Test fold {fold_idx}")

    test_row = {"fold": fold_idx, "best_val_mean_foreground_dice": best_val_dice}
    test_row.update(base.flatten_metrics("test", test_metrics))
    test_file = os.path.join(fold_dir, f"test_results_fold{fold_idx}.csv")
    base.append_dict_csv(test_file, test_row)
    base.append_per_class_csv(per_class_file, "best", "test", test_metrics)

    with open(os.path.join(fold_dir, f"test_results_fold{fold_idx}.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print(
        f"Fold [{fold_idx}] AdaMix Test Results - Loss: {test_metrics['loss']:.4f}, "
        f"Foreground Dice: {test_metrics['mean_foreground_dice']:.4f}, "
        f"Foreground IoU: {test_metrics['mean_foreground_iou']:.4f}, "
        f"Pixel Accuracy: {test_metrics['overall_accuracy']:.4f}"
    )
    return test_row, test_metrics


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = default_resume_path(args)
    resume_checkpoint = None
    completed_folds = []
    if not args.no_auto_resume and os.path.exists(checkpoint_path):
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        completed_folds = list(resume_checkpoint.get("completed_folds", []))
        print(
            f"Auto resume enabled: loaded checkpoint {checkpoint_path} "
            f"(fold={resume_checkpoint.get('fold_idx')}, epoch={resume_checkpoint.get('epoch')}, "
            f"next_batch={resume_checkpoint.get('next_batch')})"
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cv_records = base.collect_cv_samples(args)
    test_images_dir, test_masks_dir = base.split_name_to_dirs(args.dataset_dir, "test")
    test_records = base.HistologySegDataset.collect_samples(test_images_dir, test_masks_dir, split_name="test")
    folds = base.make_folds(len(cv_records), args.num_folds, args.seed)

    fold_rows = []
    fold_metrics = []
    for fold_idx, val_indices in enumerate(folds, start=1):
        if fold_idx in completed_folds:
            print(f"Skip completed fold [{fold_idx}/{args.num_folds}] from resume checkpoint.")
            row, metrics = load_existing_test_result(args.save_dir, fold_idx)
            if row is not None and metrics is not None:
                fold_rows.append(row)
                fold_metrics.append(metrics)
            continue
        completed, max_epoch = fold_reached_epoch(args.save_dir, fold_idx, args.epoch)
        if completed and not args.rerun_completed_folds:
            print(
                f"Fold [{fold_idx}/{args.num_folds}] already reached epoch {max_epoch} "
                f"(target={args.epoch}); skip training and continue."
            )
            fold_row, metrics = evaluate_completed_fold(args, fold_idx, test_records, device)
            fold_rows.append(fold_row)
            fold_metrics.append(metrics)
            continue

        val_index_set = set(int(idx) for idx in val_indices)
        train_indices = [idx for idx in range(len(cv_records)) if idx not in val_index_set]
        train_records = base.records_from_indices(cv_records, train_indices)
        val_records = base.records_from_indices(cv_records, val_indices)
        fold_resume_checkpoint = resume_checkpoint if resume_checkpoint and int(resume_checkpoint.get("fold_idx", -1)) == fold_idx else None

        print(
            f"Starting AdaMix fold [{fold_idx}/{args.num_folds}] "
            f"train={len(train_records)} val={len(val_records)} test={len(test_records)}"
        )
        fold_row, metrics = run_fold(
            args,
            fold_idx,
            train_records,
            val_records,
            test_records,
            device,
            resume_checkpoint=fold_resume_checkpoint,
            completed_folds=completed_folds,
        )
        fold_rows.append(fold_row)
        fold_metrics.append(metrics)
        if fold_idx not in completed_folds:
            completed_folds.append(fold_idx)
        resume_checkpoint = None
        save_light_resume_checkpoint(checkpoint_path, args, fold_idx + 1, completed_folds)

    summary, per_class_summary = base.write_cross_validation_summary(args.save_dir, fold_rows, fold_metrics)
    base.write_legacy_final_results(args.save_dir, args.seed, summary, per_class_summary)
    print(f"AdaMix logs saved to {args.save_dir}")


if __name__ == "__main__":
    main()
