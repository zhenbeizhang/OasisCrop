from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from oasiscrop.data import transforms as T
from oasiscrop.models import OasisCropNet


class TilePresetTrain:
    def __init__(self, size: int = 512, hflip_prob: float = 0.5) -> None:
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize([size, size], resize_mask=True),
                T.RandomScale(scale_range=(0.8, 1.2), prob=0.5),
                T.RandomRotation(45, prob=0.5),
                T.RandomHorizontalFlip(hflip_prob),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    prob=0.5,
                ),
                T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0), prob=0.3),
                T.Normalize(),
            ]
        )

    def __call__(self, image, target):
        return self.transforms(image, target)


class TilePresetEval:
    def __init__(self, size: int = 512) -> None:
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize([size, size], resize_mask=False),
                T.Normalize(),
            ]
        )

    def __call__(self, image, target):
        return self.transforms(image, target)


class PairedTileDataset(Dataset):
    """Paired GeoTIFF image-mask dataset with explicit image and mask folders."""

    def __init__(
        self,
        image_dir: str | os.PathLike,
        mask_dir: str | os.PathLike,
        transforms=None,
        band_combination: list[int] | None = None,
        use_ndvi: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
        self.band_combination = band_combination or [0, 1, 2]
        self.use_ndvi = use_ndvi

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        images = sorted(self.image_dir.glob("*.tif"))
        masks = {p.name: p for p in self.mask_dir.glob("*.tif")}
        pairs = [(p, masks[p.name]) for p in images if p.name in masks]
        if not pairs:
            raise RuntimeError(f"No paired GeoTIFF files found: {self.image_dir} / {self.mask_dir}")
        self.images_path, self.masks_path = zip(*pairs)

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, idx: int):
        image = self.read_image(self.images_path[idx])
        target = self.read_mask(self.masks_path[idx])
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def read_image(self, path: Path) -> np.ndarray:
        image = tifffile.imread(str(path))
        if image.ndim == 2:
            image = image[:, :, None]
        elif image.ndim == 3 and image.shape[0] <= 8 and image.shape[0] < image.shape[-1]:
            image = np.moveaxis(image, 0, -1)
        bands = [image[:, :, band_idx].astype(np.float32) for band_idx in range(image.shape[2])]
        bands = [np.nan_to_num(band, nan=0.0, posinf=1.0e4, neginf=-1.0e4) for band in bands]

        if self.use_ndvi and len(bands) >= 4:
            red = bands[0]
            nir = bands[3]
            bands.append((nir - red) / (nir + red + 1.0e-6))

        selected = [bands[i] for i in self.band_combination]
        image = np.stack(selected, axis=2)
        return np.clip(image, -1.0e4, 1.0e4)

    @staticmethod
    def read_mask(path: Path) -> np.ndarray:
        mask = tifffile.imread(str(path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (np.clip(mask, 0, 255) > 0).astype(np.float32)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        return cat_list(images, fill_value=0), cat_list(targets, fill_value=0)


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched


def make_boundary_target(mask: torch.Tensor, width: int = 5) -> torch.Tensor:
    if width < 3:
        width = 3
    if width % 2 == 0:
        width += 1
    pad = width // 2
    mask = (mask > 0.5).float()
    dilated = F.max_pool2d(mask, kernel_size=width, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=width, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def oasiscropnet_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    boundary_weight: float = 0.2,
    aux_weight: float = 0.1,
    dice_weight: float = 0.5,
    boundary_width: int = 5,
) -> tuple[torch.Tensor, dict[str, float]]:
    main = outputs["out"]
    aux = outputs["aux"]
    boundary = outputs["boundary"]
    boundary_target = make_boundary_target(target, width=boundary_width)

    main_bce = F.binary_cross_entropy_with_logits(main, target)
    main_dice = dice_loss_from_logits(main, target)
    aux_bce = F.binary_cross_entropy_with_logits(aux, target)
    boundary_bce = F.binary_cross_entropy_with_logits(boundary, boundary_target)
    loss = main_bce + dice_weight * main_dice + aux_weight * aux_bce + boundary_weight * boundary_bce
    return loss, {
        "loss": float(loss.detach().cpu()),
        "main_bce": float(main_bce.detach().cpu()),
        "main_dice": float(main_dice.detach().cpu()),
        "aux_bce": float(aux_bce.detach().cpu()),
        "boundary_bce": float(boundary_bce.detach().cpu()),
    }


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1.0e-4):
    groups = [{"params": [], "weight_decay": 0.0}, {"params": [], "weight_decay": weight_decay}]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            groups[0]["params"].append(param)
        else:
            groups[1]["params"].append(param)
    return groups


def create_lr_scheduler(
    optimizer,
    num_step: int,
    epochs: int,
    warmup: bool = True,
    warmup_epochs: int = 3,
    warmup_factor: float = 0.1,
    end_factor: float = 1.0e-4,
):
    if not warmup:
        warmup_epochs = 0

    def f(step):
        if warmup and step <= warmup_epochs * num_step:
            alpha = float(step) / float(max(1, warmup_epochs * num_step))
            return warmup_factor * (1.0 - alpha) + alpha
        current_step = step - warmup_epochs * num_step
        cosine_steps = max(1, (epochs - warmup_epochs) * num_step)
        return ((1.0 + math.cos(current_step * math.pi / cosine_steps)) / 2.0) * (1.0 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def compute_counts(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    pred = probs > threshold
    truth = target > 0.5
    return {
        "tp": int(torch.logical_and(pred, truth).sum().item()),
        "fp": int(torch.logical_and(pred, ~truth).sum().item()),
        "fn": int(torch.logical_and(~pred, truth).sum().item()),
        "tn": int(torch.logical_and(~pred, ~truth).sum().item()),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    eps = 1.0e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + fp + fn + tn + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    total = tp + fp + fn + tn
    po = oa
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / ((total * total) + eps)
    kappa = (po - pe) / (1.0 - pe + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "oa": oa,
        "iou": iou,
        "dice": dice,
        "kappa": kappa,
    }


@torch.no_grad()
def evaluate(model, data_loader, device, threshold: float = 0.5, max_batches: int | None = None):
    model.eval()
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for idx, (images, targets) in enumerate(data_loader, start=1):
        if max_batches is not None and idx > max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        batch_counts = compute_counts(logits, targets, threshold=threshold)
        for key in counts:
            counts[key] += batch_counts[key]
    return {**counts, **metrics_from_counts(counts)}


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch: int,
    lr_scheduler,
    scaler=None,
    print_freq: int = 20,
    max_steps: int | None = None,
    loss_kwargs: dict | None = None,
):
    model.train()
    loss_kwargs = loss_kwargs or {}
    running = {"loss": 0.0, "main_bce": 0.0, "main_dice": 0.0, "aux_bce": 0.0, "boundary_bce": 0.0}
    processed_steps = 0
    start = time.time()
    for step, (images, targets) in enumerate(data_loader, start=1):
        if max_steps is not None and step > max_steps:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None and device.type == "cuda"):
            outputs = model(images, return_aux=True)
            loss, parts = oasiscropnet_loss(outputs, targets, **loss_kwargs)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        lr_scheduler.step()

        for key in running:
            running[key] += parts[key]
        processed_steps += 1
        if step % print_freq == 0:
            elapsed = time.time() - start
            msg = " ".join(f"{key}={running[key] / step:.4f}" for key in running)
            print(f"Epoch {epoch:03d} step {step:05d}/{len(data_loader):05d} lr={optimizer.param_groups[0]['lr']:.8f} {msg} time={elapsed/60:.1f} min", flush=True)

    return {key: running[key] / max(1, processed_steps) for key in running} | {"lr": optimizer.param_groups[0]["lr"]}


def build_output_dir(args) -> Path:
    if args.resume:
        return Path(args.resume).resolve().parent
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    return Path(args.save_dir) / f"OasisCropNet_{stamp}"


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    train_dataset = PairedTileDataset(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        transforms=TilePresetTrain(args.tile_size),
        band_combination=args.band_combination,
        use_ndvi=args.use_ndvi,
    )
    test_dataset = PairedTileDataset(
        image_dir=args.test_image_dir,
        mask_dir=args.test_mask_dir,
        transforms=TilePresetEval(args.tile_size),
        band_combination=args.band_combination,
        use_ndvi=args.use_ndvi,
    )
    print(f"Train pairs: {len(train_dataset)}; test pairs: {len(test_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=train_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=test_dataset.collate_fn,
    )

    model = OasisCropNet(
        num_class=1,
        pretrained_backbone=args.pretrained_backbone,
        fpn_channels=args.fpn_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        get_params_groups(model, args.weight_decay),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    start_epoch = args.start_epoch
    best_iou = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", start_epoch)) + 1
        best_iou = float(checkpoint.get("best_iou", checkpoint.get("best_weighted_score", best_iou)))

    results_file = output_dir / "results.txt"
    start_time = time.time()
    loss_kwargs = {
        "boundary_weight": args.boundary_weight,
        "aux_weight": args.aux_weight,
        "dice_weight": args.dice_weight,
        "boundary_width": args.boundary_width,
    }

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            lr_scheduler,
            scaler=scaler,
            print_freq=args.print_freq,
            max_steps=args.max_train_steps,
            loss_kwargs=loss_kwargs,
        )

        should_eval = epoch % args.eval_interval == 0 or epoch == args.epochs - 1 or args.max_train_steps is not None
        if should_eval:
            metrics = evaluate(
                model,
                test_loader,
                device,
                threshold=args.threshold,
                max_batches=args.max_eval_batches,
            )
            line = (
                f"[epoch: {epoch}] "
                f"train_loss: {train_stats['loss']:.4f} "
                f"main_bce: {train_stats['main_bce']:.4f} "
                f"main_dice: {train_stats['main_dice']:.4f} "
                f"aux_bce: {train_stats['aux_bce']:.4f} "
                f"boundary_bce: {train_stats['boundary_bce']:.4f} "
                f"lr: {train_stats['lr']:.8f} "
                f"Precision: {metrics['precision']:.3f} "
                f"Recall: {metrics['recall']:.3f} "
                f"F1: {metrics['f1']:.3f} "
                f"OA: {metrics['oa']:.3f} "
                f"IoU: {metrics['iou']:.3f} "
                f"Dice: {metrics['dice']:.3f} "
                f"Kappa: {metrics['kappa']:.3f} "
                f"TP: {metrics['tp']} FP: {metrics['fp']} FN: {metrics['fn']} TN: {metrics['tn']}"
            )
            print(line, flush=True)
            with results_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                save_obj = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_iou": best_iou,
                    "metrics": metrics,
                    "args": vars(args),
                }
                if scaler is not None:
                    save_obj["scaler"] = scaler.state_dict()
                torch.save(save_obj, output_dir / "model_best.pth")
                print(f"Best model saved: IoU={best_iou:.4f}", flush=True)

        save_obj = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
            "args": vars(args),
        }
        if scaler is not None:
            save_obj["scaler"] = scaler.state_dict()
        latest = output_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(save_obj, latest)
        old = output_dir / f"checkpoint_epoch_{epoch - 2}.pth"
        if old.exists():
            old.unlink()

        if args.max_train_steps is not None:
            print("Stopping after limited train steps for sanity/debug run.", flush=True)
            break

    elapsed = str(dt.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}; output_dir={output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train OasisCropNet for binary visible net annual-crop segmentation.")
    parser.add_argument("--train-image-dir", required=True, help="Directory containing training image GeoTIFF tiles.")
    parser.add_argument("--train-mask-dir", required=True, help="Directory containing training binary mask GeoTIFF tiles.")
    parser.add_argument("--test-image-dir", required=True, help="Directory containing held-out image GeoTIFF tiles.")
    parser.add_argument("--test-mask-dir", required=True, help="Directory containing held-out binary mask GeoTIFF tiles.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--band-combination", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--use-ndvi", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--boundary-width", type=int, default=5)
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--save-dir", default="outputs/checkpoints")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

