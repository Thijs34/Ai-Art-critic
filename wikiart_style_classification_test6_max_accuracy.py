import argparse
import copy
import csv
import math
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

try:
    import timm
except ImportError as exc:
    raise ImportError("This script requires timm. Install with: pip install timm") from exc


@dataclass
class TrainConfig:
    model_name: str = "vit_base_patch16_384.augreg_in21k_ft_in1k"
    image_size: int = 384
    batch_size: int = 12
    effective_batch_size: int = 48
    head_epochs: int = 3
    ft_epochs: int = 28
    patience: int = 8
    warmup_epochs: int = 2
    head_lr: float = 6e-4
    ft_backbone_lr: float = 7e-6
    ft_head_lr: float = 3e-5
    weight_decay: float = 8e-5
    mixup_alpha: float = 0.5
    cutmix_alpha: float = 1.0
    mix_probability: float = 0.85
    label_smoothing: float = 0.1
    ema_decay: float = 0.9997
    grad_clip_norm: float = 1.0
    tta: bool = True
    use_weighted_sampler: bool = True
    seed: int = 42
    num_workers: int = 0
    fast_mode: bool = False


class WikiArtStyleDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, image_root: Path, transform=None):
        self.rows = rows.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        image_path = self.image_root / row["relative_path"]
        label = int(row["label"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9997):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if k in msd:
                model_v = msd[k].detach()
                if torch.is_floating_point(v):
                    v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
                else:
                    v.copy_(model_v)


class SoftTargetCrossEntropyWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(1)
        if targets.dim() == 1:
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.smoothing / (n_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            targets = true_dist
        log_probs = torch.log_softmax(logits, dim=1)
        return torch.mean(torch.sum(-targets * log_probs, dim=1))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_paths(project_root: Path) -> Tuple[Path, Path, Path]:
    wikiart_dir = project_root / "datasets" / "Wikiart"
    train_csv = wikiart_dir / "style_train.csv"
    val_csv = wikiart_dir / "style_val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Could not find style_train.csv and style_val.csv in datasets/Wikiart")
    return wikiart_dir, train_csv, val_csv


def load_style_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["relative_path", "label"])
    df["label"] = df["label"].astype(int)
    return df


def make_eval_split(val_df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped = val_df.groupby("label", group_keys=False)
    eval_test = grouped.apply(lambda x: x.sample(frac=0.5, random_state=seed) if len(x) > 1 else x)
    eval_val = val_df.drop(eval_test.index)
    eval_test = eval_test.reset_index(drop=True)
    eval_val = eval_val.reset_index(drop=True)
    if len(eval_val) == 0:
        eval_val = val_df.sample(frac=0.5, random_state=seed).reset_index(drop=True)
        eval_test = val_df.drop(eval_val.index).reset_index(drop=True)
    return eval_val, eval_test


def filter_existing_rows(df: pd.DataFrame, image_root: Path, split_name: str) -> pd.DataFrame:
    keep_mask = df["relative_path"].map(lambda p: (image_root / p).exists())
    missing_count = int((~keep_mask).sum())
    if missing_count > 0:
        print(f"[{split_name}] Filtered out {missing_count} missing files.")
    return df.loc[keep_mask].reset_index(drop=True)


def build_transforms(image_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=11),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.22, scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfms, eval_tfms


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sample_weights = np.sqrt(sample_weights)
    sample_weights = sample_weights / sample_weights.mean()
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def mixup_cutmix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha_mixup: float,
    alpha_cutmix: float,
    p: float,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if np.random.rand() > p:
        return inputs, torch.nn.functional.one_hot(targets, num_classes=num_classes).float()

    batch_size = inputs.size(0)
    indices = torch.randperm(batch_size, device=inputs.device)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    shuffled_targets = targets_onehot[indices]

    use_cutmix = alpha_cutmix > 0 and np.random.rand() < 0.5
    if use_cutmix:
        lam = np.random.beta(alpha_cutmix, alpha_cutmix)
        h, w = inputs.size(2), inputs.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        mixed = inputs.clone()
        mixed[:, :, y1:y2, x1:x2] = inputs[indices, :, y1:y2, x1:x2]
        lam_adjusted = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))
        soft_targets = targets_onehot * lam_adjusted + shuffled_targets * (1.0 - lam_adjusted)
        return mixed, soft_targets

    if alpha_mixup <= 0:
        return inputs, targets_onehot

    lam = np.random.beta(alpha_mixup, alpha_mixup)
    mixed = lam * inputs + (1.0 - lam) * inputs[indices]
    soft_targets = lam * targets_onehot + (1.0 - lam) * shuffled_targets
    return mixed, soft_targets


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> List[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        out = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            out.append(correct_k / targets.size(0))
        return out


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    num_classes: int,
    cfg: TrainConfig,
    ema: Optional[ModelEMA],
    is_train: bool,
    accum_steps: int,
) -> Dict[str, float]:
    model.train(is_train)

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    autocast_enabled = device.type == "cuda"

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            if is_train:
                images_mixed, soft_targets = mixup_cutmix(
                    images,
                    targets,
                    cfg.mixup_alpha,
                    cfg.cutmix_alpha,
                    cfg.mix_probability,
                    num_classes,
                )
                logits = model(images_mixed)
                loss = criterion(logits, soft_targets)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

            loss = loss / accum_steps

        if is_train:
            scaler.scale(loss).backward()
            if step % accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

        with torch.no_grad():
            top1, top5 = topk_accuracy(logits, targets, topk=(1, 5))

        bs = images.size(0)
        total_samples += bs
        total_loss += loss.item() * accum_steps * bs
        total_top1 += float(top1.item()) * bs
        total_top5 += float(top5.item()) * bs

    return {
        "loss": total_loss / max(1, total_samples),
        "top1": total_top1 / max(1, total_samples),
        "top5": total_top5 / max(1, total_samples),
    }


@torch.no_grad()
def evaluate_with_tta(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_tta: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        if use_tta:
            logits_flip = model(torch.flip(images, dims=[3]))
            logits = (logits + logits_flip) / 2.0

        loss = criterion(logits, targets)
        top1, top5 = topk_accuracy(logits, targets, topk=(1, 5))

        bs = images.size(0)
        total_samples += bs
        total_loss += float(loss.item()) * bs
        total_top1 += float(top1.item()) * bs
        total_top5 += float(top5.item()) * bs

    return {
        "loss": total_loss / max(1, total_samples),
        "top1": total_top1 / max(1, total_samples),
        "top5": total_top5 / max(1, total_samples),
    }


def freeze_backbone(model: nn.Module, freeze: bool):
    for p in model.parameters():
        p.requires_grad = not freeze

    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        for p in model.head.parameters():
            p.requires_grad = True

    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        for p in model.fc.parameters():
            p.requires_grad = True


def get_trainable_parameters(model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float):
    head_params = []
    backbone_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ["head", "fc", "classifier"]):
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return param_groups


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int):
    warmup_epochs = max(0, warmup_epochs)
    total_epochs = max(1, total_epochs)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.12, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)


def pick_num_workers(explicit_workers: int) -> int:
    if explicit_workers >= 0:
        return explicit_workers
    if os.name == "nt":
        return 0
    return min(8, max(2, (os.cpu_count() or 4) // 2))


def fit(project_root: Path, cfg: TrainConfig):
    set_seed(cfg.seed)
    Image.MAX_IMAGE_PIXELS = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wikiart_dir, train_csv, val_csv = discover_paths(project_root)

    train_df = load_style_csv(train_csv)
    val_df = load_style_csv(val_csv)

    train_df = filter_existing_rows(train_df, wikiart_dir, split_name="train")
    val_df = filter_existing_rows(val_df, wikiart_dir, split_name="val")

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError("No usable data after filtering missing files.")

    eval_val_df, eval_test_df = make_eval_split(val_df, seed=cfg.seed)

    num_classes = int(max(train_df["label"].max(), val_df["label"].max()) + 1)

    train_tfms, eval_tfms = build_transforms(cfg.image_size)

    train_ds = WikiArtStyleDataset(train_df, wikiart_dir, transform=train_tfms)
    val_ds = WikiArtStyleDataset(eval_val_df, wikiart_dir, transform=eval_tfms)
    test_ds = WikiArtStyleDataset(eval_test_df, wikiart_dir, transform=eval_tfms)

    num_workers = pick_num_workers(cfg.num_workers)
    pin_memory = device.type == "cuda"

    sampler = None
    shuffle = True
    if cfg.use_weighted_sampler:
        sampler = create_weighted_sampler(train_df["label"].values)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    if cfg.fast_mode:
        cfg.head_epochs = min(cfg.head_epochs, 1)
        cfg.ft_epochs = min(cfg.ft_epochs, 4)
        cfg.patience = min(cfg.patience, 2)

    print(f"Device: {device}")
    print(f"Model: {cfg.model_name}")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Classes: {num_classes}")

    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=num_classes)
    model.to(device)

    eval_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    train_criterion = SoftTargetCrossEntropyWithLabelSmoothing(smoothing=cfg.label_smoothing)

    grad_accum_steps = max(1, math.ceil(cfg.effective_batch_size / cfg.batch_size))

    freeze_backbone(model, freeze=True)
    head_groups = get_trainable_parameters(
        model,
        backbone_lr=cfg.head_lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer = torch.optim.AdamW(head_groups)
    scheduler = build_scheduler(optimizer, warmup_epochs=min(cfg.warmup_epochs, cfg.head_epochs), total_epochs=cfg.head_epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ema = ModelEMA(model, decay=cfg.ema_decay)

    history: List[Dict] = []
    best_val_top1 = -1.0
    best_epoch = -1
    patience_left = cfg.patience

    checkpoint_dir = project_root / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "wikiart_test6_best.pt"

    epoch_counter = 0

    for stage_name, stage_epochs, freeze_backbone_flag, lr_backbone, lr_head in [
        ("head-only", cfg.head_epochs, True, cfg.head_lr, cfg.head_lr),
        ("fine-tune", cfg.ft_epochs, False, cfg.ft_backbone_lr, cfg.ft_head_lr),
    ]:
        if stage_epochs <= 0:
            continue

        freeze_backbone(model, freeze=freeze_backbone_flag)
        param_groups = get_trainable_parameters(
            model,
            backbone_lr=lr_backbone,
            head_lr=lr_head,
            weight_decay=cfg.weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = build_scheduler(
            optimizer,
            warmup_epochs=min(cfg.warmup_epochs, stage_epochs),
            total_epochs=stage_epochs,
        )

        for _ in range(stage_epochs):
            epoch_counter += 1

            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=train_criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                num_classes=num_classes,
                cfg=cfg,
                ema=ema,
                is_train=True,
                accum_steps=grad_accum_steps,
            )

            eval_model = ema.ema_model if ema is not None else model
            val_metrics = evaluate_with_tta(
                model=eval_model,
                loader=val_loader,
                criterion=eval_criterion,
                device=device,
                use_tta=cfg.tta,
            )

            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch_counter,
                "stage": stage_name,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top5": train_metrics["top5"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
            }
            history.append(row)

            print(
                f"Epoch {epoch_counter:02d} | {stage_name:9s} | "
                f"lr={current_lr:.2e} | "
                f"train_loss={row['train_loss']:.4f}, train_acc={row['train_top1']:.4f} | "
                f"val_loss={row['val_loss']:.4f}, val_acc={row['val_top1']:.4f}, val_top5={row['val_top5']:.4f}"
            )

            improved = row["val_top1"] > best_val_top1
            if improved:
                best_val_top1 = row["val_top1"]
                best_epoch = epoch_counter
                patience_left = cfg.patience
                checkpoint = {
                    "model_state": copy.deepcopy(eval_model.state_dict()),
                    "config": asdict(cfg),
                    "best_val_top1": best_val_top1,
                    "best_epoch": best_epoch,
                    "num_classes": num_classes,
                    "model_name": cfg.model_name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                torch.save(checkpoint, checkpoint_path)
            else:
                patience_left -= 1

            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch_counter} (best epoch: {best_epoch}).")
                break

        if patience_left <= 0:
            break

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"])

    final_val_metrics = evaluate_with_tta(model, val_loader, eval_criterion, device=device, use_tta=cfg.tta)
    final_test_metrics = evaluate_with_tta(model, test_loader, eval_criterion, device=device, use_tta=cfg.tta)

    print("\nFinal evaluation")
    print(
        f"Validation -> loss: {final_val_metrics['loss']:.4f}, "
        f"top1: {final_val_metrics['top1']:.4f}, top5: {final_val_metrics['top5']:.4f}"
    )
    print(
        f"Test       -> loss: {final_test_metrics['loss']:.4f}, "
        f"top1: {final_test_metrics['top1']:.4f}, top5: {final_test_metrics['top5']:.4f}"
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val_top1: {best_val_top1:.4f}")

    history_df = pd.DataFrame(history)
    results_dir = project_root / "models" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    history_csv = results_dir / "wikiart_test6_history.csv"
    history_df.to_csv(history_csv, index=False)

    summary_row = {
        "notebook": "wikiart_style_classification_test6_max_accuracy.py",
        "exists": True,
        "model_name": cfg.model_name,
        "val_loss": final_val_metrics["loss"],
        "val_top1": final_val_metrics["top1"],
        "val_top5": final_val_metrics["top5"],
        "test_loss": final_test_metrics["loss"],
        "test_top1": final_test_metrics["top1"],
        "test_top5": final_test_metrics["top5"],
        "best_val_top1": best_val_top1,
        "best_epoch": best_epoch,
        "experiment": "test6",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_csv = results_dir / "wikiart_tests_1_to_6_summary.csv"
    prior_summary_csv = results_dir / "wikiart_tests_1_to_5_summary.csv"

    summary_rows = []
    if prior_summary_csv.exists():
        prior_df = pd.read_csv(prior_summary_csv)
        summary_rows.extend(prior_df.to_dict(orient="records"))

    summary_rows = [r for r in summary_rows if str(r.get("experiment", "")).lower() != "test6"]
    summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    if "experiment" in summary_df.columns:
        summary_df = summary_df.sort_values("experiment").reset_index(drop=True)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved history: {history_csv}")
    print(f"Saved summary: {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WikiArt style classification test6 (max accuracy).")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--model", type=str, default=TrainConfig.model_name)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--effective-batch-size", type=int, default=TrainConfig.effective_batch_size)
    parser.add_argument("--head-epochs", type=int, default=TrainConfig.head_epochs)
    parser.add_argument("--ft-epochs", type=int, default=TrainConfig.ft_epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--warmup-epochs", type=int, default=TrainConfig.warmup_epochs)
    parser.add_argument("--head-lr", type=float, default=TrainConfig.head_lr)
    parser.add_argument("--ft-backbone-lr", type=float, default=TrainConfig.ft_backbone_lr)
    parser.add_argument("--ft-head-lr", type=float, default=TrainConfig.ft_head_lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--mixup-alpha", type=float, default=TrainConfig.mixup_alpha)
    parser.add_argument("--cutmix-alpha", type=float, default=TrainConfig.cutmix_alpha)
    parser.add_argument("--mix-probability", type=float, default=TrainConfig.mix_probability)
    parser.add_argument("--label-smoothing", type=float, default=TrainConfig.label_smoothing)
    parser.add_argument("--ema-decay", type=float, default=TrainConfig.ema_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=TrainConfig.grad_clip_norm)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        head_epochs=args.head_epochs,
        ft_epochs=args.ft_epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        head_lr=args.head_lr,
        ft_backbone_lr=args.ft_backbone_lr,
        ft_head_lr=args.ft_head_lr,
        weight_decay=args.weight_decay,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_probability=args.mix_probability,
        label_smoothing=args.label_smoothing,
        ema_decay=args.ema_decay,
        grad_clip_norm=args.grad_clip_norm,
        tta=not args.no_tta,
        use_weighted_sampler=not args.no_weighted_sampler,
        seed=args.seed,
        num_workers=args.num_workers,
        fast_mode=args.fast,
    )

    project_root = Path(args.project_root).resolve()
    fit(project_root, cfg)


if __name__ == "__main__":
    main()
