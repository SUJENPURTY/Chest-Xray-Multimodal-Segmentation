"""Canonical training and held-out evaluation loops."""
from __future__ import annotations

import csv
import hashlib
import json
import gc
import os
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import Config
from src.data.dataset import QaTaDataset, group_key, read_manifest
from src.inference.predictor import _model_sources, load_checkpoint
from src.models.languide_medseg import create_model
from src.training.losses import DiceBCELoss
from src.training.metrics import binary_metrics, overlap_counts


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(cfg, split, tokenizer):
    csv_path = cfg.test_csv if split == "test" else cfg.train_csv
    data = QaTaDataset(csv_path, cfg.data_root, split, tokenizer,
                       cfg.image_size, augment=split == "train",
                       max_text_length=cfg.max_text_length, seed=cfg.seed)
    return DataLoader(data, batch_size=cfg.batch_size, shuffle=split == "train",
                      num_workers=cfg.num_workers,
                      pin_memory=torch.cuda.is_available() and os.name != "nt")


def _check_batch(image, target, probability=None):
    if image.ndim != 4 or image.shape[1] != 1 or target.ndim != 4 or target.shape[1] != 1 or image.shape != target.shape:
        raise ValueError("Image and mask must be matching [B,1,H,W] tensors")
    if not target.is_floating_point():
        raise TypeError("Mask target must be floating point")
    if not torch.isfinite(image).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite image or mask")
    if torch.any((target < 0) | (target > 1)):
        raise ValueError("Mask target must lie in [0,1]")
    if probability is not None:
        if probability.shape != target.shape:
            raise ValueError("Model probability and mask shapes differ")
        if not torch.isfinite(probability).all():
            raise FloatingPointError("Non-finite model probability")


def _run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None,
               threshold=0.5, mixed_precision=True):
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}
    count = 0
    for (image, text), target in loader:
        image, target = image.to(device), target.to(device)
        _check_batch(image, target)
        text = {key: value.to(device) for key, value in text.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda" and mixed_precision):
                probability = model([image, text])
            _check_batch(image, target, probability)
            # The released model already applies sigmoid. BCE on probabilities
            # is unsafe under CUDA autocast, so compute the loss in float32.
            with torch.autocast(device_type="cuda", enabled=False):
                loss = loss_fn(probability.float(), target.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite segmentation loss")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        metrics = binary_metrics(probability.detach(), target, threshold)
        batch_size = image.shape[0]
        for key, value in {"loss": loss.item(), **metrics}.items():
            sums[key] += value * batch_size
        count += batch_size
    return {key: value / count for key, value in sums.items()}


def _save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_resume_payload(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint missing: {path}")
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Cannot read resume checkpoint: {path}") from exc
    required = {"kind", "model", "optimizer", "scheduler", "epoch", "best_val_dice", "history", "rng_state", "config"}
    if not isinstance(saved, dict) or not required.issubset(saved) or saved["kind"] != "CLOUD_LANGUIDE_BINARY_FIXED_V1":
        raise ValueError("Unsupported or incomplete resume checkpoint")
    if (not isinstance(saved["epoch"], int) or saved["epoch"] < 1
            or not isinstance(saved["model"], dict) or not isinstance(saved["config"], dict)
            or not isinstance(saved["history"], list)
            or not isinstance(saved["best_val_dice"], (int, float))
            or not np.isfinite(saved["best_val_dice"])):
        raise ValueError("Invalid resume checkpoint metadata")
    if any(not isinstance(key, str) or not key.startswith("model.")
           for key in saved["model"]):
        raise ValueError("Invalid resume checkpoint model keys")
    return saved


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def train(cfg: Config, resume: Path | None = None):
    seed_all(cfg.seed)
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    manifest_hash = hashlib.sha256(cfg.train_csv.read_bytes()).hexdigest()
    checkpoint_dir = cfg.output_dir / "checkpoints"
    if resume is None and any((checkpoint_dir / name).exists() for name in
                              ("latest.ckpt", "best_dice.ckpt")):
        raise FileExistsError("Existing checkpoints found; choose a new OUTPUT_DIR or --resume")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}; mixed precision: {cfg.mixed_precision and device.type == 'cuda'}", flush=True)
    vision, text_source = _model_sources(cfg)
    tokenizer = AutoTokenizer.from_pretrained(text_source, trust_remote_code=True)
    train_loader = make_loader(cfg, "train", tokenizer)
    valid_loader = make_loader(cfg, "validation", tokenizer)
    test_names = {group_key(row["name"]) for row in read_manifest(
        cfg.test_csv, cfg.data_root, "test", cfg.seed)}
    train_names = {group_key(row["name"]) for row in train_loader.dataset.records}
    valid_names = {group_key(row["name"]) for row in valid_loader.dataset.records}
    if train_names & valid_names or (train_names | valid_names) & test_names:
        raise ValueError("Patient group overlap between train, validation or test")
    model = create_model(vision, text_source).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                   weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.mixed_precision)
    loss_fn = DiceBCELoss()
    best, start, history = -1.0, 1, []
    if resume:
        saved = _load_resume_payload(resume)
        previous = saved.get("config", {})
        if previous.get("seed") != cfg.seed or previous.get("epochs") != cfg.epochs:
            raise ValueError("Resume requires a checkpoint from this trainer with the same seed and total epochs")
        if previous.get("split_policy") != "patient_group_v1":
            raise ValueError("Resume split policy differs from the canonical patient-grouped split")
        if previous.get("train_csv_sha256") and previous["train_csv_sha256"] != manifest_hash:
            raise ValueError("Resume train manifest differs from checkpoint")
        if "batch_size" in previous and previous["batch_size"] != cfg.batch_size:
            raise ValueError("Resume batch size differs from checkpoint")
        if previous.get("mixed_precision") != cfg.mixed_precision:
            raise ValueError("Resume mixed-precision setting differs from checkpoint")
        model.load_state_dict({key.removeprefix("model."): value for key, value in saved["model"].items()})
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if "scaler" in saved:
            scaler.load_state_dict(saved["scaler"])
        best, start, history = saved["best_val_dice"], saved["epoch"] + 1, saved["history"]
        if len(history) != saved["epoch"] or "rng_state" not in saved:
            raise ValueError("Resume checkpoint has inconsistent history or no RNG state")
        _restore_rng(saved["rng_state"])
        source_best = Path(resume).parent / "best_dice.ckpt"
        destination_best = checkpoint_dir / "best_dice.ckpt"
        if not destination_best.is_file():
            if not source_best.is_file():
                raise FileNotFoundError(f"Best checkpoint missing beside resume file: {source_best}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if source_best.resolve() != destination_best.resolve():
                temporary_best = destination_best.with_suffix(".ckpt.tmp")
                try:
                    shutil.copy2(source_best, temporary_best)
                    temporary_best.replace(destination_best)
                finally:
                    temporary_best.unlink(missing_ok=True)
        # Loading optimizer/model state leaves a large CPU checkpoint payload.
        # Release it once all resume state has been restored on the device.
        del saved
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for directory in (cfg.output_dir / "metrics", cfg.output_dir / "figures",
                      cfg.output_dir / "predictions"):
        directory.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, cfg.epochs + 1):
        started = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        training = _run_epoch(model, train_loader, loss_fn, device, optimizer, scaler,
                              cfg.threshold, cfg.mixed_precision)
        with torch.inference_mode():
            validation = _run_epoch(model, valid_loader, loss_fn, device,
                                    threshold=cfg.threshold, mixed_precision=cfg.mixed_precision)
        scheduler.step()
        row = {"epoch": epoch, "learning_rate": lr, "seconds": time.monotonic() - started,
               **{f"train_{k}": v for k, v in training.items()},
               **{f"val_{k}": v for k, v in validation.items()}}
        history.append(row)
        improved = validation["dice"] > best
        if improved:
            best = validation["dice"]
        payload = {"kind": "CLOUD_LANGUIDE_BINARY_FIXED_V1",
                   "model": {"model." + key: value for key, value in model.state_dict().items()},
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "scaler": scaler.state_dict(), "epoch": epoch,
                   "best_epoch": epoch if improved else max(history, key=lambda r: r["val_dice"])["epoch"],
                   "best_val_dice": best, "history": history,
                   "rng_state": _rng_state(),
                   "config": {**{key: str(value) if isinstance(value, Path) else value
                                 for key, value in asdict(cfg).items()},
                              "train_csv_sha256": manifest_hash}}
        payload["config"]["split_policy"] = "patient_group_v1"
        _save(payload, checkpoint_dir / "latest.ckpt")
        if improved:
            _save(payload, checkpoint_dir / "best_dice.ckpt")
        with (cfg.output_dir / "metrics" / "train_history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}: train loss {training['loss']:.4f}, val loss {validation['loss']:.4f}, "
              f"val Dice {validation['dice']:.4f}, IoU {validation['iou']:.4f}, LR {lr:.2g}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return history


def evaluate(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision, text_source = _model_sources(cfg)
    tokenizer = AutoTokenizer.from_pretrained(text_source, trust_remote_code=True)
    loader = make_loader(cfg, "test", tokenizer)
    model, saved = load_checkpoint(create_model(vision, text_source), cfg.checkpoint_path, device)
    loss_fn = DiceBCELoss()
    rows = []
    total_tp = total_fp = total_fn = total_tn = 0
    with torch.inference_mode():
        for index, ((image, text), target) in enumerate(loader):
            image, target = image.to(device), target.to(device)
            _check_batch(image, target)
            text = {key: value.to(device) for key, value in text.items()}
            probability = model([image, text])
            _check_batch(image, target, probability)
            for offset in range(image.shape[0]):
                prediction = probability[offset:offset+1]
                truth = target[offset:offset+1]
                score = binary_metrics(prediction, truth, cfg.threshold)
                tp, fp, fn, tn = overlap_counts(prediction, truth, cfg.threshold)
                counts = [int(item.item()) for item in (tp, fp, fn, tn)]
                total_tp += counts[0]; total_fp += counts[1]
                total_fn += counts[2]; total_tn += counts[3]
                score["loss"] = loss_fn(prediction, truth).item()
                score["image"] = loader.dataset.records[index * cfg.batch_size + offset]["name"]
                score["image_name"] = loader.dataset.records[index * cfg.batch_size + offset]["image"].name
                rows.append(score)
    if len(rows) != len(loader.dataset):
        raise ValueError("Evaluation did not produce exactly one row per test image")
    aggregate = {f"mean_per_image_{key}": float(np.mean([row[key] for row in rows]))
                 for key in ("dice", "iou", "pixel_accuracy", "loss")}
    aggregate.update({
        "global_dice": (2 * total_tp / (2 * total_tp + total_fp + total_fn))
        if 2 * total_tp + total_fp + total_fn else 1.0,
        "global_iou": (total_tp / (total_tp + total_fp + total_fn))
        if total_tp + total_fp + total_fn else 1.0,
        "global_pixel_accuracy": (total_tp + total_tn) /
        max(total_tp + total_fp + total_fn + total_tn, 1),
    })
    aggregate["test_images"] = len(rows)
    aggregate["checkpoint_epoch"] = saved["epoch"]
    metrics_dir = cfg.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "test_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    with (metrics_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from src.training.evaluation_export import export_evaluation_excel
    export_evaluation_excel(aggregate, rows, metrics_dir / "dice_jaccard_scores.xlsx")
    return aggregate, rows
