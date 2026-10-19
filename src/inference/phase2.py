"""Read-only Phase-2 checkpoint inference and streaming binary evaluation.

Uses the existing architecture, tokenizer, MONAI transforms and display renderer.
No training, optimizer restoration, experiment imports or checkpoint writes occur.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.config import Config
from src.data.dataset import QaTaDataset, image_transform
from src.inference.predictor import _model_sources, render_prediction
from src.models.languide_medseg import create_model

METRICS = ("pixel_accuracy", "dice", "iou", "precision", "recall")
COUNTS = ("tp", "fp", "fn", "tn")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def memory_snapshot(stage):
    """Process RSS and host availability are distinct from CUDA tensor allocation."""
    snapshot = {
        "stage": stage,
        "process_rss_mib": psutil.Process().memory_info().rss / 1024**2,
        "available_ram_mib": psutil.virtual_memory().available / 1024**2,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        snapshot.update(
            gpu_free_mib=free / 1024**2,
            gpu_total_mib=total / 1024**2,
            allocated_mib=torch.cuda.memory_allocated() / 1024**2,
            reserved_mib=torch.cuda.memory_reserved() / 1024**2,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        )
    return snapshot


def metrics_from_counts(tp, fp, fn, tn):
    counts = (tp, fp, fn, tn)
    if any(not np.isfinite(v) or v < 0 or int(v) != v for v in counts):
        raise ValueError("Confusion counts must be finite nonnegative integers")
    tp, fp, fn, tn = map(int, counts)
    total = tp + fp + fn + tn
    if total == 0:
        raise ValueError("Metrics require at least one evaluated pixel")
    # Preserve the pilot's unsmoothed conventions. Undefined precision/recall
    # are explicitly zero; Dice and IoU of two empty masks are one.
    return {
        "pixel_accuracy": (tp + tn) / total,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def confusion_counts(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.dtype != np.bool_ or target.dtype != np.bool_:
        raise TypeError("Confusion inputs must be boolean masks")
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Masks must have identical nonempty shapes")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction)) - tp
    fn = int(np.count_nonzero(target)) - tp
    return dict(tp=tp, fp=fp, fn=fn, tn=int(target.size) - tp - fp - fn)


def summarize_rows(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    totals = {key: sum(int(row[key]) for row in rows) for key in COUNTS}
    # Derive macro values from integer counts, never from stored metric strings.
    per_image = [metrics_from_counts(**{k: row[k] for k in COUNTS}) for row in rows]
    return {
        "images": len(rows), "counts": totals,
        "image_mean": {key: float(np.mean([row[key] for row in per_image])) for key in METRICS},
        "dataset_global": metrics_from_counts(**totals),
    }


def read_verified_evaluation(directory, provenance, dataset):
    """Reuse this notebook's fresh run only if weights, code and every input match."""
    import csv

    directory = Path(directory)
    saved = json.loads((directory / "validation_summary.json").read_text(encoding="utf-8"))
    csv_path = directory / "validation_per_image.csv"
    if saved["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
        raise ValueError("Cached evaluation uses a different checkpoint; set PHASE2_FULL_VALIDATION=1")
    if saved["implementation_sha256"] != file_sha256(__file__):
        raise ValueError("Evaluation implementation changed; set PHASE2_FULL_VALIDATION=1")
    if saved["per_image_csv_sha256"] != file_sha256(csv_path) or saved["threshold"] != 0.5:
        raise ValueError("Cached evaluation provenance mismatch")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(dataset):
        raise ValueError("Cached evaluation population changed")
    for index, (row, record) in enumerate(zip(rows, dataset.records)):
        if int(row["validation_index"]) != index or row["image"] != record["name"]:
            raise ValueError("Cached sample order changed")
        for field in ("image", "mask"):
            if row[field + "_sha256"] != file_sha256(record[field]):
                raise ValueError(f"Cached input changed: {record[field]}")
        for key in COUNTS:
            row[key] = int(row[key])
        row.update(metrics_from_counts(**{key: row[key] for key in COUNTS}))
    recomputed = summarize_rows(rows)
    for group in ("image_mean", "dataset_global"):
        for key in METRICS:
            if not np.isclose(saved[group][key], recomputed[group][key], rtol=0, atol=1e-12):
                raise ValueError("Cached aggregate disagrees with recomputed counts")
    return rows, saved


def phase2_provenance(root, data_root=None):
    """Resolve only the pilot-selected checkpoint and verify its artifact hashes."""
    root = Path(root).resolve()
    pilot = root / "experiments/phase2_pilot"
    best = json.loads((pilot / "checkpoints/best.json").read_text(encoding="utf-8"))
    checkpoint = (pilot / best["path"].replace("\\", "/")).resolve()
    if not checkpoint.is_relative_to(pilot):
        raise ValueError("Checkpoint pointer leaves the Phase-2 experiment")
    reference = json.loads((pilot / best["validation_summary"]).read_text(encoding="utf-8"))
    training = json.loads((pilot / "config.json").read_text(encoding="utf-8"))
    cfg = replace(
        Config(), data_root=Path(data_root) if data_root else root / "_qata_compatible",
        checkpoint_path=checkpoint, seed=training["split_seed"],
        image_size=training["image_size"], max_text_length=training["max_text_length"],
        threshold=reference["threshold"], mixed_precision=False, num_workers=0, batch_size=1,
    )
    actual_sha = file_sha256(checkpoint)
    if actual_sha != best["sha256"] or actual_sha != reference["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint SHA256 mismatch")
    if file_sha256(cfg.train_csv) != training["train_csv_sha256"]:
        raise ValueError("Training annotation manifest changed since Phase 2")
    reference_csv = pilot / reference["per_image_csv"]
    if file_sha256(reference_csv) != reference["per_image_csv_sha256"]:
        raise ValueError("Historical per-image results SHA256 mismatch")
    if best["epoch"] != 2 or reference["threshold"] != 0.5:
        raise ValueError("Unexpected Phase-2 selection protocol")
    return cfg, {
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": actual_sha,
        "checkpoint_bytes": checkpoint.stat().st_size, "epoch": best["epoch"],
        "reference_csv": str(reference_csv), "reference_summary": reference,
        "training_config": training, "pilot_directory": str(pilot),
    }


def load_phase2_model(cfg, device):
    """Construct on the destination device; copy only model weights from mmap.

    No assign=True: model storage must never alias a writable checkpoint mapping.
    The checkpoint is a trusted local pickle, already SHA256-checked by the caller.
    """
    device = torch.device(device)
    vision_source, text_source = _model_sources(cfg)

    def configuration_only(source, **kwargs):
        config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
        config.output_hidden_states = kwargs.get("output_hidden_states", True)
        return AutoModel.from_config(config, trust_remote_code=True)

    # Reuse the existing constructors without loading redundant backbone weights.
    with patch.object(AutoModel, "from_pretrained", new=staticmethod(configuration_only)):
        with torch.device(device):
            model = create_model(vision_source, text_source).float()
    model.to(device)
    payload = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    try:
        if payload.get("kind") != "PHASE2_FROZEN_BN_PILOT_V1":
            raise ValueError("Expected Phase-2 frozen-BatchNorm pilot checkpoint")
        model.load_state_dict(payload["model"], strict=True)
        if payload["config"]["image_size"] != cfg.image_size or payload["config"]["max_text_length"] != cfg.max_text_length:
            raise ValueError("Checkpoint preprocessing differs from inference configuration")
    finally:
        del payload
        gc.collect()
    model.eval()
    return model


class Phase2Inference:
    """One resident model/tokenizer; outputs are small independent CPU arrays."""

    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.model = load_phase2_model(cfg, self.device)
        _, text_source = _model_sources(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_source, trust_remote_code=True, local_files_only=True)
        self.transform = image_transform(cfg.image_size)
        self.inference_calls = 0

    def validation_dataset(self, provenance):
        dataset = QaTaDataset(
            self.cfg.train_csv, self.cfg.data_root, "validation", self.tokenizer,
            image_size=self.cfg.image_size, augment=False,
            max_text_length=self.cfg.max_text_length, seed=self.cfg.seed)
        original = json.loads((Path(provenance["pilot_directory"]) / "data_manifest.json").read_text(encoding="utf-8"))["validation"]
        if len(dataset) != provenance["reference_summary"]["images"] or len(original) != len(dataset):
            raise ValueError("Validation population size changed")
        for expected, record in zip(original, dataset.records):
            if (expected["name"], expected["prompt"]) != (record["name"], record["prompt"]):
                raise ValueError("Validation sample order or prompts changed")
        return dataset

    def tokenize(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty text prompt is required")
        tokens = self.tokenizer(
            prompt.strip(), padding="max_length", truncation=True,
            max_length=self.cfg.max_text_length, return_attention_mask=True, return_tensors="pt")
        return {key: tokens[key][0] for key in ("input_ids", "attention_mask")}

    @torch.inference_mode()
    def infer_tensor(self, image, text):
        if hasattr(image, "as_tensor"):
            image = image.as_tensor()
        if tuple(image.shape) != (1, self.cfg.image_size, self.cfg.image_size) or not torch.isfinite(image).all():
            raise ValueError("Expected finite preprocessed image of shape (1,224,224)")
        image_batch = image.unsqueeze(0).to(self.device, dtype=torch.float32)
        text_batch = {key: text[key].unsqueeze(0).to(self.device) for key in ("input_ids", "attention_mask")}
        # The released forward already returns sigmoid probabilities. Never apply
        # sigmoid again, change precision, rotate inputs, or change the architecture.
        probability = self.model([image_batch, text_batch])
        self.inference_calls += 1
        expected = (1, 1, self.cfg.image_size, self.cfg.image_size)
        if tuple(probability.shape) != expected:
            raise ValueError(f"Unexpected model output shape: {tuple(probability.shape)}")
        result = probability[0, 0].float().cpu().numpy().copy()
        if not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
            raise FloatingPointError("Invalid sigmoid probability map")
        return result

    def predict_sample(self, dataset, index, prompt=None):
        (image, text), target = dataset[index]
        record = dataset.records[index]
        prompt = record["prompt"] if prompt is None else prompt
        if prompt != record["prompt"]:
            text = self.tokenize(prompt)
        probability = self.infer_tensor(image, text)
        truth = np.asarray(target[0].cpu()) > 0.5
        counts = confusion_counts(probability >= self.cfg.threshold, truth)
        with Image.open(record["image"]) as source:
            original = source.convert("RGB").copy()
        with Image.open(record["mask"]) as source:
            ground_truth = np.asarray(source.convert("L")) > 0
        visuals = render_prediction(original, probability, self.cfg.threshold)
        if ground_truth.shape != np.asarray(visuals["binary_mask"]).shape:
            raise ValueError("Original image and ground-truth dimensions differ")
        return {
            "index": index, "name": record["name"], "prompt": prompt,
            "original_prompt": record["prompt"], "original": original,
            "ground_truth": ground_truth, "probability_224": probability,
            "model_counts": counts, "model_metrics": metrics_from_counts(**counts),
            **visuals,
        }

    def predict_file(self, image_path, prompt, mask_path=None):
        """Single user-supplied file with optional aligned original-size truth."""
        image_path = Path(image_path)
        image = self.transform({"image": str(image_path)})["image"].float()
        probability = self.infer_tensor(image, self.tokenize(prompt))
        with Image.open(image_path) as source:
            original = source.convert("RGB").copy()
        result = {"name": image_path.name, "prompt": prompt, "original": original,
                  "probability_224": probability,
                  **render_prediction(original, probability, self.cfg.threshold)}
        if mask_path is not None:
            with Image.open(mask_path) as source:
                truth = np.asarray(source.convert("L")) > 0
            counts = confusion_counts(np.asarray(result["binary_mask"]) > 0, truth)
            result.update(ground_truth=truth, original_resolution_metrics=metrics_from_counts(**counts))
        return result

    def evaluate(self, dataset, progress=None):
        """One forward/image; retain scalar counts, never the full mask dataset."""
        rows, snapshots = [], [memory_snapshot("validation start")]
        started = time.perf_counter()
        for index in range(len(dataset)):
            (image, text), target = dataset[index]
            probability = self.infer_tensor(image, text)
            truth = np.asarray(target[0].cpu()) > 0.5
            counts = confusion_counts(probability >= self.cfg.threshold, truth)
            record = dataset.records[index]
            rows.append({
                "validation_index": index, "image": record["name"], **counts,
                **metrics_from_counts(**counts),
                "image_sha256": file_sha256(record["image"]),
                "mask_sha256": file_sha256(record["mask"]),
            })
            del image, text, target, probability, truth
            if (index + 1) % 200 == 0 or index + 1 == len(dataset):
                snapshots.append(memory_snapshot(f"validation {index + 1}"))
                if progress:
                    progress(index + 1, len(dataset))
        return rows, {**summarize_rows(rows), "seconds": time.perf_counter() - started,
                      "threshold": self.cfg.threshold, "precision": "FP32",
                      "resolution": [self.cfg.image_size, self.cfg.image_size],
                      "resource_snapshots": snapshots}
