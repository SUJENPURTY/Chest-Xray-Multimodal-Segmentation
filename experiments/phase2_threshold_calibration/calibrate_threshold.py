
from pathlib import Path
import sys
import json
import hashlib
import csv
import ctypes
import gc
import os
import subprocess
import threading
import time
from datetime import datetime, timezone

PROCESS_STARTED = time.perf_counter()

# Confine runtime writes to this experiment; existing model caches are read offline.
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key, value in {
    "PYTHONDONTWRITEBYTECODE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_MODULES_CACHE": str(OUTPUT_DIR / "cache" / "hf_modules"),
    "MPLCONFIGDIR": str(OUTPUT_DIR / "cache" / "matplotlib"),
    "TEMP": str(OUTPUT_DIR / "tmp"),
    "TMP": str(OUTPUT_DIR / "tmp"),
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}.items():
    os.environ[key] = value
for directory in (OUTPUT_DIR / "cache", OUTPUT_DIR / "tmp"):
    directory.mkdir(exist_ok=True)

import numpy as np
import torch

# Project paths
ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "experiments" / "phase2_pilot"
OUTPUT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PILOT_DIR))

# Import existing project components
from src.config import Config
from src.data.dataset import QaTaDataset
from src.inference.predictor import _model_sources
from src.models.languide_medseg import create_model
from transformers import AutoTokenizer

# Thresholds for validation-only calibration
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.025), 3)

# Read best checkpoint metadata
with open(PILOT_DIR / "checkpoints" / "best.json", "r") as f:
    best = json.load(f)

CHECKPOINT = PILOT_DIR / best["path"]

print("Project root:", ROOT)
print("Checkpoint:", CHECKPOINT.name)
print("Checkpoint exists:", CHECKPOINT.is_file())
print("Best epoch:", best["epoch"])
print("Original validation Dice:", best["validation_mean_dice"])
print("Original validation IoU:", best["validation_mean_iou"])
print("Number of thresholds:", len(THRESHOLDS))
print("CUDA available:", torch.cuda.is_available())


def load_best_model():
    import gc
    from unittest.mock import patch
    from transformers import AutoConfig, AutoModel

    config = Config(data_root=ROOT / "_qata_compatible")
    vision_source, text_source = _model_sources(config)

    def config_only(source, **kwargs):
        cfg = AutoConfig.from_pretrained(
            source,
            trust_remote_code=True,
            local_files_only=True,
        )
        if "output_hidden_states" in kwargs:
            cfg.output_hidden_states = kwargs["output_hidden_states"]
        return AutoModel.from_config(
            cfg,
            trust_remote_code=True,
        )

    with patch.object(AutoModel, "from_pretrained", new=config_only):
        with torch.device("cuda"):
            model = create_model(
                vision_source,
                text_source,
            ).float()

    model = model.cuda()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    assert "model" in checkpoint, "Model weights missing"

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    del checkpoint
    gc.collect()

    model.eval()

    print("Model loaded successfully!")
    print("Checkpoint:", CHECKPOINT.name)
    print("Evaluation mode:", not model.training)
    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**2, 2),
        "MiB",
    )

    return model


def load_validation_dataset():
    config = Config(data_root=ROOT / "_qata_compatible")
    _, text_source = _model_sources(config)

    tokenizer = AutoTokenizer.from_pretrained(
        text_source,
        trust_remote_code=True,
        local_files_only=True,
    )

    dataset = QaTaDataset(
        config.train_csv,
        config.data_root,
        "validation",
        tokenizer,
        augment=False,
        seed=2026,
    )

    assert len(dataset) == 1423, (
        f"Expected 1423 validation images, got {len(dataset)}"
    )

    print("Validation dataset loaded successfully!")
    print("Total validation images:", len(dataset))

    return dataset


def verify_single_inference(model, dataset):
    model.eval()

    (image, text), target = dataset[0]

    image = image.unsqueeze(0).cuda()
    text = {
        key: value.unsqueeze(0).cuda()
        for key, value in text.items()
    }
    target = target.unsqueeze(0).cuda()

    with torch.inference_mode():
        probability = model([image, text])

    print("Input shape:", tuple(image.shape))
    print("Prediction shape:", tuple(probability.shape))
    print("Target shape:", tuple(target.shape))
    print("Probability minimum:", probability.min().item())
    print("Probability maximum:", probability.max().item())

    assert probability.shape == target.shape, "Prediction/target shape mismatch"
    assert torch.isfinite(probability).all(), "Non-finite predictions"
    assert ((probability >= 0) & (probability <= 1)).all(), "Probabilities outside [0, 1]"

    print("GPU memory allocated:", round(torch.cuda.memory_allocated() / 1024**2, 2), "MiB")
    print("GPU memory reserved:", round(torch.cuda.memory_reserved() / 1024**2, 2), "MiB")
    print("GPU peak memory allocated:", round(torch.cuda.max_memory_allocated() / 1024**2, 2), "MiB")
    print("GPU peak memory reserved:", round(torch.cuda.max_memory_reserved() / 1024**2, 2), "MiB")

    print("Single-image inference verification passed!")


RESULT_FILES = (
    "threshold_results.csv",
    "calibration_summary.json",
    "dice_vs_threshold.png",
    "iou_vs_threshold.png",
    "precision_recall_vs_threshold.png",
    "per_image_threshold_scores.csv",
)
BASELINE_TOLERANCE = 1e-5
EXPECTED_DICE = 0.8193828024
EXPECTED_IOU = 0.7212754168
RESOURCE_LIMITS = {
    "startup_minimum_free_gpu_mib": 1800,
    "startup_minimum_available_commit_mib": 2500,
    "minimum_free_gpu_mib": 256,
    "minimum_available_commit_mib": 512,
    "maximum_peak_tensor_allocation_mib": 2200,
    "maximum_reserved_gpu_mib": 3500,
    "minimum_available_ram_mib": 32,
    "consecutive_low_ram_checks": 3,
    "monitor_interval_seconds": 5,
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def cuda_memory():
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }


def resource_snapshot():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("available_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    fields = "memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    values = [float(value.strip()) for value in result.stdout.strip().splitlines()[0].split(",")]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **dict(zip(("device_total_mib", "device_used_mib", "device_free_mib",
                    "gpu_utilization_percent", "temperature_c"), values)),
        "available_ram_mib": status.available_physical / 1024**2,
        "available_commit_mib": status.available_page / 1024**2,
        **cuda_memory(),
    }


class ResourceMonitor:
    """Sample whole-device and host headroom; main loop checks every image."""

    def __init__(self):
        self.samples = []
        self.stage = "startup"
        self.failure = None
        self.low_ram_checks = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self, startup=False):
        snapshot = {"stage": self.stage, **resource_snapshot()}
        self.samples.append(snapshot)
        free_limit = RESOURCE_LIMITS[
            "startup_minimum_free_gpu_mib" if startup else "minimum_free_gpu_mib"
        ]
        commit_limit = RESOURCE_LIMITS[
            "startup_minimum_available_commit_mib" if startup else "minimum_available_commit_mib"
        ]
        self.low_ram_checks = (
            self.low_ram_checks + 1
            if snapshot["available_ram_mib"] < RESOURCE_LIMITS["minimum_available_ram_mib"] else 0
        )
        if (snapshot["device_free_mib"] < free_limit
                or snapshot["available_commit_mib"] < commit_limit
                or self.low_ram_checks >= RESOURCE_LIMITS["consecutive_low_ram_checks"]):
            raise RuntimeError(f"Resource headroom limit exceeded: {snapshot}")
        self.check_allocator()
        return snapshot

    @staticmethod
    def check_allocator():
        memory = cuda_memory()
        if (memory["peak_allocated_mib"] > RESOURCE_LIMITS["maximum_peak_tensor_allocation_mib"]
                or memory["reserved_mib"] > RESOURCE_LIMITS["maximum_reserved_gpu_mib"]):
            raise RuntimeError(f"CUDA memory limit exceeded: {memory}")

    def check(self):
        if self.failure:
            raise RuntimeError(self.failure)
        self.check_allocator()
        if (OUTPUT_DIR / "STOP_REQUESTED").exists():
            raise InterruptedError("STOP_REQUESTED exists; stopping without selecting a threshold")

    def run(self):
        while not self.stop_event.wait(RESOURCE_LIMITS["monitor_interval_seconds"]):
            try:
                self.sample()
            except Exception as exc:
                self.failure = str(exc)
                return

    def finish(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=20)

    def summary(self):
        return {
            **cuda_memory(),
            "sampled_peak_device_used_mib": max(
                (s["device_used_mib"] for s in self.samples), default=None
            ),
            "sampled_minimum_device_free_mib": min(
                (s["device_free_mib"] for s in self.samples), default=None
            ),
            "sampled_minimum_available_commit_mib": min(
                (s["available_commit_mib"] for s in self.samples), default=None
            ),
            "sampling_interval_seconds": RESOURCE_LIMITS["monitor_interval_seconds"],
            "limits": RESOURCE_LIMITS,
            "samples": self.samples,
        }


def count_metrics(tp, fp, fn):
    """Exactly the unsmoothed Phase 2 per-image/pooled empty-mask conventions."""
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def score_probability_map(probability, truth):
    """Score all thresholds on CPU; no GPU probabilities survive this image."""
    rows = []
    true_pixels = int(np.count_nonzero(truth))
    for threshold in THRESHOLDS:
        predicted = probability >= float(threshold)
        predicted_pixels = int(np.count_nonzero(predicted))
        tp = int(np.count_nonzero(predicted & truth))
        fp, fn = predicted_pixels - tp, true_pixels - tp
        rows.append({
            "threshold": float(threshold), **count_metrics(tp, fp, fn),
            "tp": tp, "fp": fp, "fn": fn,
            "tn": int(truth.size) - tp - fp - fn,
            "predicted_foreground_fraction": predicted_pixels / truth.size,
            "true_foreground_fraction": true_pixels / truth.size,
        })
    return rows


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def save_calibration_plots(rows, selected_threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    thresholds = [row["threshold"] for row in rows]
    specs = (
        ("dice_vs_threshold.png", "Validation Dice vs threshold", (("mean_dice", "Mean Dice"),)),
        ("iou_vs_threshold.png", "Validation IoU vs threshold", (("mean_iou", "Mean IoU"),)),
        ("precision_recall_vs_threshold.png", "Validation precision and recall vs threshold",
         (("mean_precision", "Mean precision"), ("mean_recall", "Mean recall"))),
    )
    palette = ("#245AA5", "#B87210")
    for filename, title, series in specs:
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        for index, (metric, label) in enumerate(series):
            ax.plot(thresholds, [row[metric] for row in rows],
                    color=palette[index], marker="o" if index == 0 else "s",
                    linestyle="-" if index == 0 else "--", linewidth=1.8,
                    markersize=4, label=label)
        ax.axvline(0.5, color="#666666", linestyle=":", linewidth=1.3, label="Baseline: 0.500")
        ax.axvline(selected_threshold, color="#202020", linestyle="--", linewidth=1.3,
                   label=f"Selected by validation Dice: {selected_threshold:.3f}")
        ax.set(title=title, xlabel="Probability threshold", ylabel="Per-image macro score",
               xlim=(0.19, 0.81), ylim=(0, 1.02))
        ax.set_xticks(np.arange(0.2, 0.81, 0.1))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color="#DEDEDE", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower left", frameon=False, fontsize=9)
        fig.suptitle("1,423 validation images | Epoch 2 | Validation-selected calibration, not test performance",
                     fontsize=10, color="#444444")
        with (OUTPUT_DIR / filename).open("xb") as handle:
            fig.savefig(handle, format="png", dpi=180)
        plt.close(fig)


def calibrate_thresholds():
    """One FP32 forward per validation image, followed by 25 CPU threshold scores."""
    existing = [name for name in RESULT_FILES if (OUTPUT_DIR / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing calibration outputs: {existing}")
    if len(THRESHOLDS) != 25 or not np.array_equal(THRESHOLDS, np.round(0.2 + 0.025 * np.arange(25), 3)):
        raise ValueError("Expected exactly 25 thresholds from 0.200 to 0.800")
    if count_metrics(0, 0, 0) != {"dice": 1.0, "iou": 1.0, "precision": 0.0, "recall": 0.0}:
        raise AssertionError("Empty-mask metric convention changed")
    if count_metrics(0, 0, 4) != {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}:
        raise AssertionError("Empty-prediction metric convention changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this verified memory-safe loading path")

    monitor = ResourceMonitor()
    model = None
    completed_images = 0
    inference_calls = 0
    baseline_verification = {"passed": False, "tolerance": BASELINE_TOLERANCE}
    provenance = {}
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        monitor.sample(startup=True)
        monitor.thread.start()
        monitor.check()
        monitor.stage = "verify checkpoint and validation provenance"
        if best["epoch"] != 2:
            raise ValueError("Calibration requires the best Epoch 2 checkpoint")
        baseline_path = PILOT_DIR / "validation" / "epoch_2" / "summary.json"
        with baseline_path.open(encoding="utf-8") as handle:
            expected = json.load(handle)
        checkpoint_sha = file_sha256(CHECKPOINT)
        if checkpoint_sha != best["sha256"] or checkpoint_sha != expected["checkpoint_sha256"]:
            raise ValueError("Checkpoint SHA256 does not match the verified Epoch 2 evaluation")
        if expected["images"] != 1423 or expected["threshold"] != 0.5:
            raise ValueError("Saved baseline uses a different validation protocol")
        reference_csv = PILOT_DIR / expected["per_image_csv"]
        if file_sha256(reference_csv) != expected["per_image_csv_sha256"]:
            raise ValueError("Original per-image validation CSV hash mismatch")
        with reference_csv.open(newline="", encoding="utf-8") as handle:
            reference_rows = list(csv.DictReader(handle))
        if len(reference_rows) != 1423:
            raise ValueError("Original validation CSV is incomplete")
        with (PILOT_DIR / "config.json").open(encoding="utf-8") as handle:
            pilot_config = json.load(handle)
        annotation_sha = file_sha256(Config().train_csv)
        if annotation_sha != pilot_config["train_csv_sha256"]:
            raise ValueError("Training annotation CSV changed since Phase 2")

        monitor.stage = "load validation dataset"
        dataset = load_validation_dataset()
        with (PILOT_DIR / "data_manifest.json").open(encoding="utf-8") as handle:
            original_records = json.load(handle)["validation"]
        if len(original_records) != len(dataset):
            raise ValueError("Validation manifest size mismatch")
        for index, (record, original, reference) in enumerate(zip(dataset.records, original_records, reference_rows)):
            if (record["name"] != original["name"] or record["prompt"] != original["prompt"]
                    or record["name"] != reference["image"]
                    or record["image"].resolve() != (ROOT / original["image"]).resolve()
                    or record["mask"].resolve() != (ROOT / original["mask"]).resolve()):
                raise ValueError(f"Validation membership/prompt/path mismatch at index {index}")
        provenance = {
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_epoch": 2,
            "checkpoint_sha256": checkpoint_sha,
            "reference_summary": str(baseline_path.relative_to(ROOT)),
            "reference_summary_sha256": file_sha256(baseline_path),
            "reference_per_image_csv": str(reference_csv.relative_to(ROOT)),
            "reference_per_image_csv_sha256": expected["per_image_csv_sha256"],
            "validation_manifest_sha256": file_sha256(PILOT_DIR / "data_manifest.json"),
            "train_annotation_csv_sha256": annotation_sha,
            "validation_membership_prompts_paths_match": True,
        }
        monitor.check()
        monitor.stage = "load best Epoch 2 model"
        model = load_best_model()
        model.eval()
        if any(module.training for module in model.modules()):
            raise AssertionError("A model submodule is still in training mode")
        parameter_versions = {name: value._version for name, value in model.named_parameters()}
        bn_before = {
            name: value.detach().cpu().clone() for name, value in model.named_buffers()
            if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        }
        checkpoint_stat = (CHECKPOINT.stat().st_size, CHECKPOINT.stat().st_mtime_ns)
        monitor.sample()

        metric_names = ("dice", "iou", "precision", "recall")
        count_names = ("tp", "fp", "fn", "tn")
        metric_values = np.empty((len(dataset), len(THRESHOLDS), 4), dtype=np.float64)
        counts = np.zeros((len(THRESHOLDS), 4), dtype=np.int64)
        baseline_index = int(np.flatnonzero(THRESHOLDS == 0.5)[0])
        per_image_differences = np.zeros(4, dtype=np.float64)
        baseline_confusion_mismatch_images = 0
        loop_started = time.perf_counter()
        minimum_probability, maximum_probability = 1.0, 0.0
        fields = ("validation_index", "image", "threshold", *metric_names, *count_names,
                  "predicted_foreground_fraction", "true_foreground_fraction")
        with (OUTPUT_DIR / "per_image_threshold_scores.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            with torch.inference_mode():
                for index in range(len(dataset)):
                    monitor.stage = f"validation image {index + 1}/{len(dataset)}"
                    monitor.check()
                    (image, text), target = dataset[index]
                    # Match Phase 2's plain tensors; do not retain MONAI tensor metadata on CUDA.
                    if hasattr(image, "as_tensor"):
                        image = image.as_tensor()
                    if hasattr(target, "as_tensor"):
                        target = target.as_tensor()
                    image = image.unsqueeze(0).cuda()
                    text = {key: value.unsqueeze(0).cuda() for key, value in text.items()}
                    probability = model([image, text])
                    inference_calls += 1
                    if tuple(probability.shape) != (1, *target.shape):
                        raise ValueError(f"Prediction/target shape mismatch at validation index {index}")
                    if tuple(probability.shape) != (1, 1, 224, 224):
                        raise ValueError("Expected the original 224x224 binary segmentation protocol")
                    probability_cpu = probability[0, 0].cpu().numpy().copy()
                    truth = target[0].cpu().numpy() > 0.5
                    monitor.check_allocator()
                    del image, text, target, probability
                    if not np.isfinite(probability_cpu).all():
                        raise FloatingPointError(f"Non-finite prediction at validation index {index}")
                    pmin, pmax = float(probability_cpu.min()), float(probability_cpu.max())
                    if not 0.0 <= pmin <= pmax <= 1.0:
                        raise ValueError(f"Probabilities outside [0,1] at validation index {index}")
                    minimum_probability = min(minimum_probability, pmin)
                    maximum_probability = max(maximum_probability, pmax)
                    image_rows = score_probability_map(probability_cpu, truth)
                    for threshold_index, row in enumerate(image_rows):
                        metric_values[index, threshold_index] = [row[name] for name in metric_names]
                        counts[threshold_index] += [row[name] for name in count_names]
                        writer.writerow({"validation_index": index,
                                         "image": dataset.records[index]["name"], **row})
                    actual_baseline = image_rows[baseline_index]
                    per_image_differences = np.maximum(per_image_differences, [
                        abs(actual_baseline[name] - float(reference_rows[index][name])) for name in metric_names
                    ])
                    baseline_confusion_mismatch_images += int(any(
                        actual_baseline[name] != int(reference_rows[index][name]) for name in count_names
                    ))
                    completed_images += 1
                    handle.flush()
                    del probability_cpu, truth, image_rows
                    if completed_images == 1 or completed_images % 100 == 0 or completed_images == len(dataset):
                        print(f"CALIBRATION {completed_images}/{len(dataset)} | "
                              f"{time.perf_counter() - loop_started:.1f}s | CUDA {cuda_memory()}", flush=True)
                    if completed_images % 100 == 0:
                        gc.collect()
                os.fsync(handle.fileno())
        inference_and_scoring_seconds = time.perf_counter() - loop_started
        monitor.check()
        means = metric_values.mean(axis=0)
        results = []
        for threshold_index, threshold in enumerate(THRESHOLDS):
            tp, fp, fn, tn = (int(value) for value in counts[threshold_index])
            global_metrics = count_metrics(tp, fp, fn)
            total_pixels = tp + fp + fn + tn
            results.append({
                "threshold": float(threshold),
                **{f"mean_{name}": float(means[threshold_index, j]) for j, name in enumerate(metric_names)},
                **{f"global_{name}": value for name, value in global_metrics.items()},
                "predicted_foreground_percent": 100.0 * (tp + fp) / total_pixels,
                "GT_foreground_percent": 100.0 * (tp + fn) / total_pixels,
                "images": completed_images, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })
        baseline = results[baseline_index]
        differences = {name: abs(baseline[name] - expected[name])
                       for name in ("mean_dice", "mean_iou", "mean_precision", "mean_recall")}
        baseline_verification.update({
            "expected": {name: expected[name] for name in differences},
            "actual": baseline, "absolute_differences": differences,
            "maximum_per_image_absolute_differences": dict(zip(metric_names, per_image_differences.tolist())),
            "confusion_mismatch_images": baseline_confusion_mismatch_images,
            "user_baseline_dice_absolute_difference": abs(baseline["mean_dice"] - EXPECTED_DICE),
            "user_baseline_iou_absolute_difference": abs(baseline["mean_iou"] - EXPECTED_IOU),
        })
        baseline_verification["passed"] = (
            completed_images == 1423 and inference_calls == 1423
            and all(value <= BASELINE_TOLERANCE for value in differences.values())
            and abs(baseline["mean_dice"] - EXPECTED_DICE) <= BASELINE_TOLERANCE
            and abs(baseline["mean_iou"] - EXPECTED_IOU) <= BASELINE_TOLERANCE
        )
        if not baseline_verification["passed"]:
            raise RuntimeError(f"Baseline mismatch; no threshold selected: {baseline_verification}")
        print("THRESHOLD 0.5 BASELINE VERIFIED", baseline, flush=True)
        if any(value._version != parameter_versions[name] for name, value in model.named_parameters()):
            raise RuntimeError("Model parameters changed during calibration")
        if any(not torch.equal(value.cpu(), bn_before[name])
               for name, value in model.named_buffers() if name in bn_before):
            raise RuntimeError("Frozen BatchNorm statistics changed during calibration")
        if checkpoint_stat != (CHECKPOINT.stat().st_size, CHECKPOINT.stat().st_mtime_ns):
            raise RuntimeError("Source checkpoint file changed during calibration")
        # Only select after full baseline verification; exact ties prefer proximity to 0.5.
        selected = max(results, key=lambda row: (row["mean_dice"], -abs(row["threshold"] - 0.5)))
        monitor.stage = "write verified calibration outputs"
        with (OUTPUT_DIR / "threshold_results.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        memory_at_completion = cuda_memory()
        del model
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        monitor.check()
        save_calibration_plots(results, selected["threshold"])
        monitor.check()
        monitor.finish()
        artifacts = {name: {"path": str(OUTPUT_DIR / name), "sha256": file_sha256(OUTPUT_DIR / name)}
                     for name in RESULT_FILES if name != "calibration_summary.json"}
        summary = {
            "status": "completed", "completed_images": completed_images,
            "inference_calls": inference_calls, "thresholds": THRESHOLDS.tolist(),
            "evaluation_scope": "Validation-selected threshold calibration; not test performance",
            "test_dataset_loaded": False, "training_performed": False,
            "model_parameters_unchanged": True, "batchnorm_buffers_unchanged": True,
            "provenance": provenance,
            "protocol": {
                "split": "validation", "split_seed": 2026, "image_size": 224,
                "batch_size": 1, "precision": "FP32", "augmentation": False,
                "model_eval": True, "torch_inference_mode": True,
                "one_forward_per_image": True, "threshold_comparison": "probability >= threshold",
                "truth_comparison": "target > 0.5", "primary_aggregation": "per-image macro means",
                "global_aggregation": "pooled pixel counts", "smoothing": 0,
                "empty_mask_conventions": "Dice/IoU=1 when both masks empty; precision/recall=0 for empty denominator",
                "selection": "Maximum mean per-image Dice; exact ties closest to 0.5",
            },
            "baseline_verification": baseline_verification,
            "baseline_threshold": 0.5, "baseline_results": baseline,
            "best_validation_threshold": selected["threshold"], "best_validation_results": selected,
            "changes_vs_baseline": {
                name: selected[name] - baseline[name]
                for name in ("mean_dice", "mean_iou", "mean_precision", "mean_recall",
                             "global_dice", "global_iou", "predicted_foreground_percent")
            },
            "inference_and_scoring_seconds": inference_and_scoring_seconds,
            "total_processing_seconds": time.perf_counter() - PROCESS_STARTED,
            "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_at_inference_completion": memory_at_completion,
            "resources": monitor.summary(), "artifacts": artifacts,
        }
        write_new_json(OUTPUT_DIR / "calibration_summary.json", summary)
        print("BEST VALIDATION THRESHOLD", selected, flush=True)
        print("CHANGES VS BASELINE", summary["changes_vs_baseline"], flush=True)
        print("TOTAL PROCESSING SECONDS", summary["total_processing_seconds"], flush=True)
        print("Saved calibration outputs:", OUTPUT_DIR, flush=True)
        return summary
    except BaseException as exc:
        monitor.finish()
        failure = {
            "status": "failed", "completed_images": completed_images,
            "inference_calls": inference_calls, "best_validation_threshold": None,
            "error": f"{type(exc).__name__}: {exc}", "stage": monitor.stage,
            "baseline_verification": baseline_verification, "provenance": provenance,
            "resources": monitor.summary(),
            "total_processing_seconds": time.perf_counter() - PROCESS_STARTED,
            "note": "Partial CSVs, if any, are retained. No automatic retry or threshold selection.",
        }
        if not (OUTPUT_DIR / "calibration_summary.json").exists():
            write_new_json(OUTPUT_DIR / "calibration_summary.json", failure)
        raise
    finally:
        monitor.finish()
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\nStarting full-validation threshold calibration...", flush=True)
    calibrate_thresholds()
