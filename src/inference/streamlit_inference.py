"""FP32 demo inference using the validated model and MONAI transforms."""
from io import BytesIO
import gc
import json
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ToTensord

from src.config import from_environment
from src.data.dataset import image_transform
from src.inference.predictor import _model_sources, png_bytes
from src.models.languide_medseg import create_model
from src.training.metrics import binary_metrics


def decode_upload(data, mask=False):
    if not data or len(data) > 20 * 1024 * 1024:
        raise ValueError("Upload a nonempty image smaller than 20 MB.")
    with Image.open(BytesIO(data)) as source:
        if source.format not in {"PNG", "JPEG"}:
            raise ValueError("Only PNG and JPEG images are supported.")
        source.verify()
    with Image.open(BytesIO(data)) as source:
        if source.width * source.height > 40_000_000:
            raise ValueError("Image exceeds the 40 megapixel limit.")
        # Preserve grayscale pixel values, including 16-bit PNGs.
        return source.copy() if source.mode in {"L", "I", "I;16", "F", "1"} else source.convert("L")


def transform_upload(image, mask=False):
    transform = (Compose([LoadImaged(["image"], reader="PILReader"),
                          EnsureChannelFirstd(["image"]),
                          Resized(["image"], (224, 224), mode="nearest"),
                          ToTensord(["image"])]) if mask else image_transform(224))
    with tempfile.TemporaryDirectory(prefix="languide-demo-") as folder:
        path = Path(folder) / "image.png"
        image.save(path)
        tensor = transform({"image": str(path)})["image"].as_tensor().float().unsqueeze(0)
    if tensor.shape != (1, 1, 224, 224) or not torch.isfinite(tensor).all():
        raise ValueError("Image must produce a finite single-channel 224 x 224 tensor.")
    return (tensor > 0).float() if mask else tensor


class DemoPredictor:
    def __init__(self, checkpoint_path):
        self.cfg = from_environment()
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {path}")
        vision, text = _model_sources(self.cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(text, trust_remote_code=True)
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if saved.get("kind") != "CLOUD_LANGUIDE_BINARY_FIXED_V1" or not isinstance(saved.get("model"), dict):
            raise ValueError("Unsupported checkpoint format.")
        state = saved["model"]
        if not state or any(not key.startswith("model.") for key in state):
            raise ValueError("Unexpected checkpoint model keys.")
        self.epoch = saved.get("epoch")
        self.model = create_model(vision, text)
        self.model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()},
                                   strict=True, assign=True)
        del state, saved
        gc.collect()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device=self.device, dtype=torch.float32).eval()
        self.lock = threading.Lock()

    def predict(self, data, prompt):
        if not prompt.strip():
            raise ValueError("Enter a nonempty medical text prompt.")
        original = decode_upload(data)
        tensor = transform_upload(original).to(self.device)
        tokens = self.tokenizer(prompt.strip(), padding="max_length", truncation=True,
                                max_length=24, return_tensors="pt")
        tokens = {key: tokens[key].to(self.device) for key in ("input_ids", "attention_mask")}
        # Serialize access to the shared cached model; warm up before measuring.
        with self.lock, torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
            self.model([tensor, tokens])
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            output = self.model([tensor, tokens])
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latency = (time.perf_counter() - started) * 1000
            if output.shape != (1, 1, 224, 224):
                raise ValueError("Model output must have shape (1, 1, 224, 224).")
            probability = output.float().cpu().numpy()[0, 0].copy()
        if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise ValueError("Model produced invalid probabilities.")
        return {"original": original, "probability": probability,
                "prompt": prompt.strip(), "inference_latency_ms": latency,
                "device": str(self.device), "checkpoint_epoch": self.epoch}


def build_results(prediction, threshold=0.5, ground_truth=None):
    if not 0.1 <= threshold <= 0.9:
        raise ValueError("Threshold must be between 0.1 and 0.9.")
    probability = prediction["probability"]
    binary = (probability >= threshold).astype(np.uint8)
    if not set(np.unique(binary)).issubset({0, 1}):
        raise ValueError("Invalid binary mask.")
    # Display all predictions at native model resolution in PIL orientation.
    base = prediction["original"].convert("L").resize((224, 224), Image.Resampling.BICUBIC).convert("RGB")
    rgb = np.asarray(base).copy()
    selected = binary.T.astype(bool)
    rgb[selected] = (0.55 * rgb[selected] + 0.45 * np.array([255, 65, 65])).astype(np.uint8)
    images = {"predicted_mask.png": Image.fromarray(binary.T * 255),
              "probability_map.png": Image.fromarray(np.rint(probability.T * 255).astype(np.uint8)),
              "prediction_overlay.png": Image.fromarray(rgb)}
    metadata = {key: prediction[key] for key in ("prompt", "device", "inference_latency_ms", "checkpoint_epoch")}
    metadata.update(threshold=threshold, predicted_infection_area_percent=float(binary.mean() * 100),
                    dice=None, iou=None, pixel_accuracy=None, precision="FP32",
                    resolution=[224, 224], area_denominator="all model-image pixels",
                    probability_png_encoding="uint8: round(probability * 255)")
    comparison = None
    if ground_truth is not None:
        target = transform_upload(decode_upload(ground_truth, mask=True), mask=True)
        metadata.update(binary_metrics(torch.from_numpy(probability)[None, None], target, threshold))
        actual = target.numpy()[0, 0].T.astype(bool)
        comparison_rgb = np.asarray(base).copy()
        colors = np.zeros_like(comparison_rgb)
        colors[actual & ~selected] = [0, 220, 255]
        colors[selected & ~actual] = [255, 65, 65]
        colors[selected & actual] = [255, 220, 0]
        union = actual | selected
        comparison_rgb[union] = (0.5 * comparison_rgb[union] + 0.5 * colors[union]).astype(np.uint8)
        comparison = Image.fromarray(comparison_rgb)
    downloads = {name: png_bytes(image) for name, image in images.items()}
    downloads["inference_result.json"] = json.dumps(metadata, indent=2, allow_nan=False).encode()
    return metadata, images, comparison, downloads
