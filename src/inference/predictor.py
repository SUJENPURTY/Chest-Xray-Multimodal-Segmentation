"""Single checkpoint-compatible multimodal inference implementation."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from src.config import Config, CXR_BERT_ID, LEGACY, provision_checkpoint
from src.data.dataset import image_transform
from src.models.languide_medseg import create_model
from src.models.pretrained import resolve_model


def _model_sources(cfg):
    vision = resolve_model(cfg.vision_model, cfg.vision_model)
    bundled = LEGACY / "lib" / "BiomedVLP-CXR-BERT-specialized-official"
    text = str(bundled) if cfg.text_model == CXR_BERT_ID and bundled.is_dir() else cfg.text_model
    return vision, resolve_model(text, CXR_BERT_ID)


def load_checkpoint(model, checkpoint_path, device="cpu"):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint missing: {path}. Set MODEL_CHECKPOINT.")
    # Checkpoints are pickle files. Load only a checkpoint from a trusted source.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or saved.get("kind") != "CLOUD_LANGUIDE_BINARY_FIXED_V1" or "model" not in saved:
        raise ValueError("Unsupported LanGuide checkpoint format")
    state = saved["model"]
    if not isinstance(state, dict) or any(not isinstance(key, str) or not key.startswith("model.") for key in state):
        raise ValueError("Unexpected checkpoint state keys")
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()}, strict=True)
    return model.to(device).eval(), saved


def render_prediction(original, probability_224, threshold=0.5):
    if not 0 < threshold < 1:
        raise ValueError("Threshold must lie strictly between 0 and 1")
    if probability_224.shape != (224, 224) or not np.isfinite(probability_224).all():
        raise ValueError("Model produced an invalid probability map")
    # MONAI PILReader uses [x,y]; transpose to PIL's [y,x] before display.
    image = Image.fromarray(probability_224.T.astype(np.float32))
    full = np.asarray(image.resize(original.size, Image.Resampling.BILINEAR))
    binary = full >= threshold
    mask = Image.fromarray((binary * 255).astype(np.uint8))
    probability_image = Image.fromarray(np.clip(full * 255, 0, 255).astype(np.uint8))
    red = Image.new("RGBA", original.size, (255, 0, 0, 0))
    red.putalpha(Image.fromarray((binary * 110).astype(np.uint8)))
    overlay = Image.alpha_composite(original.convert("RGBA"), red).convert("RGB")
    return {"binary_mask": mask, "probability_image": probability_image,
            "overlay": overlay, "probability_map": full}


class LanGuidePredictor:
    def __init__(self, cfg: Config, device=None):
        cfg = provision_checkpoint(cfg)
        if not cfg.checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint missing: {cfg.checkpoint_path}. Set MODEL_CHECKPOINT or MODEL_CHECKPOINT_URL.")
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        vision, text = _model_sources(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(text, trust_remote_code=True)
        model = create_model(vision, text)
        self.model, self.checkpoint = load_checkpoint(model, cfg.checkpoint_path, self.device)
        self.transform = image_transform(cfg.image_size)

    def predict(self, image: Image.Image | bytes | str | Path, text_prompt: str, threshold=None):
        if not text_prompt or not text_prompt.strip():
            raise ValueError("Enter a nonempty clinical/location prompt")
        temporary = None
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                source.verify()
            with Image.open(image) as source:
                original = source.convert("RGB").copy()
            input_path = Path(image)
        elif isinstance(image, bytes):
            with Image.open(BytesIO(image)) as source:
                source.verify()
            with Image.open(BytesIO(image)) as source:
                original = source.convert("RGB").copy()
            if original.width * original.height > 40_000_000:
                raise ValueError("Image is too large; maximum 40 megapixels")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                handle.write(image)
                temporary = Path(handle.name)
            input_path = temporary
        elif isinstance(image, Image.Image):
            original = image.convert("RGB").copy()
            if original.width * original.height > 40_000_000:
                raise ValueError("Image is too large; maximum 40 megapixels")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                temporary = Path(handle.name)
            image.save(temporary)
            input_path = temporary
        else:
            raise TypeError("image must be a PIL image, bytes or a file path")
        if original.width * original.height > 40_000_000:
            raise ValueError("Image is too large; maximum 40 megapixels")
        try:
            tensor = self.transform({"image": str(input_path)})["image"].unsqueeze(0).to(self.device)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        tokens = self.tokenizer(text_prompt.strip(), padding="max_length",
                                max_length=self.cfg.max_text_length, truncation=True,
                                return_attention_mask=True, return_tensors="pt")
        tokens = {key: tokens[key].to(self.device) for key in ("input_ids", "attention_mask")}
        started = time.monotonic()
        with torch.inference_mode():
            probability = self.model([tensor, tokens]).float().squeeze().cpu().numpy()
        duration = time.monotonic() - started
        visuals = render_prediction(original, probability,
                                    self.cfg.threshold if threshold is None else threshold)
        return {"original": original, "probability_224": probability,
                "inference_time": duration, "device": str(self.device), **visuals}


def png_bytes(image: Image.Image):
    data = BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()
