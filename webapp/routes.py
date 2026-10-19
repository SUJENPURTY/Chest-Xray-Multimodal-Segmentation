"""Stateless upload and inference API. Uploaded names are never filesystem paths."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
import warnings

from flask import Blueprint, current_app, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from .services.inference_service import ServiceError

bp = Blueprint("web", __name__)


@bp.get("/")
def index():
    image_limit = current_app.config["MAX_IMAGE_BYTES"]
    return render_template("index.html", max_upload_mb=image_limit / 1024**2,
                           max_image_bytes=image_limit)


@bp.get("/api/health")
def health():
    return jsonify(status="ok", service="phase2-segmentation", model_ready=current_app.extensions["phase2_service"].status()["ready"])


@bp.get("/api/model-status")
def model_status():
    return jsonify(current_app.extensions["phase2_service"].status())


def failure(code, message, status):
    return jsonify(error={"code": code, "message": message}), status


def validate_image(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG"}:
                    raise ValueError("Use a PNG or JPEG image.")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Use a single-frame image.")
                if source.width * source.height > current_app.config["MAX_IMAGE_PIXELS"]:
                    raise ValueError("Image dimensions exceed the configured pixel limit.")
                if source.mode not in {"L", "I", "I;16", "I;16B", "I;16L"}:
                    raise ValueError("The validated Phase-2 pipeline requires a single-channel grayscale image. RGB images are not converted automatically.")
                suffix = ".png" if source.format == "PNG" else ".jpg"
                source.verify()
            # verify() checks structure; load() additionally catches truncated pixel data.
            with Image.open(BytesIO(data)) as decoded:
                decoded.load()
            return suffix
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("The image is damaged, unsupported, or too large. Upload a valid grayscale PNG or JPEG.") from None


def image_payload(image):
    # Strip metadata while keeping pixels and dimensions unchanged.
    clean = image.copy()
    clean.info.clear()
    buffer = BytesIO()
    clean.save(buffer, format="PNG")
    return {"mime_type": "image/png", "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "width": image.width, "height": image.height}


@bp.post("/api/predict")
def predict():
    origin = request.headers.get("Origin")
    if (request.headers.get("X-Requested-With") != "XMLHttpRequest"
            or request.headers.get("Sec-Fetch-Site") == "cross-site"
            or (origin and urlsplit(origin).netloc != request.host)):
        return failure("cross_origin_request", "Use the prediction form on this application.", 403)
    if "image" not in request.files or not request.files["image"].filename:
        return failure("image_required", "Choose a chest X-ray image first.", 400)
    if (set(request.files) != {"image"} or set(request.form) - {"prompt"}
            or len(request.files.getlist("image")) != 1 or len(request.form.getlist("prompt")) > 1):
        return failure("unsupported_fields", "Submit one image and one prompt. Ground-truth evaluation is not supported by this endpoint.", 400)
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return failure("prompt_required", "Enter a nonempty segmentation prompt.", 400)
    if len(prompt) > 4000:
        return failure("prompt_too_long", "Keep the prompt under 4,000 characters; the model uses its original tokenizer limit.", 400)
    upload = request.files["image"]
    if Path(upload.filename).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return failure("unsupported_image", "Use a grayscale PNG or JPEG file.", 415)
    data = upload.read(current_app.config["MAX_IMAGE_BYTES"] + 1)
    if len(data) > current_app.config["MAX_IMAGE_BYTES"]:
        return failure("image_too_large", "The image exceeds the displayed image-size limit.", 413)
    try:
        suffix = validate_image(data)
    except ValueError as error:
        return failure("invalid_image", str(error), 415)
    temp_root = Path(current_app.config["TEMP_UPLOAD_ROOT"])
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="request_", dir=temp_root) as directory:
            path = Path(directory) / ("input" + suffix)
            path.write_bytes(data)  # Preserve original bytes for the research transform.
            result, metadata = current_app.extensions["phase2_service"].predict(path, prompt)
            return jsonify(images={"original": image_payload(result["original"]),
                                   "mask": image_payload(result["binary_mask"]),
                                   "overlay": image_payload(result["overlay"])}, metadata=metadata)
    except ServiceError as error:
        return failure(error.code, error.message, error.status)
    except Exception:
        current_app.logger.exception("Prediction failed; request contents are not logged")
        return failure("inference_failed", "Segmentation could not finish. Check the local log; no prediction has been returned.", 500)
