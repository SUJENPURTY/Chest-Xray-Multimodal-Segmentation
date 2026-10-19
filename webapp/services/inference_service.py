"""One locked, lazy Phase-2 engine per process; no new inference implementation."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
import time

APPROVED_CHECKPOINT_SHA256 = "09487406a6783da76c97746bf24e5ba4134317dfd9d7654ef84863cf7d71c7cc"


class ServiceError(Exception):
    def __init__(self, code, message, status=503):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class Phase2Service:
    def __init__(self, root: Path, device="auto", cache_root=None):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("PHASE2_DEVICE must be auto, cpu, or cuda")
        self.root = Path(root).resolve()
        self.requested_device = device
        self.cache_root = Path(cache_root or self.root / "outputs/webapp/cache")
        self._gate = threading.Lock()
        self._state_lock = threading.Lock()
        self._state, self._message = "not_loaded", "Model verification will run on the first prediction."
        self._engine = self._provenance = None
        self._failed = False
        self.load_count = 0

    def _set_state(self, state, message):
        with self._state_lock:
            self._state, self._message = state, message

    def _missing_resources(self):
        pilot = self.root / "experiments/phase2_pilot"
        required = {"checkpoint_selection": pilot / "checkpoints/best.json",
                    "training_configuration": pilot / "config.json",
                    "training_manifest": self.root / "outputs/run/prepared_annotations/train.csv"}
        missing = [label for label, path in required.items() if not path.is_file()]
        if missing:
            return missing
        try:
            best = json.loads(required["checkpoint_selection"].read_text(encoding="utf-8"))
            checkpoint = (pilot / best["path"].replace("\\", "/")).resolve()
            summary_path = (pilot / best["validation_summary"]).resolve()
            if not checkpoint.is_relative_to(pilot) or not summary_path.is_relative_to(pilot):
                return ["invalid_checkpoint_provenance"]
            if best["sha256"] != APPROVED_CHECKPOINT_SHA256:
                return ["unapproved_checkpoint"]
            if not checkpoint.is_file():
                missing.append("selected_checkpoint")
            if not summary_path.is_file():
                missing.append("validation_summary")
            else:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                table = (pilot / summary["per_image_csv"]).resolve()
                if not table.is_relative_to(pilot) or not table.is_file():
                    missing.append("historical_validation_table")
        except (OSError, ValueError, KeyError, TypeError):
            return ["invalid_checkpoint_provenance"]
        return missing

    def status(self):
        with self._state_lock:
            state, message = self._state, self._message
        missing = self._missing_resources() if state == "not_loaded" else []
        if missing:
            state, message = "unavailable", "Restore the documented Phase-2 checkpoint and provenance bundle."
        busy = self._gate.locked()
        if busy and state == "ready":
            message = "A prediction is running. The verified model remains loaded."
        result = {"state": state, "ready": state == "ready", "busy": busy, "message": message,
                  "missing_resources": missing, "device": str(self._engine.device) if self._engine else None}
        if self._engine:
            result["configuration"] = {"epoch": self._provenance["epoch"],
                                       "threshold": self._engine.cfg.threshold,
                                       "image_size": self._engine.cfg.image_size,
                                       "max_text_length": self._engine.cfg.max_text_length,
                                       "precision": "FP32"}
        return result

    def _load(self):
        if self._engine is not None:
            return
        if self._failed:
            raise ServiceError("model_unavailable", self._message)
        if self._missing_resources():
            raise ServiceError("resources_missing", "Restore the documented Phase-2 checkpoint and provenance bundle before predicting.")
        self._set_state("loading", "Verifying the checkpoint and loading the Phase-2 model. The first request can take several minutes.")
        try:
            for name in ("hf_modules", "matplotlib"):
                (self.cache_root / name).mkdir(parents=True, exist_ok=True)
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                              HF_MODULES_CACHE=str(self.cache_root / "hf_modules"),
                              MPLCONFIGDIR=str(self.cache_root / "matplotlib"),
                              CUBLAS_WORKSPACE_CONFIG=":4096:8")
            import random
            import numpy as np
            import torch
            from src.inference.phase2 import Phase2Inference, phase2_provenance
            device = ("cuda" if torch.cuda.is_available() else "cpu") if self.requested_device == "auto" else self.requested_device
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA is not available")
            random.seed(2026)
            np.random.seed(2026)
            torch.manual_seed(2026)
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if device == "cuda":
                torch.cuda.manual_seed_all(2026)
            cfg, provenance = phase2_provenance(self.root)
            if provenance["checkpoint_sha256"] != APPROVED_CHECKPOINT_SHA256:
                raise ValueError("Checkpoint is not the approved Phase-2 artifact")
            engine = Phase2Inference(cfg, device)
            self._engine, self._provenance = engine, provenance
            self.load_count += 1
            self._set_state("ready", "Phase-2 model verified and ready.")
        except Exception:
            logging.getLogger(__name__).exception("Phase-2 initialization failed")
            self._failed = True
            self._set_state("unavailable", "The Phase-2 model could not load. Check the local log, restore its trusted resources, then restart.")
            raise ServiceError("model_unavailable", self._message) from None

    def predict(self, image_path, prompt):
        if not self._gate.acquire(blocking=False):
            raise ServiceError("model_busy", "Another prediction is running. Please try again when it finishes.", 409)
        try:
            self._load()
            started = time.perf_counter()
            # All preprocessing, tokenization, inference mode, placement, thresholding,
            # orientation, native resizing and overlay pixels come from this method.
            result = self._engine.predict_file(image_path, prompt)
            return result, {**self.status()["configuration"], "device": str(self._engine.device),
                            "processing_seconds": time.perf_counter() - started}
        finally:
            self._gate.release()
