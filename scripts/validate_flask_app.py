"""Real Phase-2 API/reference check using the previously validated sample 686.

Default: test a running local Flask application. --cpu: run an isolated Flask
test client with a real CPU model, then exit. No training or original output writes.
Requires the retained private dataset, manifest, checkpoint and reference outputs.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (args.output_dir or ROOT / "outputs/flask_validation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        parser.error("Use a new directory below outputs/.")
    output.mkdir(parents=True)
    record = json.loads((ROOT / "experiments/phase2_pilot/data_manifest.json").read_text())["validation"][686]
    image_path = ROOT / record["image"].replace("\\", "/")
    reference = ROOT / "outputs/run/phase2_live_inference"
    headers = {"X-Requested-With": "XMLHttpRequest"}
    report = {"status": "running", "real_checkpoint": True, "sample_validation_index": 686,
              "mode": "cpu_test_client" if args.cpu else "live_http", "training_performed": False}
    try:
        if args.cpu:
            from webapp import create_app
            app = create_app({"PHASE2_DEVICE": "cpu"})
            service = app.extensions["phase2_service"]
            captured = {}
            original = service.predict

            def capture(*arguments):
                result, metadata = original(*arguments)
                captured["probability"] = result["probability_224"]
                return result, metadata

            service.predict = capture
            with app.test_client() as client:
                response = client.post("/api/predict", headers=headers,
                                       data={"image": (BytesIO(image_path.read_bytes()), "validation.png"), "prompt": record["prompt"]})
                assert response.status_code == 200, response.json
                result = response.json
                assert service.load_count == 1 and service._engine.device.type == "cpu"
            import numpy as np
            expected = np.load(reference / "sample_686_probability_224.npy")
            report["probability_max_absolute_difference"] = float(np.max(np.abs(captured["probability"] - expected)))
            np.testing.assert_allclose(captured["probability"], expected, rtol=5e-4, atol=5e-5)
            report["probability_parity_tolerance"] = {"atol": 5e-5, "rtol": 5e-4}
        else:
            import requests
            health = requests.get(args.url + "/api/health", timeout=15)
            assert health.status_code == 200 and health.json()["status"] == "ok"
            print("Submitting one real Phase-2 prediction; cold startup may take several minutes.", flush=True)
            with image_path.open("rb") as handle:
                response = requests.post(args.url + "/api/predict", headers=headers,
                                         files={"image": ("validation.png", handle, "image/png")},
                                         data={"prompt": record["prompt"]}, timeout=1200)
            assert response.status_code == 200, response.text[:400]
            result = response.json()
            status = requests.get(args.url + "/api/model-status", timeout=15).json()
            assert status["ready"] and status["device"] in ("cpu", "cuda")
        assert not any(key in result.get("metadata", {}) for key in ("dice", "iou", "accuracy", "confidence"))
        decoded = {name: Image.open(BytesIO(base64.b64decode(value["base64"]))).copy() for name, value in result["images"].items()}
        with Image.open(image_path) as source:
            assert decoded["original"].size == source.size
            assert ImageChops.difference(decoded["original"].convert("RGB"), source.convert("RGB")).getbbox() is None
        differences = {}
        for name, filename in (("mask", "sample_686_prediction_native.png"), ("overlay", "sample_686_overlay_native.png")):
            with Image.open(reference / filename) as expected:
                actual = decoded[name]
                assert actual.size == expected.size
                delta = ImageChops.difference(actual.convert("RGB"), expected.convert("RGB")).convert("L")
                count = delta.width * delta.height - delta.histogram()[0]
                differences[name] = count
                # GPU reference normally matches exactly; allow <=0.01% differing
                # native pixels for CPU threshold crossings from FP32 rounding.
                assert count == 0 if not args.cpu else count / (delta.width * delta.height) <= 1e-4, (name, count)
            actual.save(output / (name + ".png"))
        assert decoded["mask"].mode == "L" and set(decoded["mask"].getdata()) <= {0, 255}
        assert not list((ROOT / "outputs/webapp/tmp").glob("request_*"))
        report.update(status="passed", device=result["metadata"]["device"],
                      metadata=result["metadata"], differing_native_pixels=differences,
                      original_alignment=True, mask_binary=True, reference_overlay_pixels_verified=True,
                      temporary_upload_removed=True, invented_metrics_absent=True)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
