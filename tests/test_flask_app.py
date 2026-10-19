"""HTTP contracts with explicit fixtures; real model parity is checked separately."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image

from webapp import create_app
from webapp.services.inference_service import Phase2Service, ServiceError

HEADERS = {"X-Requested-With": "XMLHttpRequest"}


def png(mode="L", size=(23, 31)):
    buffer = BytesIO()
    Image.new(mode, size).save(buffer, format="PNG")
    return buffer.getvalue()


class FixtureService:
    """Transport fixture only; it is never used by run.py or the real validators."""
    def __init__(self):
        self.calls = []
        self.error = None

    def status(self):
        return {"state": "not_loaded", "ready": False, "message": "Test fixture", "missing_resources": []}

    def predict(self, path, prompt):
        assert path.name == "input.png"
        self.calls.append((path.read_bytes(), prompt, path))
        if self.error:
            raise self.error
        with Image.open(path) as source:
            image = source.convert("RGB")
        return {"original": image, "binary_mask": image.convert("L"), "overlay": image}, {"epoch": 2, "device": "cpu", "precision": "FP32", "threshold": 0.5, "image_size": 224, "processing_seconds": 0.01}


@pytest.fixture
def setup(tmp_path):
    service = FixtureService()
    app = create_app({"TESTING": True, "TEMP_UPLOAD_ROOT": tmp_path / "uploads", "MAX_CONTENT_LENGTH": 100_000}, service=service)
    return app.test_client(), service, tmp_path / "uploads"


def post(client, data=None, prompt="Pulmonary infection.", filename="scan.png", headers=None):
    return client.post("/api/predict", headers=HEADERS if headers is None else headers,
                       data={"image": (BytesIO(png() if data is None else data), filename), "prompt": prompt})


def test_home_static_health_and_security_headers(setup):
    client, service, _ = setup
    for path in ("/", "/static/css/style.css", "/static/js/app.js", "/static/icon.svg", "/api/health", "/api/model-status"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert b"Not for clinical diagnosis" in client.get("/").data
    assert client.get("/api/model-status").json["ready"] is False
    assert service.calls == []


@pytest.mark.parametrize("data,prompt,filename,code", [
    (b"not an image", "text", "scan.png", 415),
    (b"", "text", "scan.png", 415),
    (png("RGB"), "text", "scan.png", 415),
    (png(), " ", "scan.png", 400),
    (png(), "x" * 4001, "scan.png", 400),
    (png(), "text", "scan.exe", 415),
])
def test_invalid_inputs_never_invoke_inference(setup, data, prompt, filename, code):
    client, service, _ = setup
    assert post(client, data, prompt, filename).status_code == code
    assert not service.calls


def test_oversize_missing_file_and_cross_origin(setup):
    client, service, _ = setup
    assert client.post("/api/predict", headers=HEADERS, data={"prompt": "text"}).status_code == 400
    assert post(client, b"x" * 110_000).status_code == 413
    assert post(client, headers={}).status_code == 403
    assert post(client, headers={**HEADERS, "Origin": "https://example.invalid"}).status_code == 403
    assert client.get("/api/health", headers={"Host": "evil.invalid"}).status_code == 400
    assert not service.calls


def test_ground_truth_is_not_silently_accepted(setup):
    client, service, _ = setup
    response = client.post("/api/predict", headers=HEADERS,
                           data={"image": (BytesIO(png()), "scan.png"),
                                 "ground_truth": (BytesIO(png()), "truth.png"), "prompt": "text"})
    assert response.status_code == 400 and not service.calls


def test_image_limit_is_distinct_from_multipart_limit():
    service = FixtureService()
    app = create_app({"TESTING": True, "MAX_IMAGE_BYTES": 1000,
                      "MAX_CONTENT_LENGTH": 4096}, service=service)
    client = app.test_client()
    image = png()
    boundary_image = image + b"\0" * (1000 - len(image))
    assert post(client, boundary_image).status_code == 200
    assert len(service.calls) == 1
    assert post(client, boundary_image + b"\0").status_code == 413
    assert len(service.calls) == 1
    assert post(client, image, prompt="x" * 3900).status_code == 413
    assert len(service.calls) == 1
    assert b'data-max-image-bytes="1000"' in client.get("/").data


def test_default_request_budget_reserves_multipart_overhead():
    app = create_app(service=FixtureService())
    assert app.config["MAX_CONTENT_LENGTH"] == app.config["MAX_IMAGE_BYTES"] + 64 * 1024


def test_advertised_limit_accepts_full_image_and_max_unicode_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_MAX_UPLOAD_MB", "1")
    service = FixtureService()
    app = create_app({"TESTING": True, "TEMP_UPLOAD_ROOT": tmp_path}, service=service)
    client = app.test_client()
    limit = app.config["MAX_IMAGE_BYTES"]
    # Valid PNG with trailing padding exercises multipart parsing at the exact
    # advertised byte boundary without allocating a large decoded image.
    image = png()
    boundary = image + b"\0" * (limit - len(image))
    assert post(client, boundary, prompt="\U0001f9ea" * 4000).status_code == 200
    assert service.calls[0][0] == boundary
    assert post(client, boundary + b"\0").status_code == 413
    assert len(service.calls) == 1
    assert b'data-max-upload-mb="1"' in client.get("/").data


def test_bytes_prompt_dimensions_and_request_cleanup(setup):
    client, service, root = setup
    data = png()
    response = post(client, data, "  finding <script>  ", "../../scan.png")
    assert response.status_code == 200
    assert service.calls[0][0] == data and service.calls[0][1] == "finding <script>"
    assert not service.calls[0][2].exists() and not list(root.iterdir())
    result = response.json
    assert set(result["images"]) == {"original", "mask", "overlay"}
    assert all(item["width"] == 23 and item["height"] == 31 for item in result["images"].values())
    assert not {"dice", "iou", "confidence", "accuracy"} & set(result["metadata"])
    assert response.headers["Cache-Control"] == "no-store"
    assert "scan.png" not in response.text


def test_failures_are_sanitized_and_clean_up(setup):
    client, service, root = setup
    service.error = RuntimeError("private/checkpoints/secret.ckpt")
    response = post(client)
    assert response.status_code == 500
    assert "secret" not in response.text and not list(root.iterdir())
    service.error = ServiceError("model_busy", "Try again.", 409)
    assert post(client).status_code == 409 and not list(root.iterdir())


def test_real_service_missing_resources_and_busy_gate(tmp_path):
    service = Phase2Service(tmp_path)
    assert service.status()["state"] == "unavailable"
    with pytest.raises(ServiceError) as missing:
        service.predict(tmp_path / "not_read.png", "prompt")
    assert missing.value.code == "resources_missing" and service.load_count == 0
    service._gate.acquire()
    try:
        with pytest.raises(ServiceError) as busy:
            service.predict(Path("unused"), "prompt")
        assert busy.value.status == 409
    finally:
        service._gate.release()


def test_service_reuses_an_existing_engine_without_loading(tmp_path):
    service = Phase2Service(tmp_path)

    class CachedEngine:
        device = "cpu"
        cfg = SimpleNamespace(threshold=0.5, image_size=224, max_text_length=24)

        def predict_file(self, path, prompt):
            assert service.status()["busy"] is True
            return {"transport_fixture": True}

    service._engine = CachedEngine()
    service._provenance = {"epoch": 2}
    service.load_count = 1
    service._set_state("ready", "Cached engine")
    for _ in range(2):
        result, metadata = service.predict(Path("fixture-only"), "prompt")
        assert result["transport_fixture"] and metadata["device"] == "cpu"
    assert service.load_count == 1 and not service.status()["busy"]
