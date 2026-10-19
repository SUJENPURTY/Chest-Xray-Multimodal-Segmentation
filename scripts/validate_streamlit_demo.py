"""Real held-out inference parity and artifact validation (no training)."""
import hashlib
import json
from io import BytesIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from src.config import ROOT
from src.data.dataset import QaTaDataset
from src.inference.streamlit_inference import DemoPredictor, transform_upload, decode_upload, build_results
from streamlit.testing.v1 import AppTest


def main():
    protected = list((ROOT / "notebooks").glob("*.ipynb")) + list((ROOT / "src").rglob("*.py"))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    assert not app.exception, app.exception
    app.button[0].click().run()
    assert app.error and not app.exception
    predictor = DemoPredictor(ROOT / "outputs/run/checkpoints/best_dice.ckpt")
    dataset = QaTaDataset(ROOT / "outputs/run/prepared_annotations/test.csv", ROOT / "_qata_compatible",
                          "test", predictor.tokenizer, augment=False)
    (image, tokens), target = dataset[0]
    record = dataset.records[0]
    data = record["image"].read_bytes()
    torch.testing.assert_close(transform_upload(decode_upload(data))[0], image.as_tensor(), rtol=0, atol=0)
    result = predictor.predict(data, record["prompt"])
    with torch.inference_mode():
        reference = predictor.model([image.unsqueeze(0).to(predictor.device),
                                     {k: v.unsqueeze(0).to(predictor.device) for k, v in tokens.items()}]).cpu()
    np.testing.assert_allclose(result["probability"], reference.numpy()[0, 0], rtol=0, atol=1e-6)
    assert reference.shape == (1, 1, 224, 224)
    assert all(p.dtype == torch.float32 for p in predictor.model.parameters())
    assert not predictor.model.training
    metadata, images, comparison, downloads = build_results(result, 0.5, record["mask"].read_bytes())
    for key, expected in {"dice": 0.271055, "iou": 0.156775, "pixel_accuracy": 0.800941}.items():
        assert abs(metadata[key] - expected) < 1e-5, (key, metadata[key])
    assert comparison is not None
    # Exercise every result panel/download with real inference artifacts.
    # AppTest cannot populate file uploaders, so use its session state plus
    # a temporary test harness that supplies the real held-out image bytes.
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    source = source.replace("data = uploaded.getvalue() if uploaded else None",
                            "data = st.session_state['test_image_bytes']")
    rendered = AppTest.from_string(source)
    rendered.session_state["test_image_bytes"] = data
    rendered.session_state["prompt"] = record["prompt"]
    rendered.session_state["prediction"] = ((hashlib.sha256(data).hexdigest(), record["prompt"].strip()), result)
    rendered.run(timeout=60)
    assert not rendered.exception, rendered.exception
    assert len(rendered.metric) == 4
    no_gt = build_results(result)[0]
    assert all(no_gt[k] is None for k in ("dice", "iou", "pixel_accuracy"))
    assert set(np.unique(np.asarray(images["predicted_mask.png"]))).issubset({0, 255})
    for name, artifact in downloads.items():
        if name.endswith("png"):
            with Image.open(BytesIO(artifact)) as output:
                output.verify()
        else:
            assert json.loads(artifact)["dice"] == metadata["dice"]
    for bad in (b"invalid image", b""):
        try:
            decode_upload(bad)
        except Exception:
            pass
        else:
            raise AssertionError("Invalid upload accepted")
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    output_dir = ROOT / "outputs/run/streamlit_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"passed": True, "sample": record["name"], "samples": len(dataset),
              "cuda_available": torch.cuda.is_available(), "notebook_probability_parity": True,
              "preprocessing_exact_parity": True, "protected_source_hashes_unchanged": True, **metadata}
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
