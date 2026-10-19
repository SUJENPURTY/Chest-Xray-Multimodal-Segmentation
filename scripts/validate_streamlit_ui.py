"""Quick result-panel regression check using saved notebook artifacts."""
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest
from src.inference import streamlit_inference  # Import ML stack before AppTest's UI timeout.

root = Path(__file__).resolve().parents[1]
folder = root / "outputs/run/inference/sample_000"
data = (folder / "xray.png").read_bytes()
prompt = "Pulmonary infection."
result = {"original": Image.open(folder / "xray.png").copy(),
          "probability": np.asarray(Image.open(folder / "probability_map.png"), dtype=np.float32).T / 255,
          "prompt": prompt, "device": "cuda", "inference_latency_ms": 1.0, "checkpoint_epoch": 18}
source = (root / "app.py").read_text(encoding="utf-8").replace(
    "data = uploaded.getvalue() if uploaded else None", "data = st.session_state['test_image_bytes']")
for with_gt in (False, True):
    test_source = source.replace("ground_truth.getvalue() if ground_truth else None",
                                "st.session_state['test_gt']") if with_gt else source
    app = AppTest.from_string(test_source)
    app.session_state["test_image_bytes"] = data
    app.session_state["test_gt"] = (folder / "ground_truth_mask.png").read_bytes()
    app.session_state["prompt"] = prompt
    app.session_state["prediction"] = ((hashlib.sha256(data).hexdigest(), prompt), result)
    app.run(timeout=60)
    assert not app.exception, app.exception
    assert not app.error, app.error
    assert len(app.metric) == (7 if with_gt else 4)
    app.slider[0].set_value(0.7).run(timeout=60)
    assert not app.exception
    assert app.metric[1].value == "0.70"
print("Final UI rendering, ground-truth panels, and threshold reruns passed.")
