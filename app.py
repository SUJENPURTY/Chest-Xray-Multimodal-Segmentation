"""Streamlit entry point: python -m streamlit run app.py."""
import hashlib
import logging
import streamlit as st
from matplotlib.figure import Figure

from src.config import from_environment
from src.inference.streamlit_inference import DemoPredictor, decode_upload, build_results

st.set_page_config(page_title="Chest X-Ray Segmentation", page_icon="🩻", layout="wide")


@st.cache_resource(show_spinner="Loading checkpoint and medical tokenizer…")
def load_predictor(checkpoint_path):
    return DemoPredictor(checkpoint_path)


st.title("Text-Guided Chest X-Ray Infection Segmentation")
st.caption("Multimodal segmentation using chest X-ray + medical text prompt")
st.warning("Research prototype only. This application is not intended for clinical diagnosis or medical decision-making.")
with st.expander("How it works"):
    st.write("Chest X-ray + Text Prompt → Vision Encoder + Medical Text Encoder → Text-guided Cross-Attention → Segmentation Decoder → Infection Mask")
st.subheader("Inputs")
left, right = st.columns([1, 1.3])
uploaded = left.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])
ground_truth = left.file_uploader("Optional ground-truth mask", type=["png", "jpg", "jpeg"])
left.caption("Ground truth must align with the X-ray. Nonzero mask pixels are foreground; lossless PNG is recommended.")
examples = ["Bilateral pulmonary infection, two infected areas, all left lung and all right lung.",
            "Pulmonary infection.",
            "Unilateral pulmonary infection, one infected area, lower right lung."]
def select_example():
    st.session_state["prompt"] = st.session_state["example"]
right.selectbox("Example prompt", examples, key="example", on_change=select_example)
prompt = right.text_area("Medical text prompt", value=examples[0], key="prompt", height=120)
threshold = right.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)
if abs(threshold - 0.5) > 1e-8:
    right.info("Custom threshold selected. Validated project results use 0.5.")
right.caption("Images use the validated MONAI preprocessing at 224 × 224. Downloads use this resolution.")
data = uploaded.getvalue() if uploaded else None
identity = (hashlib.sha256(data).hexdigest(), prompt.strip()) if data else None
if st.button("Run Segmentation", type="primary"):
    st.session_state.pop("prediction", None)
    try:
        if data is None:
            raise ValueError("Upload a chest X-ray first.")
        decode_upload(data)
        if not prompt.strip():
            raise ValueError("Enter a nonempty medical text prompt.")
        predictor = load_predictor(str(from_environment().checkpoint_path.resolve()))
        with st.spinner("Running FP32 segmentation…"):
            result = predictor.predict(data, prompt)
        st.session_state["prediction"] = (identity, result)
    except Exception as exc:
        logging.exception("Demo inference failed")
        st.error(f"Segmentation could not run: {str(exc)[:400]}")
saved = st.session_state.get("prediction")
if saved and saved[0] == identity:
    prediction = saved[1]
    try:
        metadata, images, comparison, downloads = build_results(
            prediction, threshold, ground_truth.getvalue() if ground_truth else None)
    except Exception as exc:
        st.error(f"Ground-truth comparison could not be processed: {str(exc)[:300]}")
        metadata, images, comparison, downloads = build_results(prediction, threshold)
    st.subheader("Results")
    cards = st.columns(4)
    cards[0].metric("Predicted Infection Area", f"{metadata['predicted_infection_area_percent']:.2f}%")
    cards[1].metric("Threshold", f"{threshold:.2f}")
    cards[2].metric("Warm Inference Latency", f"{metadata['inference_latency_ms']:.2f} ms")
    cards[3].metric("Device · FP32", metadata["device"].upper())
    st.caption("Area is the fraction of all 224 × 224 image pixels, not the fraction of lung tissue. Latency measures a warmed model forward pass only.")
    panels = st.columns(4)
    panels[0].image(prediction["original"], caption="Original Chest X-ray", width="stretch")
    with panels[1]:
        figure = Figure(figsize=(4, 4))
        axis = figure.subplots()
        heat = axis.imshow(prediction["probability"].T, cmap="inferno", vmin=0, vmax=1)
        axis.axis("off")
        figure.colorbar(heat, ax=axis, fraction=0.046, pad=0.04)
        st.pyplot(figure, width="stretch")
        st.caption("Probability Map · fixed scale 0–1")
    panels[2].image(images["predicted_mask.png"], caption="Predicted Binary Mask", width="stretch")
    panels[3].image(images["prediction_overlay.png"], caption="X-ray + Prediction Overlay", width="stretch")
    if comparison is not None:
        for column, key, title in zip(st.columns(3), ["dice", "iou", "pixel_accuracy"], ["Dice Score", "IoU", "Pixel Accuracy"]):
            column.metric(title, f"{metadata[key]:.6f}")
        st.image(comparison, caption="Ground Truth vs Prediction: cyan = ground truth only; red = prediction only; yellow = overlap", width=450)
    st.subheader("Download results")
    for column, (name, artifact) in zip(st.columns(4), downloads.items()):
        column.download_button(name, artifact, file_name=name,
                               mime="application/json" if name.endswith("json") else "image/png")
elif saved:
    st.info("Inputs changed. Run Segmentation to generate a result for the current image and prompt.")
with st.expander("Model details"):
    st.write("Architecture: LanGuideMedSeg · Vision + medical text multimodal model")
    st.write("Input resolution: 224 × 224 · Text length: 24 tokens · Selected checkpoint: Epoch 18")
    st.write("Inference: FP32 · Validated threshold: 0.5 · CUDA when available, otherwise CPU")
    st.code(str(from_environment().checkpoint_path))
with st.expander("Performance and limitations"):
    st.write("Held-out test samples: 2,113 · Macro Dice: 0.185754 · Macro IoU: 0.106600")
    st.write("Global Dice: 0.225872 · Global IoU: 0.127314")
    st.write("Previously validated warm latency: approximately 66.35 ms/sample on the tested CUDA setup.")
    st.write("Segmentation quality is limited; predictions can be fragmented. Text guidance has a limited effect. CPU inference may be slower.")
