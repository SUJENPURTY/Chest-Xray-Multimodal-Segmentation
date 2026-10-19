"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const state = { file: null, preview: null, urls: [], busy: false, model: null, selection: 0 };
  const maxBytes = Number(document.body.dataset.maxImageBytes);
  const prompt = $("prompt");
  function error(message = "") { $("error-message").textContent = message; $("error-message").hidden = !message; }
  function updateButton() {
    $("run-button").disabled = state.busy || !state.file || !prompt.value.trim() || !state.model || state.model.busy || state.model.state === "unavailable" || state.model.state === "loading";
  }
  function clearResult() {
    state.urls.forEach(url => URL.revokeObjectURL(url)); state.urls = [];
    ["original-result", "mask-result", "overlay-result", "compare-original", "compare-overlay"].forEach(id => $(id).removeAttribute("src"));
    ["download-mask", "download-overlay"].forEach(id => $(id).removeAttribute("href"));
    $("result-content").hidden = true; $("result-empty").hidden = false;
    $("results-state").textContent = "Awaiting input";
    $("empty-title").textContent = "A clearer view starts here";
    $("empty-description").textContent = "Upload a chest X-ray and add a text prompt. Your prediction will appear in this workspace.";
  }
  function setBusy(busy) {
    state.busy = busy;
    ["prompt", "image-input", "remove-image", "reset-button"].forEach(id => $(id).disabled = busy);
    document.querySelectorAll("[data-prompt]").forEach(button => button.disabled = busy);
    $("results-panel").setAttribute("aria-busy", String(busy));
    $("run-spinner").hidden = !busy; $("run-arrow").hidden = busy;
    $("run-label").textContent = busy ? "Running segmentation…" : "Run segmentation";
    if (busy) { $("results-state").textContent = "Processing"; $("empty-title").textContent = "Running the Phase-2 pipeline"; $("empty-description").textContent = "The first request verifies and loads the trained model. Keep this tab open; cold startup can take several minutes."; }
    updateButton();
  }
  async function selectFile(file) {
    if (!file || state.busy) return;
    // Invalidate the previous selection before validating or decoding a replacement.
    // This also cancels pending decodes, so stale images/results cannot reappear.
    removeFile(false);
    const selection = state.selection;
    error();
    if (!/\.(png|jpe?g)$/i.test(file.name) || (file.type && !["image/png", "image/jpeg"].includes(file.type))) { $("image-input").value = ""; error("Choose a grayscale PNG or JPEG image."); return; }
    if (!file.size || file.size > maxBytes) { $("image-input").value = ""; error(`Choose a nonempty image up to ${document.body.dataset.maxUploadMb} MiB.`); return; }
    const url = URL.createObjectURL(file);
    const preview = new Image(); preview.src = url;
    try { await preview.decode(); } catch { URL.revokeObjectURL(url); if (selection === state.selection) { $("image-input").value = ""; error("The browser could not read this image. Choose a valid PNG or JPEG."); } return; }
    if (selection !== state.selection) { URL.revokeObjectURL(url); return; }
    if (state.preview) URL.revokeObjectURL(state.preview);
    state.file = file; state.preview = url;
    $("input-preview").src = url; $("input-preview").hidden = false; $("upload-placeholder").hidden = true;
    $("file-details").hidden = false; $("file-name").textContent = file.name;
    clearResult(); updateButton();
  }
  function removeFile(clearInput = true) {
    state.selection++; state.file = null;
    if (state.preview) URL.revokeObjectURL(state.preview);
    state.preview = null; $("input-preview").removeAttribute("src"); $("input-preview").hidden = true;
    $("upload-placeholder").hidden = false; $("file-details").hidden = true;
    if (clearInput) $("image-input").value = "";
    clearResult(); updateButton();
  }
  async function refreshStatus() {
    try {
      const response = await fetch("/api/model-status", { cache: "no-store" });
      if (!response.ok) throw new Error("Model status unavailable");
      const model = await response.json(); state.model = model;
      $("model-pill").dataset.state = model.state;
      $("model-label").textContent = ({ ready: "Model ready", loading: "Loading model", unavailable: "Model unavailable", not_loaded: "Loads on first run" })[model.state] || "Checking model";
      if (model.ready && model.busy) $("model-label").textContent = "Prediction in progress";
      $("model-message").textContent = model.message;
    } catch {
      state.model = null; $("model-pill").dataset.state = "unavailable"; $("model-label").textContent = "Server unavailable";
      $("model-message").textContent = "The local server is not responding. Check that the application is running.";
    }
    updateButton();
  }
  function imageURL(image) {
    if (image.mime_type !== "image/png") throw new Error("Unexpected result format");
    const binary = atob(image.base64); const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: "image/png" })); state.urls.push(url); return url;
  }
  function displayResult(result) {
    const images = Object.fromEntries(Object.entries(result.images).map(([key, image]) => [key, imageURL(image)]));
    $("original-result").src = images.original; $("mask-result").src = images.mask; $("overlay-result").src = images.overlay;
    $("compare-original").src = images.original; $("compare-overlay").src = images.overlay;
    $("download-mask").href = images.mask; $("download-overlay").href = images.overlay;
    $("comparison-frame").style.aspectRatio = `${result.images.original.width} / ${result.images.original.height}`;
    $("result-meta").replaceChildren();
    const m = result.metadata;
    [`Epoch ${m.epoch}`, `${m.device.toUpperCase()} · ${m.precision}`, `Grid ${m.image_size} × ${m.image_size}`, `Threshold ${m.threshold}`, `${m.processing_seconds.toFixed(2)} s · processing`].forEach(text => { const span = document.createElement("span"); span.textContent = text; $("result-meta").append(span); });
    $("reveal").value = "50"; $("opacity").value = "100"; updateComparison();
    $("result-empty").hidden = true; $("result-content").hidden = false; $("results-state").textContent = "Prediction complete";
  }
  function updateComparison() {
    const reveal = Number($("reveal").value), opacity = Number($("opacity").value);
    $("overlay-layer").style.clipPath = `inset(0 ${100 - reveal}% 0 0)`;
    $("compare-overlay").style.opacity = String(opacity / 100);
    $("comparison-divider").style.left = `${reveal}%`;
    $("reveal-value").textContent = `${reveal}%`; $("opacity-value").textContent = `${opacity}%`;
  }
  $("image-input").addEventListener("change", event => selectFile(event.target.files[0]));
  $("drop-zone").addEventListener("click", () => { if (!state.busy) $("image-input").click(); });
  $("drop-zone").addEventListener("keydown", event => { if (["Enter", " "].includes(event.key) && !state.busy) { event.preventDefault(); $("image-input").click(); } });
  ["dragover", "dragenter"].forEach(type => $("drop-zone").addEventListener(type, event => { event.preventDefault(); if (!state.busy) $("drop-zone").classList.add("drag-over"); }));
  ["dragleave", "drop"].forEach(type => $("drop-zone").addEventListener(type, event => { event.preventDefault(); $("drop-zone").classList.remove("drag-over"); }));
  $("drop-zone").addEventListener("drop", event => { if (state.busy) return; if (event.dataTransfer.files.length !== 1) { removeFile(); error("Choose one image at a time."); } else selectFile(event.dataTransfer.files[0]); });
  $("remove-image").addEventListener("click", removeFile);
  prompt.addEventListener("input", () => { $("character-count").textContent = `${prompt.value.length.toLocaleString()} / 4,000`; clearResult(); error(); updateButton(); });
  document.querySelectorAll("[data-prompt]").forEach(button => button.addEventListener("click", () => { prompt.value = button.dataset.prompt; prompt.dispatchEvent(new Event("input")); }));
  $("reset-button").addEventListener("click", () => { removeFile(); prompt.value = ""; prompt.dispatchEvent(new Event("input")); error(); });
  $("prediction-form").addEventListener("submit", async event => {
    event.preventDefault(); if (state.busy) return;
    if (!state.file || !prompt.value.trim()) { error("Choose an image and enter a segmentation prompt."); return; }
    error(); clearResult(); setBusy(true);
    const body = new FormData(); body.append("image", state.file); body.append("prompt", prompt.value.trim());
    const poll = setInterval(refreshStatus, 2500);
    try {
      const response = await fetch("/api/predict", { method: "POST", headers: { "X-Requested-With": "XMLHttpRequest" }, body });
      if (!(response.headers.get("content-type") || "").includes("application/json")) throw new Error(response.status === 413 ? "The upload exceeds the configured size limit." : "The server could not complete the request. Check the local application log.");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error?.message || "Segmentation could not finish.");
      displayResult(data);
    } catch (failure) { clearResult(); error(failure.message || "The server is unavailable. Check the local application."); }
    finally { clearInterval(poll); setBusy(false); await refreshStatus(); }
  });
  ["reveal", "opacity"].forEach(id => $(id).addEventListener("input", updateComparison));
  let dragging = false;
  const frame = $("comparison-frame");
  function dragTo(event) { const rect = frame.getBoundingClientRect(); $("reveal").value = String(Math.round(Math.max(0, Math.min(100, 100 * (event.clientX - rect.left) / rect.width)))); updateComparison(); }
  frame.addEventListener("pointerdown", event => { dragging = true; frame.setPointerCapture(event.pointerId); dragTo(event); });
  frame.addEventListener("pointermove", event => { if (dragging) dragTo(event); });
  ["pointerup", "pointercancel"].forEach(type => frame.addEventListener(type, () => dragging = false));
  $("fullscreen-button").addEventListener("click", async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await $("viewer").requestFullscreen(); } catch { error("Fullscreen is not available in this browser. The inline comparison remains available."); } });
  refreshStatus(); setInterval(() => { if (!state.busy && !document.hidden) refreshStatus(); }, 15000);
})();
