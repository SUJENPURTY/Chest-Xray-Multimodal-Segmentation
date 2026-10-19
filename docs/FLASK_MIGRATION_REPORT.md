# Flask migration implementation and validation report

## Implementation

The new application is launched with `python run.py` at
**http://127.0.0.1:5000**. The original Streamlit `app.py` remains byte-identical
and still uses the original epoch-18 workflow. Flask uses the selected Phase-2
epoch-2 checkpoint. No model code, architecture, checkpoint, numbered notebook,
original evaluation result, or final Phase-2 inference output was rewritten.

Created files:

- `run.py`: loopback-only Waitress entry point, one process and four threads.
- `webapp/__init__.py`: Flask factory, limits, trusted hosts, response security
  headers, and sanitized error handling; startup does not import the ML stack.
- `webapp/routes.py`: homepage, health, model status, and multipart prediction API;
  grayscale validation, unchanged upload bytes, temporary cleanup, metadata-free
  PNG encoding, and no fabricated metrics.
- `webapp/services/{__init__,inference_service}.py`: thread-safe lazy Phase-2
  adapter, trusted artifact fingerprint, busy/readiness state, cached model.
- `webapp/templates/index.html`, `webapp/static/css/style.css`,
  `webapp/static/js/app.js`, `webapp/static/icon.svg`: responsive navy interface,
  drag/drop upload, prompts, previews, actual result panels, comparison and opacity
  controls, fullscreen, reset, and native-resolution PNG downloads.
- `scripts/validate_flask_app.py`: real live/CPU reference checks, private reports.
- `scripts/validate_flask_browser.py`: real Chromium integration, responsive views,
  upload interactions, sliders, fullscreen, downloads and reference pixel checks.
- `tests/test_flask_app.py`: explicit transport fixtures, errors, input limits,
  resource readiness, model reuse, locking and request cleanup.
- `requirements-web-test.txt`: optional browser/live validation dependencies.
- `docs/FLASK_APPLICATION.md`, this report, and `docs/assets/flask_workspace.png`:
  operating documentation and a genuinely captured empty-interface screenshot.

Modified existing files:

- `README.md`: Flask-first startup, screenshot, inputs/outputs, tests and limitations.
- `requirements.txt`: Flask and Waitress runtime requirements.
- `.gitignore`: local Flask instance/uploads and Streamlit secret exclusions.
- `docs/CHECKPOINTS.md`, `docs/REPRODUCIBILITY.md`,
  `docs/GITHUB_CLEANUP_REVIEW.md`: links and migration scope, with prior evidence retained.

The migration baseline is stored privately in
`outputs/flask_migration/20261019/{baseline.json,backup/,packages_before.json}`.
The preservation comparison covers 152 original records, with only six authorized
documentation/configuration files modified; 52 weight metadata records match.
The selected checkpoint is verified by the unchanged provenance loader before
deserialization. The original source module hashes, notebook hashes, and final
Phase-2 output hashes match their baseline. Existing installed package versions
did not change. Only Flask 3.1.3, Werkzeug 3.1.9, Waitress 3.0.2, and Blinker 1.9.0
were added to the existing environment.

## Reused research implementation

`src/inference/phase2.py` owns `phase2_provenance()`, `Phase2Inference`,
`load_phase2_model()`, `tokenize()`, `infer_tensor()` and `predict_file()`.
The web adapter calls these existing functions; it does not duplicate transforms
or execute a training/optimizer path. `src/data/dataset.py:image_transform()` and
`src/inference/predictor.py:render_prediction()` remain unchanged.

The existing MONAI PILReader axes, bicubic resize, intensity normalization,
tokenization, FP32 tensors, inference mode, evaluation mode, native image alignment,
probability resize and thresholding are retained. Threshold, resolution and text
length are read from the verified research configuration. The red overlay is the
original `(255, 0, 0)` foreground with alpha 110. Browser opacity/reveal controls
do not alter masks or downloaded overlay pixels.

## Recorded validation

| Check | Status | Evidence and scope |
| --- | --- | --- |
| Flask/Waitress startup | PASS | Actual `python run.py` process and loopback HTTP response |
| Homepage and static CSS/JavaScript | PASS | Flask tests and real Chromium page loads |
| Image upload | PASS | Actual retained grayscale validation image; invalid inputs rejected |
| Text input | PASS | Real Phase-2 annotation prompt; blank/oversized prompts rejected |
| Model loading and readiness | PASS | Trusted checkpoint loaded on CUDA and CPU; status distinguishes unloaded/loading/ready/unavailable |
| Actual Phase-2 inference | PASS | Real live CUDA API and separate real CPU Flask test client |
| Binary segmentation mask | PASS | Binary PNG, original dimensions, zero differing native pixels versus reference |
| Color overlay | PASS | Canonical renderer, zero differing native pixels versus reference on both devices |
| Result downloads | PASS | Real browser downloaded mask/overlay PNGs; both matched reference pixels even after display slider changes |
| CPU inference | PASS | Real checkpoint; maximum absolute probability difference `2.562999725341797e-6` |
| CUDA inference | PASS | GTX 1650; exact native mask/overlay equality to saved Phase-2 reference |
| Reference parity | PASS | Validation sample 686; CPU probabilities within `atol=5e-5, rtol=5e-4`; zero mask/overlay differences |
| Responsive interface | PASS | Empty and real prediction views at widths 1440, 1024, 768 and 390; no horizontal overflow |
| Comparison, opacity, fullscreen and reset | PASS | Actual Chromium controls, original/reference downloads unaffected, reset clears image/prompt/results |
| Runtime errors and private-path protection | PASS | Explicit fixture errors sanitized; invalid and cross-origin requests rejected |
| Shared inference regressions + Flask tests | PASS | 32 selected tests; unfiltered suite not claimed green |
| Original research preservation | PASS | Seven notebooks, Streamlit app, original source and final Phase-2 outputs unchanged |

CPU/GPU parity is a bounded check on the same retained validation image, not a new
full-dataset evaluation. No clinical confidence, held-out Phase-2 test score, or
new research metric was generated. Processing times in private reports are
observations from a constrained machine, not a performance benchmark.

One early browser test overlapped a separate live validation request and was
correctly rejected as busy. It did not receive a fabricated prediction. A later
serial browser run passed the full acceptance check, including real inference,
drag/drop and replace, example prompt selection, invalid-file feedback, fullscreen,
both downloads, reset and zero browser JavaScript errors. Its prediction screenshots
were inspected locally and remain private.

Private evidence is under `outputs/flask_migration/20261019/`:
`live_validation/report.json`, `cpu_validation/report.json`,
`regression-tests.xml`, `preservation_report.json`, and `browser_polished/report.json`.
The original historical records are not overwritten by these reports.

## Delivery and remaining decisions

Exact Windows PowerShell startup:

```powershell
Set-Location E:\x-ray
.\.venv\Scripts\Activate.ps1
python run.py
```

Overall readiness: **READY for the local research demonstration** with the restored
trusted artifacts. Real local CPU, CUDA and browser inference have passed. The
application is left running on the loopback URL above. Public GitHub release remains
subject to the earlier notebook privacy, image rights and checkpoint-distribution
review. A Git-only clone cannot run real inference until its trusted private
resources are restored.

Known limitations:

- Grayscale single-channel PNG/JPEG only; no silent RGB conversion or DICOM support.
- Slow cold startup on the tested low-memory Windows machine; one prediction at a time.
- No supported ground-truth evaluation upload, authentication, remote TLS deployment,
  clinical validation or full-dataset reevaluation in this migration.
- The historical memory-optimization test has its previously documented stale
  expectation and was preserved without alteration.

Approval is still required before removing the original Streamlit app, its
supporting module/tests/dependency, any original research artifact, dataset or
checkpoint, or publishing medical screenshots/notebooks with private paths.
No such removal or publication was performed. No staging, commit or push occurred.

GitHub readiness checklist:

- [x] Source and frontend are modular, documented and locally testable.
- [x] Original research pipeline and its checkpoint format are preserved.
- [x] Local environments, datasets, checkpoints, uploads and generated evidence excluded.
- [x] Public screenshot is an actual empty UI without medical image data.
- [x] Checkpoint/dataset restoration and dependency instructions documented.
- [ ] Clear original notebook images/private paths for public distribution.
- [ ] Approve any external checkpoint/provenance release and test a fresh clone.
- [ ] Approve any future Streamlit retirement or cache/duplicate deletion.
