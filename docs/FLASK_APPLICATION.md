# Phase-2 Flask research application

## Inspection and migration boundary

The preserved `app.py` and `src/inference/streamlit_inference.py` use the original
epoch-18 `CLOUD_LANGUIDE_BINARY_FIXED_V1` checkpoint. They are not the Phase-2
interface. Notebook 07 uses `src/inference/phase2.py`, the selected epoch-2
`PHASE2_FROZEN_BN_PILOT_V1` checkpoint, its strict provenance checks, and FP32
inference. All seven notebooks, original model/inference modules, checkpoints,
metrics and experiment records are retained. The new package is named `webapp/`
to avoid colliding with the original `app.py`.

The Flask service calls the existing `phase2_provenance()` and constructs one
`Phase2Inference`. Every request calls its unchanged `predict_file()` method.
That method owns the MONAI PILReader orientation, bicubic 224-grid resize,
channel-wise normalization, original tokenizer length, FP32 inference-mode model
execution, and `render_prediction()` postprocessing. No sigmoid is added, no
architecture or weights are changed, and no training is performed.

The loader enforces the trusted checkpoint identity documented in
[CHECKPOINTS.md](CHECKPOINTS.md), in addition to the existing SHA256/provenance
checks. There is no API checkpoint-upload or checkpoint-selection feature.
Changing the approved artifact requires a separately reviewed change.

## Install and run on Windows PowerShell

```powershell
Set-Location path\to\x-ray
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

Open **http://127.0.0.1:5000**. The entry point uses Waitress with one process and
four request threads. A nonblocking inference lock permits one model operation at
a time while health/status requests remain responsive. Do not use multiple worker
processes on a small GPU; each process would load its own model.

Flask and Waitress are required; a notebook frontend is not required for the web
application. The migration installed Flask 3.1.3, Werkzeug 3.1.9, Waitress 3.0.2,
and Blinker 1.9.0 in the existing `.venv`. Existing ML package versions were not
intentionally upgraded. The original Streamlit installation remains available.

Cold initialization imports the ML stack, verifies a roughly 980 MB checkpoint,
and loads weights. It may take several minutes on a machine with limited RAM.
The homepage starts without loading Torch or weights. `not_loaded` is different
from `ready`; the first valid prediction triggers initialization. Subsequent
requests reuse the resident engine. Initialization failures require restoring
the trusted resources and restarting; no placeholder result is returned.

Optional PowerShell settings, before `python run.py`:

```powershell
$env:PHASE2_DEVICE = "cpu"  # auto (default), cpu, or cuda
$env:WEB_PORT = "5000"
$env:WEB_MAX_UPLOAD_MB = "12"
$env:WEB_MAX_IMAGE_PIXELS = "16000000"
```

The device setting changes execution placement only; inference precision and
research parameters stay unchanged. CUDA is selected automatically only when
PyTorch reports availability. Explicit CUDA without supported hardware fails
clearly. There is no silent CPU fallback after a CUDA initialization error.

The server binds to loopback. `WEB_TRUSTED_HOSTS` configures an administrator's
comma-separated trusted hostnames if necessary. Remote deployment, authentication,
TLS and a shared clinical environment are not configured or validated here.
Never enable Flask debug mode or expose a development debugger with medical data.

## Restore resources

Follow [DATASET.md](DATASET.md) and [CHECKPOINTS.md](CHECKPOINTS.md). The web route
processes an uploaded image and does not instantiate the dataset. However, the
existing provenance function still requires the preserved training CSV and
historical validation records alongside the checkpoint. Complete ConvNeXt and
CXR-BERT resources must already be available locally; offline mode is enforced.
No public checkpoint distribution URL has been approved. A Git-only clone is
therefore not sufficient for real inference.

## Inputs, outputs and visualization

- Upload one **single-channel grayscale PNG or JPEG**, within the request-byte and
  pixel limits. The image allowance defaults to 12 MiB (`WEB_MAX_UPLOAD_MB`).
  `MAX_IMAGE_BYTES` is enforced separately from `MAX_CONTENT_LENGTH`, which adds
  64 KiB for the multipart envelope and up to 4,000 prompt characters. Waitress
  uses the same request-byte cap. An excessive request or image returns HTTP 413.
  RGB/RGBA, animated files, DICOM and unsupported formats are rejected.
  RGB-to-grayscale conversion is deliberately not introduced into the validated
  Phase-2 preprocessing. Original bytes are passed to the existing file predictor.
- Selecting any replacement first clears the previous selection, result and
  download links. Invalid extensions, oversize or damaged images cannot leave an
  old prediction attributed to the new file. A selection token also invalidates
  pending browser decodes. This changes interaction correctness, not UI design.
- Enter a nonempty text prompt. The request has a 4,000-character transport limit;
  the original tokenizer's configured truncation is unchanged.
- Outputs are the original RGB display image, binary mask, and canonical overlay,
  all at the original input dimensions. No Dice, IoU, accuracy or clinical confidence
  is returned for an uploaded image. Ground-truth uploads are not supported here;
  extra fields are rejected rather than silently treated as evaluated.
- The overlay comes directly from `render_prediction`: red foreground with the
  existing alpha value 110. The white mask is the existing binary foreground.
  The comparison and opacity controls affect browser display only. Downloaded PNGs
  retain the canonical renderer's pixels and do not depend on slider position.
- The reported processing time measures the existing predictor's preprocessing,
  forward and rendering, excluding initial model loading. It is not an independent
  benchmark or a clinical confidence measurement.

Temporary uploads use generated names under `outputs/webapp/tmp/` and are removed
when the request finishes, including failures. Uploaded filenames never become
server paths. Results are returned as PNG data, with metadata removed and without
server-side result retention. Downloads are browser Blob files, not public URLs.
Closing or resetting the tab releases those results. The browser may keep a copy
if the user downloads one. Application responses use `Cache-Control: no-store`.

## API

| Endpoint | Behavior |
| --- | --- |
| `GET /` | Jinja2 interface; no model initialization |
| `GET /api/health` | Process health plus current model-ready flag |
| `GET /api/model-status` | `not_loaded`, `loading`, `ready`, or `unavailable`; sanitized resource labels/configuration |
| `POST /api/predict` | Multipart `image` and `prompt`; actual Phase-2 inference |

The status response also includes `busy`, independently of whether weights are
ready. The interface disables a new submission while another request is using the
model; the backend lock also rejects races without loading another engine.

Example from PowerShell using the native curl executable:

```powershell
curl.exe http://127.0.0.1:5000/api/health
curl.exe -X POST http://127.0.0.1:5000/api/predict -H "X-Requested-With: XMLHttpRequest" -F "image=@path/to/grayscale.png" -F "prompt=Pulmonary infection."
```

Successful JSON contains `images.{original,mask,overlay}` with PNG MIME type,
base64 data, width and height, plus verified model configuration and measured
processing time. Decode the base64 bytes as PNG; there is no server download URL.
API errors contain a stable code and sanitized message. No checkpoint paths,
patient identifiers, uploaded filenames, input prompts or exception tracebacks
are returned as metadata. Private diagnostic tracebacks stay in the local console.
Cross-origin browser inference is rejected; same-origin requests use the custom
header above. There is no CORS wildcard. Request sizes, form parts, image format,
pixel counts and single-channel mode are checked before loading the model.

## Validation commands

```powershell
python -m pytest tests/test_flask_app.py tests/test_pipeline.py tests/test_phase2_inference.py -q
python scripts/validate_flask_app.py
python scripts/validate_flask_app.py --cpu
python scripts/validate_flask_browser.py
python scripts/audit_public_files.py
```

The browser/live helpers additionally use `requirements-web-test.txt` and an
installed Chrome or Edge executable. `--browser` selects another Chromium binary.
All helpers create new ignored output directories and refuse existing ones.
Their real inference checks require the private validation manifest, input sample
686 and retained reference outputs. They never regenerate or overwrite reference
artifacts. The CPU helper loads a separate real CPU model and checks FP32
probabilities against the preserved reference (`atol=5e-5`, `rtol=5e-4`), allowing
at most 0.01% differing native mask pixels for threshold-boundary rounding.
GPU/live comparisons require exact mask and overlay pixel equality. Numeric
tolerances are validation checks, not changes to inference or its threshold.

`tests/test_flask_app.py` uses explicitly named transport fixtures; those tests
alone do not establish real inference success. Real live, CPU and browser reports
must be reviewed separately. The historical `test_memory_optimizations.py` remains
retained with its previously documented stale checkpointing expectation; no green
claim is made for the complete unfiltered test suite.

Screenshots of the empty interface may be published. Screenshots showing medical
images, real uploaded names, or predictions remain private under `outputs/` until
publication rights and de-identification are approved. The original Streamlit app,
its tests and dependencies must not be removed without the user's approval.

## Implementation references

The framework setup follows Flask's [application factory pattern](https://flask.palletsprojects.com/en/stable/patterns/appfactories/),
[upload-size controls](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/),
and [web security guidance](https://flask.palletsprojects.com/en/stable/web-security/).
The local WSGI entry point uses [Waitress](https://docs.pylonsproject.org/projects/waitress/en/latest/).
Browser verification uses the [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/).
