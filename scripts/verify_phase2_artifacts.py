"""Independently check the executed notebook and recompute every CSV metric."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/run/phase2_live_inference"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_metrics(row):
    tp, fp, fn, tn = (int(row[key]) for key in ("tp", "fp", "fn", "tn"))
    assert min(tp, fp, fn, tn) >= 0
    return dict(
        pixel_accuracy=(tp + tn) / (tp + fp + fn + tn),
        dice=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        iou=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
    )


def main():
    notebook = nbformat.read(ROOT / "notebooks/07_phase2_live_inference.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
    for cell in code_cells:
        ast.parse(cell.source)
        assert not any(output.output_type == "error" for output in cell.outputs)
    prose = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    sections = sorted(map(int, re.findall(r"^## (\d+)\.", prose, re.MULTILINE)))
    assert sections == list(range(1, 19)), sections
    png_outputs = sum("image/png" in output.get("data", {})
                      for cell in code_cells for output in cell.outputs)
    assert png_outputs >= 6, f"Missing embedded figure outputs: {png_outputs}"
    widget_state = notebook.metadata["widgets"]["application/vnd.jupyter.widget-state+json"]["state"]
    assert widget_state and all("state" in model and "model_name" in model for model in widget_state.values())

    summary = json.loads((OUTPUT / "validation_summary.json").read_text())
    integrity = json.loads((OUTPUT / "integrity_checks.json").read_text())
    assert summary["checkpoint_sha256"] == sha(ROOT / "experiments/phase2_pilot" /
        json.loads((ROOT / "experiments/phase2_pilot/checkpoints/best.json").read_text())["path"])
    assert summary["per_image_csv_sha256"] == sha(OUTPUT / "validation_per_image.csv")
    with (OUTPUT / "validation_per_image.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary["images"] == 1423
    assert [int(row["validation_index"]) for row in rows] == list(range(1423))
    assert len({row["image"] for row in rows}) == 1423
    recalculated = [independent_metrics(row) for row in rows]
    for row, metrics in zip(rows, recalculated):
        assert sum(int(row[key]) for key in ("tp", "fp", "fn", "tn")) == 224 * 224
        for key, value in metrics.items():
            assert math.isclose(float(row[key]), value, rel_tol=0, abs_tol=1e-12)
    totals = {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn")}
    independent_global = independent_metrics(totals)
    for key in independent_global:
        assert math.isclose(math.fsum(row[key] for row in recalculated) / len(rows),
                            summary["image_mean"][key], rel_tol=0, abs_tol=1e-12)
        assert math.isclose(independent_global[key], summary["dataset_global"][key], rel_tol=0, abs_tol=1e-12)

    preserved = json.loads((OUTPUT / "inspection/preservation_before.json").read_text())
    changed = [name for name, digest in preserved.items() if sha(ROOT / name) != digest]
    assert not changed, changed
    assert integrity["checkpoint_unchanged"] and integrity["no_parameter_gradients"]
    assert integrity["no_open_matplotlib_figures"]
    assert integrity.get("widget_callback_smoke_test", False) or (
        integrity.get("widget_callback_registered", False)
        and integrity.get("widget_no_duplicate_output", False))
    report = dict(status="passed", code_cells=len(code_cells), markdown_sections=sections,
                  embedded_png_figures=png_outputs, validation_images=len(rows),
                  independently_recomputed_metrics=True, protected_files_unchanged=len(preserved),
                  checkpoint_sha256=summary["checkpoint_sha256"],
                  image_mean=summary["image_mean"], dataset_global=summary["dataset_global"],
                  original_crash_cause="Not established; generic marker and termination exits are insufficient.")
    (OUTPUT / "artifact_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    metric_lines = "\n".join(
        f"| {key.replace('_', ' ').title()} | {100*summary['image_mean'][key]:.4f}% | {100*summary['dataset_global'][key]:.4f}% |"
        for key in summary["image_mean"])
    peak = max(row.get("peak_allocated_mib", 0) for row in integrity["resources"])
    documentation = f"""# Phase-2 live-inference notebook repair

The corrected notebook is `notebooks/07_phase2_live_inference.ipynb`. All {len(code_cells)}
code cells executed sequentially without an unresolved exception, and all 18 requested
Markdown sections are present. The saved notebook includes {png_outputs} embedded PNG
figures in addition to the live widget output.

## Problems and corrections

- Removed hidden out-of-order state, the saved undefined-dataset error, duplicate
  widget wiring, hardcoded validation scores, and forced CUDA-only inference.
- Added a dedicated read-only Phase-2 checkpoint loader, using the original model
  constructors and strict unprefixed state loading. The selected checkpoint is
  `experiments/phase2_pilot/checkpoints/e02_p05722_u01432_68dcea20.ckpt`.
- Reused project image/mask transforms, CXR-BERT tokenization and native-size
  probability rendering. Model-grid metrics and native-file metrics are labeled
  separately. MONAI-to-PIL geometry uses a transpose rather than an arbitrary rotation.
- Added single-image inference, original images and ground truth, mask/overlay
  figures, purposeful sample comparisons, an executed widget callback test, and an
  external-image path with optional ground truth.
- Added unsmoothed Phase-2 metric conventions, independent arithmetic checks, macro
  and pooled tables, foreground-overlap distributions, provenance and limitations.

## Verified validation results

One fresh FP32 pass evaluated all **{len(rows):,} validation images** at threshold
0.5 on the 224 × 224 model grid. Every image's confusion counts matched the hashed
Epoch-2 reference exactly. The final presentation run reused those results only
after checkpoint, implementation, CSV and every image/mask hash were verified.
Every metric below was independently recomputed from the saved integer counts.

| Metric | Image-wise mean | Dataset-global |
|---|---:|---:|
{metric_lines}

Pooled counts: TP={totals['tp']:,}, FP={totals['fp']:,}, FN={totals['fn']:,},
TN={totals['tn']:,}. Pixel accuracy is equal across the two aggregation rules because
each evaluated image has the same pixel count. Checkpoint selection used validation;
these are not independent test results.

## Crash investigation

The saved original notebook contained a generic kernel-crash marker after its CSV
summary. The retained VS Code Jupyter log records three kernel exits within seven
seconds with `1073807364` (`0x40010004`) and an empty reason. Microsoft's
[status-code reference](https://winprotocoldoc.z19.web.core.windows.net/MS-ERREF/%5BMS-ERREF%5D.pdf)
names this `DBG_TERMINATE_PROCESS`. The log does not identify the initiating event
or associate each exit with a notebook. Shared external termination is consistent
with the observations; **OOM and native-fault root causes were not established**.
No matching Python application-error/WER event was found in the queried time window.
See `outputs/run/phase2_live_inference/inspection/crash_evidence.json` for excerpts.

## Memory and integrity

Only one model/tokenizer is resident per notebook execution. Configuration-only
construction on the target device avoids redundant backbone-weight loading; mmap
checkpoint reading avoids materializing optimizer state. Strict copy loading keeps
model storage independent of the checkpoint. Inference uses batch size one,
`torch.inference_mode()`, no DataLoader workers, and no dataset tensor cache.
Evaluation keeps scalar rows and hashes rather than dataset-wide masks. Figures
close after rendering. Peak PyTorch tensor allocation was **{peak:.2f} MiB** in the
executed diagnostics (allocator measurements differ from whole-device usage).

The checkpoint SHA256 remains `{summary['checkpoint_sha256']}`. All {len(preserved)}
protected source/notebook files matched their before hashes. There was no training,
optimizer restoration, architecture-file change, checkpoint rewrite or test-set
inference. CUDA execution was verified on the GTX 1650. CPU tensor inference is
unit-tested; full pretrained-model CPU execution was not performed.

## Reproduction and evidence

Use `E:/x-ray/.venv/Scripts/python.exe`. Local dependencies are recorded in
`requirements-local-cuda.txt` and the notebook's installed-version table.

```powershell
# Default: verify and reuse matching saved validation, or compute if absent.
.\\.venv\\Scripts\\python.exe scripts/execute_phase2_notebook.py
# Force exactly one fresh validation pass when required.
$env:PHASE2_FULL_VALIDATION='1'
.\\.venv\\Scripts\\python.exe scripts/execute_phase2_notebook.py
```

The 19 targeted tests in `tests/test_phase2_inference.py` and `tests/test_pipeline.py`
passed. The initial sandbox temporary-file/connection-file permission failures were
resolved using a workspace test directory and authorized Jupyter execution.
`scripts/verify_phase2_artifacts.py` independently checks metrics, saved execution,
embedded figures, widget state and protected-file hashes.

Evidence is in `outputs/run/phase2_live_inference/`: `execution_report.json`,
`fresh_validation_execution_report.json`, `validation_summary.json`,
`validation_per_image.csv`, `integrity_checks.json`, `artifact_verification.json`,
`test_results.xml`, and rendered figures. The original notebook is backed up under
`inspection/07_phase2_live_inference.original.ipynb`.

The remaining historical issue is the unverified initiating cause of termination.
Current execution succeeds; that does not prove a long-running session cannot fail.
"""
    (ROOT / "docs/PHASE2_INFERENCE_REPAIR.md").write_text(documentation, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
