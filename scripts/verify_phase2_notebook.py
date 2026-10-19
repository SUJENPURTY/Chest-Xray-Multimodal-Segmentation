"""Execute a redirected copy of notebook 07; preserve original research artifacts.

Requires the restored Phase-2 bundle and an existing verified validation cache.
Performs a small number of fresh predictions, never training or full evaluation.
Run with the project's .venv interpreter. All new artifacts stay under outputs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from datetime import datetime, timezone

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/07_phase2_live_inference.ipynb"
CACHE = ROOT / "outputs/run/phase2_live_inference"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory below outputs/")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or ROOT / "outputs/verification" / stamp).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()) or output.exists():
        parser.error("Use a new directory strictly below outputs/; existing files are never overwritten.")
    expected_python = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if Path(sys.executable).resolve() != expected_python.resolve():
        parser.error(f"Run with {expected_python}")
    for name in ("validation_summary.json", "validation_per_image.csv"):
        if not (CACHE / name).is_file():
            parser.error(f"Restore the verified evaluation cache first: {CACHE / name}")
    original_sha = sha256(TARGET)
    notebook = nbformat.read(TARGET, as_version=4)
    nbformat.validate(notebook)
    setups = [cell for cell in notebook.cells if cell.cell_type == "code"
              and "OUTPUT = ROOT / 'outputs/run/phase2_live_inference'" in cell.source]
    widgets = [cell.source for cell in notebook.cells if cell.cell_type == "code"
               and "def on_predict(" in cell.source]
    evaluations = [cell for cell in notebook.cells if cell.cell_type == "code"
                   and "summary_path = OUTPUT / 'validation_summary.json'" in cell.source]
    if len(setups) != 1 or len(widgets) != 1 or len(evaluations) != 1:
        raise RuntimeError("Notebook layout changed; review output redirection before execution.")
    output.mkdir(parents=True)
    for name in ("validation_summary.json", "validation_per_image.csv"):
        shutil.copy2(CACHE / name, output / name)
    setups[0].source = setups[0].source.replace(
        "OUTPUT = ROOT / 'outputs/run/phase2_live_inference'", "OUTPUT = Path(" + repr(str(output)) + ")")
    evaluations[0].source = (
        "assert not RUN_FULL_VALIDATION, 'Verification must reuse validated evaluation evidence'\n"
        "assert (OUTPUT / 'validation_summary.json').is_file()\n"
        "assert (OUTPUT / 'validation_per_image.csv').is_file()\n" + evaluations[0].source)
    notebook.cells.append(nbformat.v4.new_code_cell(
        "# Verification-only: exercise real callbacks and widget-cell re-execution.\n"
        "previous_button = predict_button\n"
        "exec(" + repr(widgets[0]) + ")\n"
        "assert len(previous_button._click_handlers.callbacks) == 0\n"
        "assert len(predict_button._click_handlers.callbacks) == 1\n"
        "calls_before_clicks = engine.inference_calls\n"
        "for _ in range(2):\n"
        "    predict_button.click()\n"
        "    assert last_callback_error is None, last_callback_error\n"
        "    assert not predict_button.disabled\n"
        "    assert sum('image/png' in item.get('data', {}) for item in prediction_output.outputs) == 1\n"
        "assert engine.inference_calls == calls_before_clicks + 2\n"
        "image_item = next(item for item in prediction_output.outputs if 'image/png' in item.get('data', {}))\n"
        "(OUTPUT / 'widget_single_output.png').write_bytes(base64.b64decode(image_item['data']['image/png']))\n"
        "prompt_input.value = ' '\n"
        "predict_button.click()\n"
        "assert engine.inference_calls == calls_before_clicks + 2\n"
        "assert not any('image/png' in item.get('data', {}) for item in prediction_output.outputs)\n"
        "assert plt.get_fignums() == []\n"
        "assert id(engine.model) == model_identity\n"
        "(OUTPUT / 'widget_verification.json').write_text(json.dumps({\n"
        "    'two_clicks_two_predictions': True, 'one_image_per_click': True,\n"
        "    'old_callback_disconnected': True, 'blank_prompt_clears_image': True,\n"
        "    'one_resident_model': True, 'no_open_figures': True}, indent=2), encoding='utf-8')\n"
        "print('Widget verification passed: two clicks, one prediction image per click.')\n"))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    notebook.metadata.pop("widgets", None)
    for name in ("cache/ipython", "cache/jupyter"):
        (output / name).mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PHASE2_FULL_VALIDATION="0", PYTHONUNBUFFERED="1",
                       PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                       IPYTHONDIR=str(output / "cache/ipython"),
                       JUPYTER_RUNTIME_DIR=str(output / "cache/jupyter"))
    manager = KernelManager(kernel_name="python3",
                            connection_file=str(output / "cache/jupyter/kernel.json"))
    manager.kernel_spec.argv = [sys.executable, "-X", "faulthandler", "-m",
                                "ipykernel_launcher", "-f", "{connection_file}"]

    class LoggingClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            result = super().process_message(msg, cell, cell_index)
            if msg["msg_type"] == "stream":
                print(msg["content"]["text"], end="", flush=True)
            return result

    client = LoggingClient(notebook, km=manager, timeout=1200, allow_errors=False,
                           store_widget_state=False, resources={"metadata": {"path": str(ROOT)}})
    report = {"status": "running", "original_notebook_sha256": original_sha,
              "full_validation_recomputed": False, "training_performed": False,
              "completed_code_cells": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        with (output / "kernel_stderr.log").open("w", encoding="utf-8") as stderr:
            with client.setup_kernel(cleanup_kc=True, cwd=str(ROOT), env=environment, stderr=stderr):
                for index, cell in enumerate(notebook.cells):
                    print(f"CELL {index + 1}/{len(notebook.cells)} {cell.cell_type}", flush=True)
                    client.execute_cell(cell, index)
                    if cell.cell_type == "code":
                        report["completed_code_cells"].append(index)
                    nbformat.write(notebook, output / TARGET.name)
        nbformat.validate(notebook)
        counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == "code"]
        assert counts == list(range(1, len(counts) + 1)), counts
        report.update(status="passed", execution_counts=counts)
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        report["original_notebook_unchanged"] = sha256(TARGET) == original_sha
        if not report["original_notebook_unchanged"]:
            report["status"] = "failed"
        nbformat.write(notebook, output / TARGET.name)
        (output / "execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("EXECUTION STATUS:", report["status"], "| artifacts:", output, flush=True)
    if report["status"] != "passed":
        raise RuntimeError("Notebook preservation or execution check failed")


if __name__ == "__main__":
    main()
