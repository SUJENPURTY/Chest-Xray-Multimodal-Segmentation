"""Execute notebook 07 only, using this exact interpreter; save every completed cell.

Run with .venv/Scripts/python.exe. Set PHASE2_FULL_VALIDATION=1 to force a new
streaming evaluation; the default verifies/reuses an existing matching run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/07_phase2_live_inference.ipynb"
OUTPUT = ROOT / "outputs/run/phase2_live_inference"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                       IPYTHONDIR=str(OUTPUT / "cache/ipython"),
                       JUPYTER_RUNTIME_DIR=str(OUTPUT / "cache/jupyter"),
                       PYTHONDONTWRITEBYTECODE="1")
    notebook = nbformat.read(TARGET, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    notebook.metadata.pop("widgets", None)
    manager = KernelManager(kernel_name="xray-e-drive")
    # Runtime-only override: no user kernelspec installation or mutation.
    manager.kernel_spec.argv = [sys.executable, "-X", "faulthandler", "-m",
                                "ipykernel_launcher", "-f", "{connection_file}"]

    class LoggingClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            result = super().process_message(msg, cell, cell_index)
            if msg["msg_type"] == "stream":
                print(msg["content"]["text"], end="", flush=True)
            return result

    client = LoggingClient(notebook, km=manager, timeout=1200, allow_errors=False,
                           resources={"metadata": {"path": str(ROOT)}})
    report = {"notebook": str(TARGET), "interpreter": sys.executable,
              "started_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
              "completed_code_cells": [], "status": "running"}
    (OUTPUT / "execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        with (OUTPUT / "kernel_stderr.log").open("w", encoding="utf-8") as stderr:
            with client.setup_kernel(cleanup_kc=True, cwd=str(ROOT), env=environment, stderr=stderr):
                for index, cell in enumerate(notebook.cells):
                    print(f"CELL {index + 1}/{len(notebook.cells)} {cell.cell_type}", flush=True)
                    client.execute_cell(cell, index)
                    if cell.cell_type == "code":
                        report["completed_code_cells"].append(index)
                    nbformat.write(notebook, TARGET)
        nbformat.validate(notebook)
        assert not any(output.output_type == "error" for cell in notebook.cells
                       if cell.cell_type == "code" for output in cell.outputs)
        counts = [cell.execution_count for cell in notebook.cells if cell.cell_type == "code"]
        assert counts == list(range(1, len(counts) + 1)), counts
        report.update(status="passed", execution_counts=counts)
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        # Preserve widget state after all cells have finished.
        client.set_widgets_metadata()
        nbformat.write(notebook, TARGET)
        (OUTPUT / "execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("EXECUTION STATUS:", report["status"], flush=True)


if __name__ == "__main__":
    main()
