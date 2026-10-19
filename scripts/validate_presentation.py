"""Validate remaining presentation repository without training/full evaluation."""
import ast
import gc
import importlib
import json
from pathlib import Path
import runpy
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat
import pytest
import torch
from streamlit.testing.v1 import AppTest
from src.inference.streamlit_inference import DemoPredictor


def main():
    imported = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        name = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name.removesuffix(".__init__")
        importlib.import_module(name)
        imported.append(name)
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 6
    for path in notebooks:
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type != "code":
                continue
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        assert alias.name == "*" or hasattr(module, alias.name), (path.name, node.module, alias.name)
    checkpoint = ROOT / "outputs/run/checkpoints/best_dice.ckpt"
    assert checkpoint.is_file()
    predictor = DemoPredictor(checkpoint)
    assert predictor.model.__class__.__name__ == "LanGuideMedSeg" and not predictor.model.training
    assert predictor.epoch == 18
    assert all(p.dtype == torch.float32 for p in predictor.model.parameters())
    device = str(predictor.device)
    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=90)
    assert not app.exception, app.exception
    runpy.run_path(str(ROOT / "scripts/validate_streamlit_ui.py"), run_name="__main__")
    for relative in ("outputs/run/metrics/test_metrics.json", "outputs/run/evaluation/test_metrics_summary.json",
                     "outputs/run/inference/final_inference_summary.csv"):
        assert (ROOT / relative).is_file()
    temporary = ROOT / ".cache" / ("presentation-pytest-" + uuid.uuid4().hex)
    assert temporary.resolve().is_relative_to(ROOT.resolve()) and not temporary.exists()
    test_status = pytest.main([str(ROOT / "tests"), "-q", "-p", "no:cacheprovider", "--basetemp", str(temporary)])
    report = {"src_imports": imported, "canonical_notebooks": [p.name for p in notebooks],
              "notebook_imports_resolve": True, "nbformat_valid": True,
              "model_creation_and_strict_checkpoint_load": True, "checkpoint": str(checkpoint),
              "epoch": 18, "fp32": True, "device": device, "app_import": True,
              "result_ui_and_threshold_reruns": True, "final_outputs_exist": True,
              "pytest_exit_code": int(test_status), "training_or_full_evaluation_run": False}
    (ROOT / "docs/presentation_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
