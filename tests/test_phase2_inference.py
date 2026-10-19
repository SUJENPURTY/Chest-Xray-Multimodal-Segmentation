"""Independent metric, geometry and no-gradient checks; no pretrained downloads."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.phase2 import (
    Phase2Inference, confusion_counts, metrics_from_counts, summarize_rows,
)
from src.inference.predictor import render_prediction


def test_confusion_and_hand_calculated_metrics():
    prediction = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
    target = np.array([[1, 0, 1], [1, 0, 0]], dtype=bool)
    counts = confusion_counts(prediction, target)
    assert counts == dict(tp=1, fp=1, fn=2, tn=2)
    assert metrics_from_counts(**counts) == pytest.approx(dict(
        pixel_accuracy=0.5, dice=0.4, iou=0.25, precision=0.5, recall=1/3))


@pytest.mark.parametrize("counts,expected", [
    ((0, 0, 0, 8), (1, 1, 1, 0, 0)),
    ((0, 0, 8, 0), (0, 0, 0, 0, 0)),
    ((0, 8, 0, 0), (0, 0, 0, 0, 0)),
    ((8, 0, 0, 0), (1, 1, 1, 1, 1)),
])
def test_empty_full_and_disjoint_masks(counts, expected):
    assert tuple(metrics_from_counts(*counts).values()) == expected


def test_macro_global_are_distinct_and_pixel_weighted():
    rows = [dict(tp=1, fp=0, fn=0, tn=0), dict(tp=0, fp=0, fn=9, tn=0)]
    summary = summarize_rows(rows)
    assert summary["image_mean"]["pixel_accuracy"] == 0.5
    assert summary["dataset_global"]["pixel_accuracy"] == 0.1
    assert summary["image_mean"]["dice"] == 0.5
    assert summary["dataset_global"]["dice"] == pytest.approx(2/11)
    assert summary["counts"] == dict(tp=1, fp=0, fn=9, tn=0)


def test_metric_validation():
    for values in [(-1, 0, 0, 1), (float("nan"), 0, 0, 1), (0.5, 0, 0, 1), (0, 0, 0, 0)]:
        with pytest.raises(ValueError):
            metrics_from_counts(*values)
    with pytest.raises(ValueError):
        confusion_counts(np.zeros((2, 2), bool), np.zeros((2, 3), bool))
    with pytest.raises(TypeError):
        confusion_counts(np.zeros((2, 2)), np.zeros((2, 2), bool))


def test_native_display_uses_transpose_and_threshold_after_resize():
    probability = np.zeros((224, 224), np.float32)
    probability[10:20, 30:40] = 1
    result = render_prediction(Image.new("RGB", (224, 224)), probability)
    np.testing.assert_array_equal(np.asarray(result["binary_mask"]) > 0, probability.T >= 0.5)
    assert np.asarray(result["binary_mask"])[35, 15] == 255
    assert np.asarray(result["binary_mask"])[15, 35] == 0


def test_inference_is_fp32_no_gradient_and_returns_independent_cpu_copy():
    class Model(torch.nn.Module):
        def forward(self, inputs):
            image, tokens = inputs
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            assert image.dtype == torch.float32 and image.shape == (1, 1, 224, 224)
            assert tokens["input_ids"].shape == (1, 24)
            return image.sigmoid()

    engine = Phase2Inference.__new__(Phase2Inference)
    engine.cfg = SimpleNamespace(image_size=224)
    engine.device, engine.model, engine.inference_calls = torch.device("cpu"), Model(), 0
    image = torch.zeros(1, 224, 224, requires_grad=True)
    tokens = {k: torch.ones(24, dtype=torch.long) for k in ("input_ids", "attention_mask")}
    result = engine.infer_tensor(image, tokens)
    assert engine.inference_calls == 1 and image.grad is None
    assert result.dtype == np.float32 and result.shape == (224, 224)
    np.testing.assert_array_equal(result, 0.5)
    result[:] = 0
    np.testing.assert_array_equal(engine.infer_tensor(image, tokens), 0.5)
