"""Fast checks that need neither QaTa data nor pretrained model downloads."""
import csv
from io import BytesIO
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from src.data.dataset import QaTaDataset, group_key, read_manifest
from src.inference.predictor import LanGuidePredictor, load_checkpoint, render_prediction, png_bytes
from src.config import Config, from_environment, with_output_dir
from src.training.losses import DiceBCELoss
from src.training.metrics import binary_metrics
from src.training.trainer import _check_batch, _load_resume_payload, _save


class DummyTokenizer:
    def __call__(self, prompt, **kwargs):
        assert prompt
        return {"input_ids": torch.ones(1, 24, dtype=torch.long),
                "attention_mask": torch.ones(1, 24, dtype=torch.long)}


def test_empty_and_perfect_metrics():
    empty = torch.zeros(2, 1, 8, 8)
    assert binary_metrics(empty, empty)["dice"] == pytest.approx(1)
    assert binary_metrics(empty, empty)["iou"] == pytest.approx(1)
    target = empty.clone()
    target[:, :, :4] = 1
    assert binary_metrics(target, target)["dice"] == pytest.approx(1)
    assert binary_metrics(target, target)["iou"] == pytest.approx(1)
    wrong = binary_metrics(empty, target)
    assert wrong["dice"] < 0.01 and wrong["iou"] < 0.01


def test_loss_is_finite():
    target = torch.zeros(1, 1, 8, 8)
    probability = torch.full_like(target, 0.5, requires_grad=True)
    loss = DiceBCELoss()(probability, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(probability.grad).all()
    empty_loss = DiceBCELoss()(torch.zeros_like(target), target)
    wrong_loss = DiceBCELoss()(torch.ones_like(target), target)
    assert empty_loss < 0.001
    assert wrong_loss > empty_loss


def test_training_tensor_validation():
    image = torch.zeros(1, 1, 8, 8)
    target = torch.zeros_like(image)
    _check_batch(image, target, torch.full_like(image, 0.5))
    with pytest.raises(FloatingPointError):
        _check_batch(image.fill_(float("nan")), target)
    image.zero_()
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        DiceBCELoss()(torch.full_like(target, 0.5), target.fill_(2))
    target.zero_()
    with pytest.raises(TypeError, match="floating point"):
        DiceBCELoss()(torch.full_like(target, 0.5), target.long())
    with pytest.raises(ValueError, match="shapes differ"):
        _check_batch(image, target, torch.zeros(1, 1, 4, 4))


def test_output_paths_follow_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("TRAIN_CSV", raising=False)
    monkeypatch.delenv("TEST_CSV", raising=False)
    monkeypatch.delenv("MODEL_CHECKPOINT", raising=False)
    cfg = from_environment()
    assert cfg.output_dir == tmp_path
    assert cfg.train_csv == tmp_path / "prepared_annotations" / "train.csv"
    assert cfg.test_csv == tmp_path / "prepared_annotations" / "test.csv"
    assert cfg.checkpoint_path == tmp_path / "checkpoints" / "best_dice.ckpt"
    assert with_output_dir(Config(), tmp_path).train_csv == cfg.train_csv


def test_resume_checkpoint_validation_and_atomic_save(tmp_path):
    path = tmp_path / "latest.ckpt"
    with pytest.raises(FileNotFoundError, match="Resume checkpoint missing"):
        _load_resume_payload(path)
    _save({"kind": "wrong"}, path)
    with pytest.raises(ValueError, match="Unsupported or incomplete"):
        _load_resume_payload(path)
    assert not (tmp_path / "latest.ckpt.tmp").exists()


def test_dataset_pairs_image_mask_text(tmp_path):
    for folder in ("Train/Images", "Train/GTs", "Test/Images", "Test/GTs"):
        (tmp_path / folder).mkdir(parents=True)
    image = Image.new("L", (16, 16), 100)
    mask = Image.new("L", (16, 16), 0)
    mask.putpixel((3, 4), 255)
    for index in range(5):
        image.save(tmp_path / "Train" / "Images" / f"sample{index}.png")
        mask.save(tmp_path / "Train" / "GTs" / f"mask_sample{index}.png")
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Image", "Description"])
        writer.writeheader()
        for index in range(5):
            writer.writerow({"Image": f"mask_sample{index}.png", "Description": f"location {index}"})
    assert len(read_manifest(csv_path, tmp_path, "train")) == 4
    assert len(read_manifest(csv_path, tmp_path, "validation")) == 1
    dataset = QaTaDataset(csv_path, tmp_path, "train", DummyTokenizer(), image_size=32)
    (xray, tokens), target = dataset[0]
    assert xray.shape[-2:] == target.shape[-2:] == (32, 32)
    assert target.unique().tolist() == [0., 1.]
    assert tokens["input_ids"].shape == (24,)
    (tmp_path / "Train" / "GTs" / "mask_sample1.png").unlink()
    with pytest.raises(FileNotFoundError):
        read_manifest(csv_path, tmp_path, "train")


def test_patient_grouped_split(tmp_path):
    for folder in ("Train/Images", "Train/GTs"):
        (tmp_path / folder).mkdir(parents=True)
    path = tmp_path / "train.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Image", "Description"])
        writer.writeheader()
        for patient in range(10):
            for visit in range(2):
                name = f"sub-S{patient:03}_visit{visit}.png"
                Image.new("L", (4, 4)).save(tmp_path / "Train" / "Images" / name)
                Image.new("L", (4, 4)).save(tmp_path / "Train" / "GTs" / ("mask_" + name))
                writer.writerow({"Image": "mask_" + name, "Description": "lower lung"})
    train = read_manifest(path, tmp_path, "train")
    valid = read_manifest(path, tmp_path, "validation")
    assert len(train) + len(valid) == 20
    assert not ({group_key(r["name"]) for r in train} &
                {group_key(r["name"]) for r in valid})


def test_render_prediction_orientation_and_png():
    original = Image.new("RGB", (300, 150), "white")
    probability = np.zeros((224, 224), dtype=np.float32)
    probability[10:20, 30:40] = 1
    result = render_prediction(original, probability)
    assert result["binary_mask"].size == original.size
    assert result["overlay"].size == original.size
    assert png_bytes(result["binary_mask"]).startswith(b"\x89PNG")
    square = render_prediction(Image.new("RGB", (224, 224)), probability)["binary_mask"]
    displayed = np.asarray(square)
    assert displayed[35, 15] == 255 and displayed[15, 35] == 0


def test_checkpoint_load_and_shared_predictor(tmp_path):
    model = torch.nn.Conv2d(1, 1, 1)
    path = tmp_path / "tiny.ckpt"
    torch.save({"kind": "CLOUD_LANGUIDE_BINARY_FIXED_V1",
                "model": {"model." + key: value for key, value in model.state_dict().items()}}, path)
    restored, _ = load_checkpoint(torch.nn.Conv2d(1, 1, 1), path)
    assert torch.equal(restored.weight, model.weight)
    assert restored.weight.device.type == "cpu"
    torch.save({"kind": "wrong", "model": {}}, path)
    with pytest.raises(ValueError, match="Unsupported LanGuide checkpoint format"):
        load_checkpoint(torch.nn.Conv2d(1, 1, 1), path)
    torch.save({"kind": "CLOUD_LANGUIDE_BINARY_FIXED_V1", "model": {}}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(torch.nn.Conv2d(1, 1, 1), path)
    torch.save(["invalid checkpoint"], path)
    with pytest.raises(ValueError, match="Unsupported LanGuide checkpoint format"):
        load_checkpoint(torch.nn.Conv2d(1, 1, 1), path)

    class RecordingTokenizer:
        prompt = None

        def __call__(self, prompt, **kwargs):
            self.prompt = prompt
            return {"input_ids": torch.ones(1, 24, dtype=torch.long),
                    "attention_mask": torch.ones(1, 24, dtype=torch.long)}

    class RecordingModel:
        called = False

        def __call__(self, inputs):
            image, text = inputs
            self.called = image.shape == (1, 1, 224, 224) and text["input_ids"].shape == (1, 24)
            return torch.full((1, 1, 224, 224), 0.75)

    predictor = LanGuidePredictor.__new__(LanGuidePredictor)
    predictor.cfg = Config()
    predictor.device = torch.device("cpu")
    predictor.tokenizer = RecordingTokenizer()
    predictor.model = RecordingModel()
    predictor.transform = lambda _: {"image": torch.zeros(1, 224, 224)}
    output = predictor.predict(Image.new("L", (32, 16), 120), "right lower lung")
    assert predictor.tokenizer.prompt == "right lower lung"
    assert predictor.model.called
    assert output["binary_mask"].size == (32, 16)
    uploaded = png_bytes(Image.new("L", (32, 16), 120))
    assert predictor.predict(uploaded, "left lower lung")["overlay"].size == (32, 16)
    jpeg = BytesIO()
    Image.new("L", (32, 16), 120).save(jpeg, format="JPEG")
    assert predictor.predict(jpeg.getvalue(), "left lower lung")["overlay"].size == (32, 16)
    with pytest.raises(UnidentifiedImageError):
        predictor.predict(b"not an image", "left lower lung")


def test_missing_checkpoint_fails_before_model_download(tmp_path):
    cfg = replace(Config(), checkpoint_path=tmp_path / "absent.ckpt")
    with pytest.raises(FileNotFoundError, match="Model checkpoint missing"):
        LanGuidePredictor(cfg, device="cpu")
