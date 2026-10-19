"""Display verified, already-saved Phase 2 output without model inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PHASE2 = ROOT / "experiments" / "phase2_pilot"
SAMPLE_INDEX = 686
CHECKPOINT = PHASE2 / "checkpoints" / "e02_p05722_u01432_68dcea20.ckpt"
OUTPUT = ROOT / "presentation" / "final_segmentation_output.png"


def load_verified_saved_output():
    """Check provenance and reproduce metrics from saved masks on the CPU."""
    summary = json.loads(
        (PHASE2 / "validation" / "epoch_2" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    csv_path = PHASE2 / summary["per_image_csv"]
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == summary[
        "per_image_csv_sha256"
    ], "Saved validation CSV hash mismatch"
    assert summary["threshold"] == 0.5, "Unexpected prediction threshold"
    best = json.loads((PHASE2 / "checkpoints" / "best.json").read_text())
    assert best["epoch"] == 2, "Best checkpoint is not Epoch 2"
    assert (PHASE2 / best["path"]).resolve() == CHECKPOINT.resolve()
    assert best["sha256"] == summary["checkpoint_sha256"]

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["validation_index"]) == SAMPLE_INDEX
        ]
    assert len(rows) == 1, "Expected exactly one CSV row for sample 686"
    row = rows[0]
    manifest = json.loads((PHASE2 / "data_manifest.json").read_text())
    records = [
        record
        for record in manifest["validation"]
        if record["index"] == SAMPLE_INDEX
    ]
    assert len(records) == 1, "Expected exactly one validation manifest record"
    record = records[0]
    assert record["name"] == row["image"], "Manifest/CSV sample mismatch"

    directory = PHASE2 / summary["prediction_directory"] / "sample_0686"
    filenames = (
        "00_xray.png",
        "00_ground_truth.png",
        "00_prediction.png",
        "00_overlay.png",
    )
    images = []
    for index, filename in enumerate(filenames):
        with Image.open(directory / filename) as opened:
            images.append(np.array(opened.convert("RGB" if index == 3 else "L")))
    assert all(image.shape[:2] == (224, 224) for image in images)
    truth = images[1] > 0
    prediction = images[2] > 0
    probability = np.load(directory / "00_probability.npy", allow_pickle=False)
    assert np.isfinite(probability).all()
    assert np.array_equal(probability >= 0.5, prediction), (
        "Saved mask does not match the existing probability map at threshold 0.5"
    )
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    assert (tp, fp, fn) == tuple(int(row[key]) for key in ("tp", "fp", "fn"))
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    assert abs(dice - float(row["dice"])) < 1e-12
    assert abs(iou - float(row["iou"])) < 1e-12
    return images, record, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the PNG without opening an interactive Matplotlib window.",
    )
    args = parser.parse_args()
    # Guard against accidental replacement on subsequent runs.
    if OUTPUT.exists():
        raise FileExistsError(f"Existing figure preserved: {OUTPUT}")
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt

    images, record, row = load_verified_saved_output()
    fig, axes = plt.subplots(1, 4, figsize=(16, 6.2), facecolor="white")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.82, bottom=0.24, wspace=0.05)
    fig.suptitle(
        "Phase 2 Epoch 2: Validation Sample 686",
        fontsize=22,
        fontweight="bold",
        color="#142E49",
        y=0.945,
    )
    headings = ("Input X-ray", "Ground Truth", "Predicted Mask", "Overlay")
    for index, (axis, image, heading) in enumerate(zip(axes, images, headings)):
        axis.imshow(
            image,
            cmap="gray" if index < 3 else None,
            vmin=0 if index < 3 else None,
            vmax=255 if index < 3 else None,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(heading, fontsize=17, fontweight="bold", color="#142E49", pad=13)
        axis.axis("off")
    fig.text(
        0.5,
        0.184,
        f"Sample: {record['name'].removeprefix('mask_')}",
        ha="center",
        fontsize=11.5,
        color="#4E6170",
    )
    fig.text(
        0.5,
        0.135,
        f"Threshold 0.5   Dice {float(row['dice']):.10f}   IoU {float(row['iou']):.10f}",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#087F8C",
    )
    fig.text(
        0.5, 0.086, f"Text prompt: {record['prompt']}", ha="center", fontsize=11.5
    )
    fig.text(
        0.5,
        0.04,
        "Dataset infection annotation: extensive bilateral involvement. "
        "Overlay: ground truth green, prediction red.",
        ha="center",
        fontsize=11,
        color="#4E6170",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as handle:
        fig.savefig(handle, format="png", dpi=300, facecolor="white")
    with Image.open(OUTPUT) as saved:
        saved.verify()
    print("Sample identifier:", record["name"], flush=True)
    print("Text prompt:", record["prompt"], flush=True)
    print("Dice Score:", row["dice"], flush=True)
    print("IoU Score:", row["iou"], flush=True)
    print("Saved PNG:", OUTPUT, flush=True)
    print("Verified: saved masks, probability threshold, CSV metrics and provenance.", flush=True)
    if args.no_show:
        plt.close(fig)
    else:
        print("Opening Matplotlib window. Close it when finished viewing.", flush=True)
        plt.show()


if __name__ == "__main__":
    main()
