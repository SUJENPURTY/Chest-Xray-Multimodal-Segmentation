# Phase 1: isolated numerical and segmentation diagnostics

This directory contains new experiment code and outputs only. The six notebooks,
original source, checkpoints, histories and evaluation files are read-only inputs.
SHA256 manifests record preservation before and after the run.

Run from `E:\x-ray` using the project environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.\.venv\Scripts\python.exe -B -u experiments\phase1_diagnostics\run_phase1.py
```

The runner refuses to overwrite an existing experiment manifest. Do not rerun it
in this directory. Use a separate versioned directory for a new experiment.

## Protocol fixed before execution

- Parent: original epoch-18 `best_dice.ckpt`, strict loading, ordinary parameter copying.
- Data: original development CSV, canonical filename-group split, seed 2026.
- Validation: 1,423 images, FP32, no augmentation, microbatch 1.
- Threshold grid: 0.05–0.95, step 0.025. Select by macro validation Dice;
  ties choose the threshold closest to 0.5. The baseline 0.5 remains reported.
- No test images or test ground-truth masks are loaded. Previous test metrics
  are quoted only as a historical reference and are not selection inputs.
- Matched diagnostics: original/stable loss in FP32, then AMP with scales 65536
  and 1. Restore checkpoint, BatchNorm buffers and random seed between conditions.
- Stable loss: BCEWithLogits plus soft Dice from FP32 sigmoid probabilities.
- Eight-image overfit: seed-selected training images from distinct filename-derived
  groups, no augmentation, FP32, microbatch 1, accumulation 8. Frozen BERT is kept
  in evaluation mode during this test; decoder BatchNorm uses training mode.
- Fresh AdamW: ConvNeXt LR 3e-5, remaining trainable layers LR 3e-4, weight decay
  0.01, gradient L2 clipping 1, foreach disabled. No scheduler during this short test.
- Limit: 2,000 microsteps / 250 optimizer updates or 30 minutes. Stop earlier on
  numerical failure, memory failure, plateau, or two checks with mean Dice >=0.90
  and minimum per-image Dice >=0.75. Evaluate the training subset every 80 steps.

## Interpretation

Validation threshold gains estimate calibration on that validation set and may be
optimistic because the threshold was selected there. They are not new test scores.
Training-subset gains establish learnability only. They do not measure generalization
or reproduce Ariadne's Thread. No full-data training run is authorized by this script.

Pixel metrics are foreground Dice `2TP/(2TP+FP+FN)`, IoU `TP/(TP+FP+FN)`,
precision `TP/(TP+FP)` and recall `TP/(TP+FN)`. Macro scores average per-image
ratios. Global scores aggregate counts before division. Undefined precision for an
empty prediction is reported as zero; all original targets are nonempty.

Probability arrays retain MONAI's tensor orientation. Saved visualizations transpose
all masks/probabilities back to the original PIL image orientation.

`run.log` is the execution log. See `REPORT.md`, `phase1_summary.json`, the
`diagnostics`, `validation` and `overfit` subdirectories, and
`preservation_verification.json` for actual results after execution.

## Actual stop status

The first run stopped during the eight-image backward pass with a CUDA OOM after
the last persisted check at 1,120 microsteps / 140 optimizer updates. Training was
not restarted. Numerical comparisons, all validation probabilities, the threshold
sweep, initial subset visualizations and persisted training curves are available.
The post-training checkpoint and masks were not exported before the failure and
are unavailable. The queued AMP module trace refused to allocate a second model
after the failure. Do not run further GPU work in this attempt.

`recover_partial_results.py` consolidates persisted measurements on CPU without
reconstructing or fabricating the missing model. `verify_results.py` independently
checks validation counts and historical consistency. `REPORT.md` identifies the
measured gains, OOM, incomplete sanity criterion and next approval boundary.
