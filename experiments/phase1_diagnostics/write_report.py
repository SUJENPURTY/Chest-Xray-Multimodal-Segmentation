"""Render measured Phase 1 results into a local Markdown report and plots."""
import csv
import json
from pathlib import Path
import os
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / 'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def row(score, label):
    return f"| {label} | {score['threshold']:.3f} | {score['macro_dice']:.6f} | {score['macro_iou']:.6f} | {score['macro_precision']:.6f} | {score['macro_recall']:.6f} |"


def main():
    s = load('phase1_summary.json')
    verification = load('independent_verification.json')
    preservation = load('preservation_verification.json')
    checkpoint = load('checkpoint_verification.json')
    baseline, chosen, subset = s['validation_baseline'], s['validation_calibration']['selected'], s['overfit']
    trace = load('diagnostics/amp_localization.json') if (HERE / 'diagnostics/amp_localization.json').exists() else []
    diag_table = '\n'.join(f"| {r['condition']} | {r['loss']:.9f} | {r['nonfinite_gradient_tensors']} / {r['gradient_tensors']} | {r['finite_gradient_l2_norm']:.6f} | {r['optimizer_step_applied']} | {r['peak_allocated_mib']:.1f} |" for r in s['diagnostics'])
    trace_table = '\n'.join(f"| {r['condition']} | {r['nonfinite_gradient_tensors']} | {r['first_nonfinite_tensor_hook']['module'] if r['first_nonfinite_tensor_hook'] else 'None'} | {r['peak_allocated_mib']:.1f} |" for r in trace)
    trace_section = ('| Trace condition | Non-finite parameter gradient tensors | First non-finite output hook | Peak allocated MiB |\n'
                     '|---|---:|---|---:|\n' + trace_table) if trace else ''
    history = list(csv.DictReader((HERE / 'overfit/evaluation_progression.csv').open(newline='')))
    last = history[-1]
    if (HERE / 'overfit/best_per_image.csv').exists():
        with (HERE / 'overfit/best_per_image.csv').open(newline='') as f:
            sample_rows = list(csv.DictReader(f))
        sample_table = '\n'.join(f"| {r['image']} | {float(r['true_foreground_fraction']) * 100:.3f}% | {float(r['dice']):.6f} | {float(r['iou']):.6f} |" for r in sample_rows)
    else:
        sample_table = '| Post-training per-image values were not persisted before OOM | Unavailable | Unavailable | Unavailable |'
    sample_section = ('| Training image at the best check | Ground-truth area | Dice | IoU |\n'
                      '|---|---:|---:|---:|\n' + sample_table) if (HERE / 'overfit/best_per_image.csv').exists() else ''
    next_gate = ('The subset sanity criterion passed; a separately approved short full-development pilot is justified.'
                 if subset['success_criterion_met'] else
                 'The subset sanity criterion did not pass and a CUDA OOM stopped the run. Resolve resource handling first; a new approved bounded attempt must save checkpoints/results at every check and monitor host RAM/commit. Then inspect the weakest training image and compare BatchNorm/loss handling before a full-development pilot. Do not equate the high subset mean with success on every sample.')
    p = load('validation/probability_distribution.json')
    controls = load('validation/constant_mask_controls.json')
    preserved = preservation['checked_files']
    gpu = load('gpu_monitor_summary.json')
    host = load('host_memory_after_failure.json') if (HERE / 'host_memory_after_failure.json').exists() else None
    partial = not subset.get('post_training_checkpoint_saved', True)
    trace_status = ('The queued module-level AMP trace was **not executed**: its guard refused a second GPU model after the primary OOM. Module/backend localization remains unresolved.' if not trace else 'The following no-update backward traces were executed:')
    export_status = ('Post-training checkpoint, masks and per-image rows were **not saved**. The harness held the best full-model snapshot in CPU RAM and exported it only on normal completion; the OOM lost that snapshot. This is a limitation of the new diagnostic harness, not an original-project bug. Initial images/masks/overlays and all persisted curves remain available. No substitute checkpoint or masks were fabricated.' if partial else 'The new checkpoint `overfit/best_subset.ckpt` and best/final predictions are saved; they are diagnostic training-subset outputs only.')
    host_text = (f"Post-failure host RAM was {host['load_percent']}% used, with {host['available_physical_bytes'] / 2**20:.0f} MiB available out of {host['total_physical_bytes'] / 2**30:.2f} GiB. This was measured **after** the failure and cannot prove its cause. Full-model CPU snapshots, Windows WDDM backing/commit pressure, transient CUDA workspace and allocator/driver behaviour are hypotheses; no single cause is established." if host else '')
    text = f"""# Phase 1 controlled diagnostics — measured results

## Outcome and limits

The principal diagnostics and validation calibration completed; the eight-image test **stopped early on a CUDA out-of-memory error**. No training restart, full-data training run or new test inference occurred. All **{preserved} protected original files** have unchanged SHA256 hashes, including the six notebooks, both checkpoints, source, histories and evaluation outputs. Independent verification of persisted measurements passed. Missing post-training outputs are explicitly identified below.

The stable FP32 loss produced finite gradients and updates. Its matched loss equals the original FP32 loss on the diagnostic sample. Changing the loss alone did not resolve AMP instability: four AMP comparisons had 256 non-finite gradient tensors and their updates were skipped.

Validation threshold calibration changed mean Dice from **{baseline['macro_dice']:.6f}** at 0.5 to **{chosen['macro_dice']:.6f}** at **{chosen['threshold']:.3f}**. This is a validation-selected result, not a new test result. It is potentially optimistic because the same validation set selects and measures the threshold.

The eight-image training-subset test changed Dice from **{subset['initial']['macro_dice']:.6f}** to a best of **{subset['best']['macro_dice']:.6f}** after **{subset['best_microstep']} microsteps**. This tests learnability only. Generalization and reproduction of the paper's Dice 0.8978 have not been established.

## Original baseline

Original saved test metrics, quoted only as a historical reference: mean Dice **0.185754449**, mean IoU **0.106599931**, epoch-18 checkpoint, threshold 0.5. No test images or test masks were opened in this experiment.

Parent checkpoint SHA256: `{checkpoint['sha256']}`. Strict loading matched **{checkpoint['exactly_matching_state_tensors']} state tensors** exactly, with no non-finite weights. The model has {checkpoint['parameters']:,} parameters and {checkpoint['trainable_parameters']:,} parameters marked trainable. All modules were in evaluation mode for validation. Repeated forwards and legacy-versus-logit sigmoid outputs were identical on three validation examples.

The newly measured validation baseline agrees with the original validation workbook: maximum per-image Dice difference **{verification['historical_validation_max_per_image_dice_difference']:.3g}**, IoU difference **{verification['historical_validation_max_per_image_iou_difference']:.3g}**; no predicted/target foreground count mismatches. Historical AMP epoch-selection Dice 0.196610 is distinct from the FP32 checkpoint re-evaluation reported here.

## Numerical stability

| Condition | Loss | Non-finite gradient tensors | Finite-gradient L2 norm | Update applied | Peak allocated MiB |
|---|---:|---:|---:|---|---:|
{diag_table}

All conditions restore identical epoch-18 weights, BatchNorm buffers and seed. Original training-mode dropout and BatchNorm are retained for this comparison. AdamW uses LR 3e-4, weight decay 0.01, microbatch 1, foreach disabled. Gradient norms are measured after unscaling; for unstable runs the norm includes finite tensors only and must not be interpreted as the full norm. Logit/probability hook gradients in JSON are recorded before unscaling.

Stable loss: BCEWithLogits on FP32 logits plus soft Dice from FP32 sigmoid probabilities. The original and stable FP32 watched gradients differed by only {s['diagnostics'][1]['fp32_gradient_max_abs_difference']['out']:.3g} in the output weights and {s['diagnostics'][1]['fp32_gradient_max_abs_difference']['encoder']:.3g} in the watched encoder weights. This verifies compatibility on this real sample, not universal equivalence for extreme/clamped probabilities.

AMP logit gradients remained finite in the matched comparisons, including scale 1. The observed overflow therefore cannot be attributed solely to the BCE-to-sigmoid boundary for this sample. Non-finite gradients include the final decoder transpose-convolution and preceding encoder/attention parameters, while downstream final-convolution/head parameters remain finite. This narrows the boundary but does not identify the first failing operation.

{trace_status}

{trace_section}

Targeted FP32 islands and a cuDNN comparison are prepared in `trace_amp.py`, but no unexecuted intervention is claimed as a fix or adopted for longer training.

## Validation baseline and threshold calibration

Population: the canonical 1,423 validation images; development split seed 2026; no augmentation; FP32; microbatch 1; foreground metrics. Grid fixed before execution: 0.05–0.95 in steps of 0.025. Selection criterion: macro per-image validation Dice; ties choose closest to 0.5.

| Validation result | Threshold | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|
{row(baseline, 'Baseline')}
{row(chosen, 'Selected threshold')}

| Foreground analysis | Baseline 0.5 | Selected threshold |
|---|---:|---:|
| Mean predicted area | {baseline['macro_predicted_foreground_fraction'] * 100:.3f}% | {chosen['macro_predicted_foreground_fraction'] * 100:.3f}% |
| Mean ground-truth area | {baseline['macro_true_foreground_fraction'] * 100:.3f}% | {chosen['macro_true_foreground_fraction'] * 100:.3f}% |
| True-positive pixels | {baseline['tp']:,} | {chosen['tp']:,} |
| False-positive pixels | {baseline['fp']:,} | {chosen['fp']:,} |
| False-negative pixels | {baseline['fn']:,} | {chosen['fn']:,} |
| Global Dice | {baseline['global_dice']:.6f} | {chosen['global_dice']:.6f} |
| Global IoU | {baseline['global_iou']:.6f} | {chosen['global_iou']:.6f} |

The selected threshold raises recall but lowers precision. False-positive pixels increase by **{chosen['fp'] / baseline['fp']:.2f}x**, and mean predicted area is **{chosen['macro_predicted_foreground_fraction'] / chosen['macro_true_foreground_fraction']:.2f}x** the ground-truth area. The Dice gain is real on validation but does not imply accurate mask boundaries or adequate localization.

All-pixel probabilities: minimum {p['minimum']:.6g}, maximum {p['maximum']:.6g}, mean {p['mean']:.6f}. Class-conditional histograms show whether foreground and background probabilities separate. Precision is zero for an empty prediction; masks are nonempty. Counts and macro/global scores were independently recomputed from saved FP32 arrays.

For context, an all-foreground constant mask scores validation macro Dice **{controls['all_foreground_macro_dice']:.6f}** and IoU **{controls['all_foreground_macro_iou']:.6f}**. An all-background prediction scores Dice/IoU zero but pixel accuracy **{controls['all_background_pixel_accuracy']:.6f}**. These controls highlight why pixel accuracy or modest overlap alone cannot establish good localization. They are not candidates in threshold selection.

![Validation probabilities](validation/probability_distribution.png)

![Threshold comparison](validation/threshold_comparison.png)

Complete threshold table: `validation/threshold_comparison.csv`. Baseline per-image results: `validation/per_image_threshold_0.5.csv`. Probability arrays and targets are saved with MONAI tensor orientation; visualizations restore original PIL orientation.

## Eight-image overfitting sanity test

Samples are seeded random training images from distinct filename-derived groups; none are validation images. Augmentation is disabled. FP32, microbatch 1, accumulation 8, fresh AdamW; ConvNeXt LR 3e-5, other trainable layers LR 3e-4, weight decay 0.01, L2 clipping 1. Frozen BERT is explicitly in eval mode during training; decoder BatchNorm remains in training mode. This deterministic-text behaviour is a documented variant from upstream training dropout. No scheduler is used in this bounded test.

| Training-subset result at threshold 0.5 | Dice | IoU |
|---|---:|---:|
| Initial checkpoint | {subset['initial']['macro_dice']:.6f} | {subset['initial']['macro_iou']:.6f} |
| Best measured check | {subset['best']['macro_dice']:.6f} | {subset['best']['macro_iou']:.6f} |
| Last persisted check | {subset['final']['macro_dice']:.6f} | {subset['final']['macro_iou']:.6f} |

The last persisted check records {subset['completed_microsteps']} microsteps and {subset['optimizer_updates']} optimizer updates. Additional unlogged steps before the failure are unknown. Stop reason: **{subset['stop_reason']}**. Proposed success criterion (mean Dice >=0.90 and minimum image Dice >=0.75 twice): **{subset['success_criterion_met']}**. Last checked mean loss {float(last['mean_loss']):.6f}, minimum per-image Dice {float(last['min_dice']):.6f}. No full-data epochs were run.

{export_status}

{sample_section}

![Training progression](overfit/training_progression.png)

Initial original images, ground truth, predicted masks, probability arrays and overlays are saved under `overfit/initial`. GT is green and predictions red; both are in original image orientation. Post-training `best`/`final` images are unavailable after the OOM. See `overfit/CHECKPOINT_NOT_SAVED.md`.

## Measured resources

GPU: GTX 1650, 4,095.7 MiB reported VRAM. One model is allocated at a time for the principal experiments. AdamW foreach is disabled. Baseline validation took **{baseline['seconds']:.1f} seconds** for 1,423 images, including data loading, CPU export and checks.

The last persisted overfit check occurred at **{subset['seconds']:.1f} seconds**, averaging **{subset['seconds_per_microstep_including_checks']:.3f} seconds per microstep**, including evaluations and CPU snapshots up to that point. Total failure time was not recorded. Peak PyTorch allocation was **{subset['peak_allocated_mib']:.1f} MiB** and peak reservation **{subset['peak_reserved_mib']:.1f} MiB**. The 5-second external monitor observed a whole-device peak of **{gpu['maximum_device_used_mib']:.0f} MiB**, including desktop and other applications. Sampling may miss brief peaks; PyTorch peaks exclude external/context allocations. The OOM means sustained microbatch-1 safety is **not yet established**, despite the lower observed allocation. No claim is made that physical batch 32 fits.

{host_text}

The queued tracing process initialized ML libraries while waiting, but its guard prevented loading a second GPU model or running a backward trace after the OOM. Its host/context overhead was not separately measured. A future attempt should use one ML process, monitor host physical RAM and commit continuously, serialize checkpoints immediately to disk instead of retaining full CPU snapshots, and preserve outputs in the failure handler.

## Confirmed improvements and remaining problems

- Stable FP32 optimization is verified on the matched backward diagnostic and bounded subset run.
- Threshold calibration has a measured validation-only Dice change of **{chosen['macro_dice'] - baseline['macro_dice']:+.6f}**. It does not establish a test-set improvement.
- Training-subset learnability changed by **{subset['best']['macro_dice'] - subset['initial']['macro_dice']:+.6f} Dice**. It does not establish generalization.
- AMP remains unsuitable without further validated handling; logits loss alone did not fix it.
- A CUDA OOM stopped the subset run; no new post-training checkpoint or prediction exports survived. Further GPU experiments were stopped.
- The original low segmentation performance and published-score gap remain unresolved until a controlled full-development pilot is evaluated.
- Microbatch BatchNorm, the scheduler/physical-batch difference, incomplete patient metadata and version-dependent upstream loss conventions remain material issues.

## Recommended next configuration — proposal only

**Next decision gate:** {next_gate}

After the decision gate passes, retain the original epoch-18 checkpoint and locked validation split as the starting reference. Use a new run directory for a separately approved **2–5 epoch pilot**, FP32, microbatch 1, accumulation 8, fresh AdamW with separate vision/decoder learning rates, explicit gradient/update monitoring and measured memory. Use the same stable logits loss, and keep threshold-0.5 validation results primary alongside calibrated results. Re-select calibration on validation if weights change; never carry a threshold from these old weights into a new model without checking validation.

Start from the measured subset configuration (vision 3e-5, decoder 3e-4) as a hypothesis, not a validated full-data optimum. Compare BatchNorm handling separately; accumulation does not recreate physical-batch-32 statistics. Use a scheduler whose horizon is explicitly longer than the short pilot rather than exhausting the LR at the pilot endpoint. Do not copy the eight-image model into the main project. Keep AMP disabled until multi-sample backward parity and finite gradients are established. Longer training needs approval and evidence of validation improvement.

## Reproducibility and preservation

Code: `run_phase1.py`, `trace_amp.py` (prepared, not executed past its safety gate), `recover_partial_results.py` (CPU-only consolidation), `verify_results.py`, `write_report.py`, `monitor_gpu.py`. Protocol: `README.md`, `protocol.json`, `overfit/config.json`. Logs: `run.log`, `amp_trace.log`, `gpu_monitor.csv`. Measured summaries: `phase1_summary.json`, `independent_verification.json`, `failure.json`. Hash records: `preservation_before.json`, `preservation_verification.json`.

**Stop boundary:** Phase 1 ends here. No longer training, new test evaluation, notebook changes or PowerPoint generation has started.
"""
    (HERE / 'REPORT.md').write_text(text, encoding='utf-8')
    with (HERE / 'gpu_monitor.csv').open(newline='') as f:
        gpu_rows = list(csv.DictReader(f))
    plt.figure(figsize=(9, 4))
    plt.plot([float(r['device_used_mib']) for r in gpu_rows], label='Whole device VRAM used')
    plt.axhline(4096, color='red', linestyle='--', label='4GB device capacity')
    plt.xlabel('5-second monitoring sample')
    plt.ylabel('MiB')
    plt.title('Phase 1 GPU usage, including desktop applications')
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / 'gpu_memory_usage.png', dpi=160)
    plt.close()
    plt.figure(figsize=(9, 5))
    plt.plot([int(r['microsteps']) for r in history], [float(r['macro_dice']) for r in history], marker='o', label='Subset mean Dice')
    plt.plot([int(r['microsteps']) for r in history], [float(r['min_dice']) for r in history], marker='.', label='Minimum image Dice')
    plt.axhline(0.9, color='grey', linestyle='--', label='Proposed mean Dice sanity criterion')
    plt.xlabel('Persisted microsteps (8 per optimizer update)')
    plt.ylabel('Eight-image scores at threshold 0.5')
    plt.title('Training-subset progression; stopped on CUDA OOM')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / 'overfit/training_progression.png', dpi=160)
    plt.close()
    print('REPORT.md written from measured experiment outputs')


if __name__ == '__main__':
    main()
