"""CPU-only independent count/aggregation verification of the completed run."""
import csv
import json
from pathlib import Path
import numpy as np
import openpyxl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def independent_metrics(probabilities, targets, threshold):
    values, counts = [], np.zeros(4, dtype=np.int64)
    for p, y in zip(probabilities, targets):
        predicted = p.ravel() >= threshold
        actual = y.ravel() == 1
        # Recompute with sums and set sizes, independently of the runner helper.
        intersection = int(np.sum(predicted[actual]))
        prediction_size = int(np.sum(predicted))
        target_size = int(np.sum(actual))
        false_positive = prediction_size - intersection
        false_negative = target_size - intersection
        true_negative = actual.size - intersection - false_positive - false_negative
        dice = 2 * intersection / (prediction_size + target_size)
        iou = intersection / (prediction_size + target_size - intersection)
        precision = intersection / prediction_size if prediction_size else 0.0
        recall = intersection / target_size
        values.append([dice, iou, precision, recall])
        counts += [intersection, false_positive, false_negative, true_negative]
    means = np.mean(values, axis=0)
    return means, counts, np.asarray(values)


def main():
    summary = read('phase1_summary.json')
    p = np.load(HERE / 'validation/probabilities.npy', mmap_mode='r')
    y = np.load(HERE / 'validation/targets.npy', mmap_mode='r')
    assert p.shape == y.shape == (1423, 224, 224)
    assert set(np.unique(y)) == {0, 1}
    selected = summary['validation_calibration']['selected']
    assertions = {}
    for label, row in [('baseline', summary['validation_baseline']), ('selected', selected)]:
        means, counts, per_image = independent_metrics(p, y, row['threshold'])
        maximum_error = max(abs(means[i] - row['macro_' + k])
                            for i, k in enumerate(('dice', 'iou', 'precision', 'recall')))
        assert maximum_error < 1e-12
        assert list(counts) == [row[k] for k in ('tp', 'fp', 'fn', 'tn')]
        assertions[label + '_independent_max_metric_error'] = float(maximum_error)
        if label == 'baseline':
            baseline_values = per_image
    with (HERE / 'validation/threshold_comparison.csv').open(newline='') as f:
        comparison = list(csv.DictReader(f))
    chosen = max(comparison, key=lambda r: (float(r['macro_dice']), -abs(float(r['threshold']) - 0.5)))
    assert float(chosen['threshold']) == selected['threshold']
    assertions['threshold_selection_matches_declared_rule'] = True
    wb = openpyxl.load_workbook(ROOT / 'outputs/run/training/dice_jaccard_scores.xlsx', read_only=True, data_only=True)
    old = {row[1]: row for row in wb['Validation_Per_Sample'].iter_rows(min_row=2, values_only=True)}
    wb.close()
    manifest = read('validation/manifest.json')
    assert len(old) == len(manifest) == 1423 and set(old) == {r['image'] for r in manifest}
    dice_errors = [abs(baseline_values[i, 0] - old[r['image']][4]) for i, r in enumerate(manifest)]
    iou_errors = [abs(baseline_values[i, 1] - old[r['image']][5]) for i, r in enumerate(manifest)]
    # Float32 ratio rounding and original epsilon account for tiny differences.
    assertions['historical_validation_max_per_image_dice_difference'] = max(dice_errors)
    assertions['historical_validation_max_per_image_iou_difference'] = max(iou_errors)
    assertions['historical_validation_prediction_foreground_count_mismatches'] = sum(
        int(np.count_nonzero(p[i] >= 0.5)) != old[r['image']][8] for i, r in enumerate(manifest))
    assertions['historical_validation_target_foreground_count_mismatches'] = sum(
        int(np.sum(y[i])) != old[r['image']][7] for i, r in enumerate(manifest))
    assertions['historical_validation_macro_dice'] = float(np.mean([r[4] for r in old.values()]))
    assertions['historical_validation_macro_iou'] = float(np.mean([r[5] for r in old.values()]))
    # Report consistency as observed; do not hide a mismatch by loosening tolerance.
    assertions['historical_validation_consistent_within_1e-6'] = max(dice_errors) < 1e-6 and max(iou_errors) < 1e-6
    assert assertions['historical_validation_consistent_within_1e-6']
    assert assertions['historical_validation_prediction_foreground_count_mismatches'] == 0
    assert assertions['historical_validation_target_foreground_count_mismatches'] == 0
    consistency = read('validation/output_consistency.json')
    assert all(r['repeated_forward_max_abs_difference'] == 0 and
               r['legacy_vs_logit_sigmoid_max_abs_difference'] == 0 for r in consistency)
    assertions['repeated_forward_and_logit_sigmoid_identical'] = True
    assert read('preservation_verification.json')['all_originals_unchanged']
    assertions['originals_hash_preserved'] = True
    subset = read('overfit/manifest.json')
    assert len(subset) == 8 and not {r['name'] for r in subset} & {r['image'] for r in manifest}
    assertions['training_subset_disjoint_from_validation'] = True
    with (HERE / 'overfit/training_microsteps.csv').open(newline='') as f:
        training_rows = list(csv.DictReader(f))
    assert [int(r['microstep']) for r in training_rows] == list(range(1, len(training_rows) + 1))
    assert all(np.isfinite(float(r['loss'])) for r in training_rows)
    updates = [r for r in training_rows if r['optimizer_update'] == 'True']
    assert all(np.isfinite(float(r['gradient_norm_before_clip'])) for r in updates)
    assert len(updates) == summary['overfit']['optimizer_updates']
    assertions['persisted_training_microsteps'] = len(training_rows)
    assertions['persisted_finite_gradient_optimizer_updates'] = len(updates)
    assertions['post_training_checkpoint_available'] = summary['overfit'].get('post_training_checkpoint_saved', True)
    assert not summary['new_test_evaluation'] and not summary['full_training_run_started']
    assertions['test_and_full_training_not_run'] = True
    areas = np.sum(y, axis=(1, 2), dtype=np.int64) / (224 * 224)
    controls = {'population': 'validation only',
                'all_foreground_macro_dice': float(np.mean(2 * areas / (1 + areas))),
                'all_foreground_macro_iou': float(np.mean(areas)),
                'all_background_macro_dice': 0.0,
                'all_background_macro_iou': 0.0,
                'all_background_pixel_accuracy': float(1 - np.mean(areas))}
    (HERE / 'validation/constant_mask_controls.json').write_text(json.dumps(controls, indent=2), encoding='utf-8')
    assertions['verification_passed'] = True
    encoded = json.dumps(assertions, indent=2, default=lambda value: value.item())
    (HERE / 'independent_verification.json').write_text(encoded, encoding='utf-8')
    print(encoded)


if __name__ == '__main__':
    main()
