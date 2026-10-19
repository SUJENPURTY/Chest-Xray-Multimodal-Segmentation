"""CPU-only consolidation after the mandated OOM stop; no model is loaded.

Only persisted measurements are reported. The lost in-memory model cannot be
reconstructed, and no replacement predictions/checkpoint are fabricated.
"""
import csv
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2), encoding='utf-8')


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load_percent', ctypes.c_ulong),
                ('total_physical_bytes', ctypes.c_ulonglong), ('available_physical_bytes', ctypes.c_ulonglong),
                ('total_commit_limit_bytes', ctypes.c_ulonglong), ('available_commit_bytes', ctypes.c_ulonglong),
                ('total_virtual_bytes', ctypes.c_ulonglong), ('available_virtual_bytes', ctypes.c_ulonglong),
                ('available_extended_virtual_bytes', ctypes.c_ulonglong)]


def main():
    failure = load('failure.json')
    assert 'out of memory' in failure['message']
    with (HERE / 'overfit/evaluation_progression.csv').open(newline='') as f:
        history = list(csv.DictReader(f))
    last = history[-1]
    best = max(history, key=lambda r: float(r['macro_dice']))
    def measured(row):
        return {'threshold': 0.5, 'images': 8, 'macro_dice': float(row['macro_dice']),
                'macro_iou': float(row['macro_iou']), 'min_dice': float(row['min_dice']),
                'mean_loss': float(row['mean_loss']),
                'microsteps': int(row['microsteps']), 'optimizer_updates': int(row['optimizer_updates'])}
    subset = {'initial': measured(history[0]), 'best': measured(best), 'final': measured(last),
              'final_label': 'last persisted evaluation, not recovered terminal model',
              'best_microstep': int(best['microsteps']), 'completed_microsteps': int(last['microsteps']),
              'optimizer_updates': int(last['optimizer_updates']),
              'counts_scope': 'persisted counts; additional unlogged steps before failure are unknown',
              'stop_reason': 'CUDA out-of-memory error during backward; no restart',
              'success_criterion_met': False, 'observed_nonfinite_gradient_updates': 0,
              'seconds': float(last['seconds']), 'time_scope': 'time to last persisted check; total failure time unknown',
              'seconds_per_microstep_including_checks': float(last['seconds']) / int(last['microsteps']),
              'training_subset_only': True, 'generalization_measured': False,
              'post_training_checkpoint_saved': False,
              'post_training_predictions_saved': False,
              'initial_predictions_saved': True, **failure['memory']}
    write('overfit/summary.json', subset)
    calibration = load('validation/threshold_selection.json')
    summary = {'completion_status': 'stopped at mandatory OOM boundary; partial subset results preserved',
               'original_test_metrics_reference_only': {
                   'mean_dice': 0.18575444877203687, 'mean_iou': 0.10659993084794975,
                   'source': 'original saved report; no new test inference'},
               'diagnostics': load('diagnostics/numerical_stability.json'),
               'validation_baseline': load('validation/baseline.json'),
               'validation_calibration': calibration, 'overfit': subset,
               'amp_localization_status': 'not executed; guard refused GPU allocation after primary failure',
               'full_training_run_started': False, 'new_test_evaluation': False,
               'training_restarted_after_oom': False, 'failure': failure}
    write('phase1_summary.json', summary)
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    host = {name: getattr(status, name) for name, _ in status._fields_}
    host.update({'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                 'scope': 'post-failure observation; not a during-failure memory measurement'})
    write('host_memory_after_failure.json', host)
    (HERE / 'overfit/CHECKPOINT_NOT_SAVED.md').write_text(
        '# Missing post-training checkpoint and predictions\n\n'
        'The subset run stopped on a CUDA OOM during backward after the last persisted '
        'evaluation at 1120 microsteps / 140 optimizer updates. The best snapshot and '
        'post-training predictions were held in CPU memory and scheduled for export at '
        'normal completion; process failure lost them. Only initial masks/overlays and '
        'the persisted training curves/logs are available. No substitute checkpoint or '
        'post-training masks have been fabricated, and training has not restarted.\n\n'
        'A future approved attempt must save checkpoints and per-image results at each '
        'evaluation, avoid full-model CPU snapshots, monitor host RAM/commit as well '
        'as VRAM, and flush partial results when handling a resource failure.\n', encoding='utf-8')
    print(json.dumps(subset, indent=2))


if __name__ == '__main__':
    main()
