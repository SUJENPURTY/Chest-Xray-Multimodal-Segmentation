"""Bounded, isolated Phase 1 experiment. No held-out test images are opened.

Run with the project Python and -B. All writes are confined to this directory.
The pretrained checkpoint is restored between diagnostic conditions. Overfitting
is explicitly a training-subset sanity test, not evidence of generalization.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_MODULES_CACHE'] = str(HERE / 'cache' / 'hf_modules')
os.environ['MPLCONFIGDIR'] = str(HERE / 'cache' / 'matplotlib')
os.environ['TEMP'] = os.environ['TMP'] = str(HERE / 'tmp')
(HERE / 'tmp').mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer

from src.config import Config
from src.data.dataset import QaTaDataset, group_key
from src.inference.predictor import _model_sources
from src.models.languide_medseg import create_model
from src.training.losses import DiceBCELoss

SEED = 2026
CKPT = ROOT / 'outputs/run/checkpoints/best_dice.ckpt'
CSV = ROOT / 'outputs/run/prepared_annotations/train.csv'
DATA = ROOT / '_qata_compatible'
DEVICE = torch.device('cuda')


def dump(name, value):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(name, rows):
    if not rows:
        return
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(4 * 1024 * 1024):
            result.update(data)
    return result.hexdigest()


def protected_files():
    # Include source files and every existing output, not merely the two weights.
    return sorted(set([*ROOT.glob('notebooks/*.ipynb'),
                       *ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py'),
                       *[p for p in (ROOT / 'outputs/run').rglob('*') if p.is_file()]]))


def preservation_start():
    inventory = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in protected_files()}
    dump('preservation_before.json', inventory)
    return inventory


def preservation_end(before):
    differences = []
    for name, previous in before.items():
        path = ROOT / name
        if not path.is_file() or sha(path) != previous['sha256']:
            differences.append(name)
    result = {'checked_files': len(before), 'all_originals_unchanged': not differences,
              'changed_or_missing': differences}
    dump('preservation_verification.json', result)
    assert not differences, differences


def seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def memory():
    return {'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            'allocated_mib': torch.cuda.memory_allocated() / 2**20,
            'reserved_mib': torch.cuda.memory_reserved() / 2**20}


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def to_device(item):
    (image, text), target = item
    if image.ndim == 3:
        image, target = image.unsqueeze(0), target.unsqueeze(0)
        text = {k: v.unsqueeze(0) for k, v in text.items()}
    return image.to(DEVICE), {k: v.to(DEVICE) for k, v in text.items()}, target.to(DEVICE)


class LogitCapture:
    """Capture pre-sigmoid output without changing weights or original source."""
    def __init__(self, model):
        self.logits = None
        self.handle = model.out.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.logits = output

    def clear(self):
        self.logits = None


def stable_loss(logits, target):
    z, y = logits.float(), target.float()
    probability = torch.sigmoid(z)
    intersection = (probability * y).sum((1, 2, 3))
    soft_dice = (2 * intersection + 1e-7) / (probability.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1e-7)
    dice_loss = 1 - soft_dice.mean()
    bce = F.binary_cross_entropy_with_logits(z, y)
    return dice_loss + bce, probability, dice_loss, bce


def metrics(probability, target, threshold=0.5):
    p, y = probability >= threshold, target > 0.5
    tp = np.count_nonzero(p & y)
    fp = np.count_nonzero(p & ~y)
    fn = np.count_nonzero(~p & y)
    tn = np.count_nonzero(~p & ~y)
    return {'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'predicted_foreground_fraction': float(p.mean()),
            'true_foreground_fraction': float(y.mean())}


def aggregate(rows, threshold=0.5):
    result = {'threshold': float(threshold), 'images': len(rows)}
    for key in ('dice', 'iou', 'precision', 'recall', 'predicted_foreground_fraction', 'true_foreground_fraction'):
        result['macro_' + key] = float(np.mean([r[key] for r in rows]))
    tp, fp, fn, tn = [sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn', 'tn')]
    result.update({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                   'global_dice': 2 * tp / (2 * tp + fp + fn),
                   'global_iou': tp / (tp + fp + fn),
                   'global_precision': tp / (tp + fp) if tp + fp else 0.0,
                   'global_recall': tp / (tp + fn),
                   'global_pixel_accuracy': (tp + tn) / (tp + fp + fn + tn)})
    return result


def tensor_info(tensor):
    if tensor is None:
        return None
    a = tensor.detach().float()
    finite = torch.isfinite(a)
    return {'dtype': str(tensor.dtype), 'nonfinite_elements': int((~finite).sum()),
            'finite_abs_max': float(a[finite].abs().max()) if finite.any() else None}


def grad_info(model):
    bad = []
    square = 0.0
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        count += 1
        if not torch.isfinite(parameter.grad).all():
            bad.append(name)
        else:
            square += parameter.grad.detach().double().square().sum().item()
    return {'gradient_tensors': count, 'nonfinite_gradient_tensors': len(bad),
            'nonfinite_parameter_names': bad, 'finite_gradient_l2_norm': square**0.5}


def restore(model, state):
    model.zero_grad(set_to_none=True)
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def diagnostics(model, capture, state, item):
    original_loss = DiceBCELoss()
    conditions = [('original_fp32', False, False, 1.0),
                  ('stable_fp32', True, False, 1.0),
                  ('original_amp_65536', False, True, 65536.0),
                  ('stable_amp_65536', True, True, 65536.0),
                  ('original_amp_1', False, True, 1.0),
                  ('stable_amp_1', True, True, 1.0)]
    results, reference = [], None
    for name, stable, amp, scale in conditions:
        print('DIAGNOSTIC', name, flush=True)
        restore(model, state)
        model.train()  # Preserve upstream dropout and BN for matched comparisons.
        seed()
        reset_memory()
        image, text, target = to_device(item)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=3e-4, weight_decay=0.01, foreach=False)
        scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=scale)
        watched = {'out': model.out.conv.conv.weight,
                   'encoder': model.encoder.model.embeddings.patch_embeddings.weight}
        old = {k: p.detach().clone() for k, p in watched.items()}
        started = time.perf_counter()
        with torch.autocast('cuda', enabled=amp):
            original_probability = model([image, text])
        logits = capture.logits
        logits.retain_grad()
        original_probability.retain_grad()
        with torch.autocast('cuda', enabled=False):
            if stable:
                loss, probability, dl, bl = stable_loss(logits, target)
            else:
                probability = original_probability.float()
                loss = original_loss(probability, target.float())
                dl = bl = None
        assert torch.isfinite(loss), name
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        row = {'condition': name, 'stable_logits_loss': stable, 'amp': amp,
               'scale_before': scaler.get_scale(), 'loss': float(loss.detach()),
               'dice_loss': float(dl.detach()) if dl is not None else None,
               'bce_loss': float(bl.detach()) if bl is not None else None,
               'logits': tensor_info(logits), 'logit_gradient_scaled': tensor_info(logits.grad),
               'original_probability_gradient_scaled': tensor_info(original_probability.grad),
               **grad_info(model)}
        if name == 'original_fp32':
            reference = {k: p.grad.detach().cpu().clone() for k, p in watched.items()}
        elif name == 'stable_fp32':
            row['fp32_gradient_max_abs_difference'] = {
                k: float((p.grad.detach().cpu() - reference[k]).abs().max()) for k, p in watched.items()}
        if amp or row['nonfinite_gradient_tensors'] == 0:
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        row['scale_after'] = scaler.get_scale()
        row['parameter_max_abs_changes'] = {k: float((p.detach() - old[k]).abs().max())
                                           for k, p in watched.items()}
        row['optimizer_step_applied'] = any(v > 0 for v in row['parameter_max_abs_changes'].values())
        row['seconds'] = time.perf_counter() - started
        row.update(memory())
        results.append(row)
        dump('diagnostics/numerical_stability.json', results)
        print(name, 'loss', row['loss'], 'bad gradients', row['nonfinite_gradient_tensors'],
              'update', row['optimizer_step_applied'], 'peak MiB', row['peak_allocated_mib'], flush=True)
        capture.clear()
        del optimizer, scaler, image, text, target, original_probability, probability, logits, loss, old
        gc.collect()
        torch.cuda.empty_cache()
    restore(model, state)
    model.eval()
    return results


def validate(model, capture, dataset):
    folder = HERE / 'validation'
    folder.mkdir(exist_ok=True)
    assert len(dataset) == 1423
    shape = (len(dataset), 224, 224)
    probs = np.lib.format.open_memmap(folder / 'probabilities.npy', mode='w+', dtype='float32', shape=shape)
    masks = np.lib.format.open_memmap(folder / 'targets.npy', mode='w+', dtype='uint8', shape=shape)
    rows, hist_all, hist_fg, hist_bg = [], np.zeros(100, np.int64), np.zeros(100, np.int64), np.zeros(100, np.int64)
    bins = np.linspace(0, 1, 101)
    model.eval()
    assert not any(m.training for m in model.modules())
    reset_memory()
    started = time.perf_counter()
    consistency = []
    with torch.inference_mode():
        for index, item in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)):
            image, text, target = to_device(item)
            p = model([image, text])
            assert torch.isfinite(p).all() and torch.isfinite(capture.logits).all()
            if index < 3:
                stable_p = torch.sigmoid(capture.logits.float())
                repeated = model([image, text])
                consistency.append({'index': index,
                    'legacy_vs_logit_sigmoid_max_abs_difference': float((p - stable_p).abs().max()),
                    'repeated_forward_max_abs_difference': float((p - repeated).abs().max())})
            a, y = p[0, 0].cpu().numpy(), target[0, 0].cpu().numpy().astype(bool)
            probs[index], masks[index] = a, y
            row = {'image': dataset.records[index]['name'], **metrics(a, y)}
            rows.append(row)
            hist_all += np.histogram(a, bins)[0]
            hist_fg += np.histogram(a[y], bins)[0]
            hist_bg += np.histogram(a[~y], bins)[0]
            capture.clear()
            if (index + 1) % 100 == 0:
                print('VALIDATION', index + 1, '/', len(dataset), flush=True)
    torch.cuda.synchronize()
    probs.flush()
    masks.flush()
    baseline = aggregate(rows)
    baseline.update({'seconds': time.perf_counter() - started, 'precision': 'FP32',
                     'checkpoint_epoch': 18, 'model_and_all_children_eval': True, **memory()})
    csv_write('validation/per_image_threshold_0.5.csv', rows)
    dump('validation/baseline.json', baseline)
    dump('validation/output_consistency.json', consistency)
    distribution = {'minimum': float(probs.min()), 'maximum': float(probs.max()),
                    'mean': float(probs.mean()), 'bins': bins.tolist(),
                    'all_pixel_counts': hist_all.tolist(), 'foreground_counts': hist_fg.tolist(),
                    'background_counts': hist_bg.tolist()}
    dump('validation/probability_distribution.json', distribution)
    plt.figure(figsize=(9, 5))
    centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(centers, hist_fg / hist_fg.sum(), label='Ground-truth foreground')
    plt.plot(centers, hist_bg / hist_bg.sum(), label='Ground-truth background')
    plt.axvline(0.5, color='black', linestyle='--', label='Baseline threshold 0.5')
    plt.xlabel('Predicted foreground probability')
    plt.ylabel('Fraction of pixels within each ground-truth class')
    plt.title('Epoch-18 checkpoint: validation probability distributions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / 'probability_distribution.png', dpi=160)
    plt.close()
    dump('validation/manifest.json', [{'index': i, 'image': r['name'], 'group': group_key(r['name'])}
                                    for i, r in enumerate(dataset.records)])
    return probs, masks, baseline


def calibrate(probs, masks):
    # Predeclared grid, selected by macro validation Dice. Tie: closest to 0.5.
    thresholds = np.round(np.arange(0.05, 0.951, 0.025), 3)
    rows = []
    for threshold in thresholds:
        score = aggregate([metrics(probs[i], masks[i].astype(bool), threshold)
                           for i in range(len(probs))], threshold)
        rows.append(score)
    best = max(rows, key=lambda r: (r['macro_dice'], -abs(r['threshold'] - 0.5)))
    csv_write('validation/threshold_comparison.csv', rows)
    result = {'selection_population': '1423 validation images only',
              'selection_metric': 'macro per-image foreground Dice',
              'tie_rule': 'closest to 0.5', 'test_evaluated': False,
              'selected': best, 'baseline': next(r for r in rows if r['threshold'] == 0.5)}
    dump('validation/threshold_selection.json', result)
    plt.figure(figsize=(9, 5))
    for key, label in [('macro_dice', 'Dice'), ('macro_iou', 'IoU'),
                       ('macro_precision', 'Precision'), ('macro_recall', 'Recall')]:
        plt.plot([r['threshold'] for r in rows], [r[key] for r in rows], label=label)
    plt.axvline(0.5, color='grey', linestyle='--', label='Baseline 0.5')
    plt.axvline(best['threshold'], color='black', linestyle=':', label=f"Selected {best['threshold']:.3f}")
    plt.xlabel('Threshold')
    plt.ylabel('Validation macro score')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / 'validation/threshold_comparison.png', dpi=160)
    plt.close()
    print('CALIBRATION selected', best['threshold'], 'Dice', best['macro_dice'], flush=True)
    return result


def subset_eval(model, capture, items, names):
    model.eval()
    rows, arrays = [], []
    with torch.inference_mode():
        for item, name in zip(items, names):
            image, text, target = to_device(item)
            p = model([image, text])
            loss, _, dl, bl = stable_loss(capture.logits, target)
            a, y = p[0, 0].cpu().numpy(), target[0, 0].cpu().numpy().astype(bool)
            assert np.isfinite(a).all()
            rows.append({'image': name, **metrics(a, y), 'loss': float(loss),
                         'dice_loss': float(dl), 'bce_loss': float(bl)})
            arrays.append((a.copy(), y.copy()))
            capture.clear()
    return rows, arrays


def examples(items, records, arrays, label):
    folder = HERE / 'overfit' / label
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(items), 4, figsize=(12, 3 * len(items)), squeeze=False)
    for i, (item, record, (a, y)) in enumerate(zip(items, records, arrays)):
        original = np.asarray(Image.open(record['image']).convert('L'))
        # MONAI tensor x,y -> original PIL y,x for all exported visualizations.
        prediction, truth = (a >= 0.5).T, y.T
        Image.fromarray(original).save(folder / f'{i:02d}_xray.png')
        Image.fromarray((truth * 255).astype('uint8')).save(folder / f'{i:02d}_ground_truth.png')
        Image.fromarray((prediction * 255).astype('uint8')).save(folder / f'{i:02d}_prediction.png')
        overlay = np.repeat(original[:, :, None], 3, axis=2).astype(float) / 255
        overlay[truth] = overlay[truth] * 0.6 + np.array([0, 1, 0]) * 0.4
        overlay[prediction] = overlay[prediction] * 0.6 + np.array([1, 0, 0]) * 0.4
        Image.fromarray((overlay * 255).astype('uint8')).save(folder / f'{i:02d}_overlay.png')
        np.save(folder / f'{i:02d}_probability.npy', a.T)
        score = metrics(a, y)
        for j, (data, title) in enumerate([(original, 'X-ray'), (truth, 'Ground truth'),
                                          (prediction, f"Prediction Dice {score['dice']:.3f}"),
                                          (overlay, 'GT green / prediction red')]):
            axes[i, j].imshow(data, cmap='gray' if j < 3 else None)
            axes[i, j].set_title(title)
            axes[i, j].axis('off')
    fig.suptitle(f'{label}: eight training images only; threshold 0.5')
    plt.tight_layout()
    plt.savefig(folder / 'comparison.png', dpi=120)
    plt.close(fig)


def overfit(model, capture, state, dataset, indices, max_microsteps):
    restore(model, state)
    seed()
    records = [dataset.records[i] for i in indices]
    names = [r['name'] for r in records]
    items = [dataset[i] for i in indices]  # Cached in CPU RAM, augmentation disabled.
    dump('overfit/manifest.json', [{'training_index': i, 'name': r['name'], 'prompt': r['prompt']}
                                  for i, r in zip(indices, records)])
    initial_rows, initial_arrays = subset_eval(model, capture, items, names)
    csv_write('overfit/initial_per_image.csv', initial_rows)
    examples(items, records, initial_arrays, 'initial')
    vision = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('encoder.model.')]
    decoder = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.model.')]
    optimizer = torch.optim.AdamW([{'params': vision, 'lr': 3e-5}, {'params': decoder, 'lr': 3e-4}],
                                  weight_decay=0.01, foreach=False)
    config = {'seed': SEED, 'sample_count': 8, 'microbatch': 1, 'accumulation': 8,
              'max_microsteps': max_microsteps, 'maximum_optimizer_updates': max_microsteps // 8,
              'max_wall_seconds': 1800, 'augmentation': False, 'precision': 'FP32',
              'vision_lr': 3e-5, 'decoder_lr': 3e-4, 'weight_decay': 0.01,
              'gradient_clip_l2': 1.0, 'text_backbone_eval_during_training': True,
              'decoder_batchnorm': 'training mode; microbatch statistics',
              'evaluation_threshold': 0.5, 'success_macro_dice': 0.9,
              'early_stop_plateau_checks': 6, 'minimum_microsteps_before_plateau_stop': 640,
              'status': 'training subset sanity test, not a generalization experiment'}
    dump('overfit/config.json', config)
    initial = aggregate(initial_rows)
    curve = [{'microsteps': 0, 'optimizer_updates': 0, 'seconds': 0.0,
              'macro_dice': initial['macro_dice'], 'macro_iou': initial['macro_iou'],
              'mean_loss': float(np.mean([r['loss'] for r in initial_rows])), 'min_dice': min(r['dice'] for r in initial_rows)}]
    best_dice, best_microstep, best_rows, best_arrays = initial['macro_dice'], 0, initial_rows, initial_arrays
    best_state = None
    updates, training_rows, successes = 0, [], 0
    reset_memory()
    started = time.perf_counter()
    order = list(range(8))
    stop = 'microstep limit'
    for step in range(1, max_microsteps + 1):
        if (step - 1) % 8 == 0:
            random.shuffle(order)
            optimizer.zero_grad(set_to_none=True)
        model.train()
        model.text_encoder.model.eval()  # Explicit deterministic frozen-text variant.
        image, text, target = to_device(items[order[(step - 1) % 8]])
        model([image, text])
        loss, probability, dl, bl = stable_loss(capture.logits, target)
        if not torch.isfinite(loss) or not torch.isfinite(probability).all():
            stop = 'nonfinite forward or loss'
            break
        (loss / 8).backward()
        row = {'microstep': step, 'loss': float(loss.detach()), 'dice_loss': float(dl.detach()),
               'bce_loss': float(bl.detach()), 'optimizer_update': False, 'gradient_norm_before_clip': None}
        capture.clear()
        del loss, probability, dl, bl, image, text, target
        if step % 8 == 0:
            info = grad_info(model)
            if info['nonfinite_gradient_tensors']:
                stop = 'nonfinite accumulated gradients'
                dump('overfit/instability.json', info)
                break
            row['gradient_norm_before_clip'] = info['finite_gradient_l2_norm']
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                           error_if_nonfinite=True, foreach=False)
            optimizer.step()
            updates += 1
            row['optimizer_update'] = True
            if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                stop = 'nonfinite updated parameters'
                break
        training_rows.append(row)
        if step % 80 == 0 or step == max_microsteps:
            rows, arrays = subset_eval(model, capture, items, names)
            result = aggregate(rows)
            elapsed = time.perf_counter() - started
            point = {'microsteps': step, 'optimizer_updates': updates, 'seconds': elapsed,
                     'macro_dice': result['macro_dice'], 'macro_iou': result['macro_iou'],
                     'mean_loss': float(np.mean([r['loss'] for r in rows])),
                     'min_dice': min(r['dice'] for r in rows)}
            curve.append(point)
            csv_write('overfit/evaluation_progression.csv', curve)
            csv_write('overfit/training_microsteps.csv', training_rows)
            print('OVERFIT', step, 'updates', updates, 'Dice', point['macro_dice'],
                  'loss', point['mean_loss'], 'peak MiB', memory()['peak_allocated_mib'], flush=True)
            if point['macro_dice'] > best_dice:
                best_dice, best_microstep, best_rows, best_arrays = point['macro_dice'], step, rows, arrays
                # One CPU snapshot only; no duplicate model on the GPU.
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            successes = successes + 1 if point['macro_dice'] >= 0.9 and point['min_dice'] >= 0.75 else 0
            if successes >= 2:
                stop = 'success criterion met twice'
                break
            if elapsed >= config['max_wall_seconds']:
                stop = '30-minute wall-time limit'
                break
            if step >= 640 and len(curve) >= 7 and max(r['macro_dice'] for r in curve[-6:]) - curve[-6]['macro_dice'] < 0.003:
                stop = 'training-subset Dice plateau'
                break
    torch.cuda.synchronize()
    resource = memory()
    total_seconds = time.perf_counter() - started
    final_rows, final_arrays = subset_eval(model, capture, items, names)
    csv_write('overfit/final_per_image.csv', final_rows)
    csv_write('overfit/best_per_image.csv', best_rows)
    csv_write('overfit/training_microsteps.csv', training_rows)
    examples(items, records, final_arrays, 'final')
    examples(items, records, best_arrays, 'best')
    if best_state is not None:
        torch.save({'kind': 'PHASE1_TRAINING_SUBSET_SANITY_V1', 'model': best_state,
                    'parent_checkpoint_sha256': sha(CKPT), 'config': config,
                    'best_microstep': best_microstep, 'best_training_subset_dice': best_dice},
                   HERE / 'overfit/best_subset.ckpt')
    result = {'initial': initial, 'best': aggregate(best_rows), 'final': aggregate(final_rows),
              'best_microstep': best_microstep, 'completed_microsteps': len(training_rows),
              'optimizer_updates': updates, 'stop_reason': stop,
              'success_criterion_met': successes >= 2, 'nonfinite_updates': 0 if 'nonfinite' not in stop else 1,
              'seconds': total_seconds, 'seconds_per_microstep_including_checks': total_seconds / max(len(training_rows), 1),
              'training_subset_only': True, 'generalization_measured': False, **resource}
    dump('overfit/summary.json', result)
    plt.figure(figsize=(9, 5))
    plt.plot([r['microsteps'] for r in curve], [r['macro_dice'] for r in curve], marker='o', label='Training subset Dice')
    plt.plot([r['microsteps'] for r in curve], [r['macro_iou'] for r in curve], marker='.', label='Training subset IoU')
    plt.axhline(0.9, color='grey', linestyle='--', label='Proposed Dice sanity criterion')
    plt.xlabel('Microsteps (8 per optimizer update)')
    plt.ylabel('Eight-image score at threshold 0.5')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / 'overfit/training_progression.png', dpi=160)
    plt.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-microsteps', type=int, default=2000)
    args = parser.parse_args()
    assert 80 <= args.max_microsteps <= 2000 and args.max_microsteps % 8 == 0
    assert torch.cuda.is_available(), 'A CUDA GPU is required; do not silently start CPU training.'
    assert not (HERE / 'preservation_before.json').exists(), 'Existing experiment: use a new isolated directory.'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed()
    before = preservation_start()
    try:
        dump('environment.json', {'python': sys.version, 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(), 'total_vram_mib': torch.cuda.get_device_properties(0).total_memory / 2**20,
            'test_data_opened': False, 'seed': SEED, 'checkpoint': str(CKPT), 'data_root': str(DATA)})
        saved = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
        assert saved['epoch'] == 18 and saved['kind'] == 'CLOUD_LANGUIDE_BINARY_FIXED_V1'
        state = {k.removeprefix('model.'): v for k, v in saved['model'].items()}
        cfg = Config(data_root=DATA)
        vision, text_source = _model_sources(cfg)
        tokenizer = AutoTokenizer.from_pretrained(text_source, trust_remote_code=True, local_files_only=True)
        train = QaTaDataset(CSV, DATA, 'train', tokenizer, augment=False, seed=SEED)
        validation = QaTaDataset(CSV, DATA, 'validation', tokenizer, augment=False, seed=SEED)
        assert len(train) == 5722 and len(validation) == 1423
        assert not {r['name'] for r in train.records} & {r['name'] for r in validation.records}
        rng = random.Random(SEED)
        indices = rng.sample(range(len(train)), 8)
        # Ensure eight distinct recognized/filename-derived groups.
        used, selected = set(), []
        for i in indices + rng.sample(range(len(train)), len(train)):
            key = group_key(train.records[i]['name'])
            if key not in used:
                used.add(key)
                selected.append(i)
            if len(selected) == 8:
                break
        dump('protocol.json', {'validation_count': len(validation), 'training_count': len(train),
            'test_images_read': 0, 'subset_selection': 'seeded random training samples, distinct filename-derived groups',
            'subset_indices': selected, 'threshold_grid': '0.05 through 0.95 inclusive in steps of 0.025',
            'threshold_selection_metric': 'macro validation Dice', 'overfit_threshold': 0.5,
            'pretrained_vision_source': vision, 'pretrained_text_source': text_source})
        print('Constructing one model; strictly loading epoch 18', flush=True)
        model = create_model(vision, text_source)
        restore(model, state)
        matched, nonfinite = 0, []
        for k, value in model.state_dict().items():
            assert torch.equal(value, state[k]), k
            matched += 1
            if value.is_floating_point() and not torch.isfinite(value).all():
                nonfinite.append(k)
        assert not nonfinite
        dump('checkpoint_verification.json', {'epoch': saved['epoch'], 'sha256': sha(CKPT),
            'strict_load': True, 'exactly_matching_state_tensors': matched,
            'nonfinite_state_tensors': nonfinite, 'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
        model.to(DEVICE)
        capture = LogitCapture(model)
        diagnostic_results = diagnostics(model, capture, state, train[selected[0]])
        probs, masks, baseline = validate(model, capture, validation)
        calibration = calibrate(probs, masks)
        del probs, masks
        subset = overfit(model, capture, state, train, selected, args.max_microsteps)
        dump('phase1_summary.json', {'original_test_metrics_reference_only': {
                'mean_dice': 0.18575444877203687, 'mean_iou': 0.10659993084794975,
                'source': 'outputs/run/evaluation/test_metrics_summary.json; no new test inference'},
            'diagnostics': diagnostic_results, 'validation_baseline': baseline,
            'validation_calibration': calibration, 'overfit': subset,
            'full_training_run_started': False, 'new_test_evaluation': False})
        print('PHASE 1 COMPLETE', flush=True)
    except Exception as exc:
        dump('failure.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
                              'memory': memory() if torch.cuda.is_initialized() else None})
        raise
    finally:
        preservation_end(before)


if __name__ == '__main__':
    main()
