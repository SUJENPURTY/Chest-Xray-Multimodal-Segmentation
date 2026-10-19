"""CPU-only artifact verification/report. Never loads a model or test dataset."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']=str(HERE/'cache/matplotlib')
os.environ['TEMP']=os.environ['TMP']=str(HERE/'tmp')
for p in [HERE/'cache/matplotlib',HERE/'tmp']: p.mkdir(parents=True,exist_ok=True)
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES=['A_current_bn','B_frozen_bn','C_groupnorm']
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def write(name,obj): (HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024*1024): h.update(b)
    return h.hexdigest()

def verify_images(directory,table):
    result=[]
    for i,row in enumerate(table):
        def mask(suffix):
            with Image.open(directory/f'{i:02d}_{suffix}.png') as f: return np.asarray(f)>0
        y,p=mask('ground_truth'),mask('prediction')
        with Image.open(ROOT/'_qata_compatible/Train/GTs'/row['image']) as f: raw=np.asarray(f)>0
        assert np.array_equal(y,raw),'Export GT orientation mismatch'
        prob=np.load(directory/f'{i:02d}_probability.npy')
        assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
        assert np.array_equal(p,prob>=.5)
        tp,fp,fn,tn=[int(a.sum()) for a in [p&y,p&~y,~p&y,~p&~y]]
        dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
        iou=tp/(tp+fp+fn) if tp+fp+fn else 1.
        assert all(int(row[k])==v for k,v in zip(['tp','fp','fn','tn'],[tp,fp,fn,tn]))
        assert abs(float(row['dice'])-dice)<1e-12 and abs(float(row['iou'])-iou)<1e-12
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        assert abs(float(row['precision'])-precision)<1e-12 and abs(float(row['recall'])-recall)<1e-12
        assert abs(float(row['predicted_foreground_fraction'])-float(p.mean()))<1e-12
        assert abs(float(row['true_foreground_fraction'])-float(y.mean()))<1e-12
        result.append({'image':row['image'],'dice':dice,'iou':iou,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                       'precision':precision,'recall':recall,'predicted_foreground_fraction':float(p.mean()),
                       'pixels':int(y.size),'GT_foreground_fraction':float(y.mean())})
    return result


assert all((HERE/n/'summary.json').exists() for n in NAMES),'All three completed variants required'
summaries={n:read(HERE/n/'summary.json') for n in NAMES}
configs={n:read(HERE/n/'config.json') for n in NAMES}
inventories={n:read(HERE/n/'normalization_inventory.json') for n in NAMES}
checks={}
selection={}
resources={}
shared_keys=['seed','microbatch','accumulation','max_microsteps','vision_lr','decoder_lr','weight_decay',
             'clip_l2','threshold','precision','AMP','augmentation','max_training_wall_seconds']
assert all(len({str(configs[n][k]) for n in NAMES})==1 for k in shared_keys)
assert len({inventories[n]['initial_parameter_sha256'] for n in NAMES})==1
assert len({inventories[n]['parent_checkpoint_sha256'] for n in NAMES})==1
assert len({json.dumps(read(HERE/n/'sample_manifest.json'),sort_keys=True) for n in NAMES})==1
assert len({json.dumps(read(HERE/n/'validation_manifest.json'),sort_keys=True) for n in NAMES})==1
orders=[]
comparison=[]
validation_comparison=[]
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for name in NAMES:
    p=HERE/name;s=summaries[name]
    assert s['diagnostic_completed'] and s['completed_microsteps']==2000 and s['optimizer_updates']==250
    history=rows(p/'training_microsteps.csv');curve=rows(p/'evaluation_progression.csv')
    assert len(history)==2000 and [int(r['microstep']) for r in history]==list(range(1,2001))
    orders.append([int(r['sample_position']) for r in history])
    updates=[r for r in history if int(r['gradient_tensors'])]
    assert len(updates)==250 and all(int(r['nonfinite_gradient_tensors'])==0 for r in updates)
    assert all(math.isfinite(float(r['loss'])) for r in history)
    assert all(float(r['watched_parameter_max_update'])>0 for r in updates)
    table=rows(p/'final_per_image.csv');verified=verify_images(p/'predictions/final',table)
    for metric,key,fn in [('dice','mean_dice',statistics.mean),('iou','mean_iou',statistics.mean),('dice','minimum_dice',min)]:
        assert abs(fn(v[metric] for v in verified)-s['final'][key])<1e-12
    opt_table=rows(p/'final_optimization_mode_per_image.csv')
    verified_opt=verify_images(p/'predictions/final_optimization_mode',opt_table)
    val_table=rows(p/'final_validation_per_image.csv')
    verified_val=verify_images(p/'predictions/validation_final',val_table)
    assert len(verified)==8 and len(verified_val)==32
    assert abs(statistics.mean(v['dice'] for v in verified_val)-s['validation_32']['mean_dice'])<1e-12
    assert abs(statistics.mean(v['iou'] for v in verified_val)-s['validation_32']['mean_iou'])<1e-12
    totals={k:sum(v[k] for v in verified_val) for k in ['tp','fp','fn','tn','pixels']}
    tp,fp,fn=totals['tp'],totals['fp'],totals['fn']
    val_metrics={'variant':name,'images':32,'threshold':.5,
        'mean_dice':statistics.mean(v['dice'] for v in verified_val),
        'mean_iou':statistics.mean(v['iou'] for v in verified_val),
        'mean_precision':statistics.mean(v['precision'] for v in verified_val),
        'mean_recall':statistics.mean(v['recall'] for v in verified_val),
        'predicted_foreground_percent':100*statistics.mean(v['predicted_foreground_fraction'] for v in verified_val),
        'GT_foreground_percent':100*statistics.mean(v['GT_foreground_fraction'] for v in verified_val),
        'global_dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
        'global_iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,
        'global_precision':tp/(tp+fp) if tp+fp else 0.,
        'global_recall':tp/(tp+fn) if tp+fn else 0.,**totals}
    validation_comparison.append(val_metrics)
    persist=read(p/'checkpoint_persistence_verification.json')
    assert persist['verified'] and persist['optimizer_step_counters']==[250.]
    final_pointer=read(p/'checkpoints/latest.json')
    assert final_pointer==s['final_checkpoint'] and not final_pointer['path'].endswith('.tmp')
    assert (p/final_pointer['path']).stat().st_size==final_pointer['bytes']
    print('VERIFY FINAL CHECKPOINT SHA256',name,flush=True)
    assert sha(p/final_pointer['path'])==final_pointer['sha256']
    last_two=[r for r in curve if int(r['microstep']) in (1800,2000)]
    reliable=len(last_two)==2 and all(float(r['mean_dice'])>=.90 and float(r['minimum_dice'])>=.75 for r in last_two)
    selection[name]={'mean_minimum_criterion_passed':reliable,
                     'last_two_checks':[{k:r[k] for k in ['microstep','mean_dice','minimum_dice']} for r in last_two]}
    gpu=rows(p/'gpu_monitor.csv')
    resources[name]={'whole_GPU_sampled_peak_mib':max(float(r['device_used_mib']) for r in gpu),
                     'minimum_GPU_free_mib':min(float(r['device_free_mib']) for r in gpu),
                     'minimum_available_host_RAM_mib':min(float(r['available_ram_mib']) for r in gpu),
                     'minimum_available_commit_mib':min(float(r['available_commit_mib']) for r in gpu),
                     'maximum_temperature_c':max(float(r['temperature_c']) for r in gpu)}
    checks[name]={'saved_mask_metrics_verified':True,'original_GT_orientation_matches':True,
                 'final_images':verified,'optimization_mode_images':verified_opt,'validation_images':verified_val,
                 'finite_gradient_updates':len(updates),'watched_parameter_updates_nonzero':250,
                 'updates_with_preclip_norm_above_one':sum(float(r['gradient_norm_before_clip'])>1 for r in updates),
                 'gradient_norm_range':[min(float(r['gradient_norm_before_clip']) for r in updates),
                                        max(float(r['gradient_norm_before_clip']) for r in updates)],
                 'loss_range':[min(float(r['loss']) for r in history),max(float(r['loss']) for r in history)],
                 'checkpoint_reload_verified':True}
    checks[name]['target_distribution']={
        label:{'images':len(values),'empty_GT_images':sum(v['tp']+v['fn']==0 for v in values),
               'foreground_fraction_range':[min(v['GT_foreground_fraction'] for v in values),
                                            max(v['GT_foreground_fraction'] for v in values)]}
        for label,values in [('training8',verified),('validation32',verified_val)]}
    for ax,key in zip(axes,['mean_dice','minimum_dice','mean_iou']):
        ax.plot([int(r['microstep']) for r in curve],[float(r[key]) for r in curve],marker='.',label=name)
        ax.set(xlabel='Microsteps (8 per update)',ylabel=key.replace('_',' '),ylim=(0,1))
    comparison.append({'variant':name,'architecture_modified':s['architecture_modified'],**s['final'],
                       'optimization_mode_mean_dice':s['training_evaluation_consistency']['optimization_mode']['mean_dice'],
                       'optimization_mode_min_dice':s['training_evaluation_consistency']['optimization_mode']['minimum_dice'],
                       'mode_probability_max_difference':s['training_evaluation_consistency']['maximum_probability_difference'],
                       'validation32_mean_dice':s['validation_32']['mean_dice'],
                       'validation32_mean_iou':s['validation_32']['mean_iou'],
                       'validation32_mean_precision':val_metrics['mean_precision'],
                       'validation32_mean_recall':val_metrics['mean_recall'],
                       'validation32_predicted_foreground_percent':val_metrics['predicted_foreground_percent'],
                       'validation32_GT_foreground_percent':val_metrics['GT_foreground_percent'],
                       'peak_allocated_mib':s['peak_allocated_mib'],'peak_reserved_mib':s['peak_reserved_mib'],
                       'compute_seconds_per_microstep':s['compute_seconds_per_microstep'],
                       'reliability_criterion_passed':reliable,'completion_status':'Complete: 2000 microsteps / 250 updates'})
assert orders[0]==orders[1]==orders[2]
axes[0].legend(fontsize=8)
fig.suptitle('Phase 1C: same eight training images, matched 250 updates, evaluation mode')
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(HERE/'normalization_progression.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for name in NAMES:
    history=rows(HERE/name/'training_microsteps.csv')
    curve=rows(HERE/name/'evaluation_progression.csv')
    cycle_losses=[statistics.mean(float(r['loss']) for r in history[i:i+8]) for i in range(0,2000,8)]
    axes[0].plot(range(1,251),cycle_losses,label=name)
    axes[1].plot([int(r['microstep'])//8 for r in curve],[float(r['mean_loss']) for r in curve],marker='.',label=name)
axes[0].set(title='Optimization-mode mean loss per eight-sample update',xlabel='Optimizer updates',ylabel='BCE + soft Dice')
axes[1].set(title='Evaluation-mode loss on the same eight images',xlabel='Optimizer updates',ylabel='BCE + soft Dice')
axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(HERE/'loss_progression.png',dpi=160);plt.close(fig)
with (HERE/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)

# The user's continuation request supersedes the historical training-only decision rule.
# Compare final, equal-budget checkpoints using validation first, with reliability as a safety qualifier.
eligible=[n for n in NAMES if selection[n]['mean_minimum_criterion_passed']]
raw_best=max(eligible or NAMES,key=lambda n:(summaries[n]['final']['minimum_dice'],summaries[n]['final']['mean_dice']))
candidate=max(eligible or NAMES,key=lambda n:summaries[n]['validation_32']['mean_dice'])
decision={'selected_diagnostic_variant':candidate,'raw_highest_minimum_variant':raw_best,
          'criterion':'Among numerically stable completed variants passing the final two all-image fitting checks, select highest final fixed-32 validation mean Dice; review evaluation consistency and memory. Training Dice is not the ranking objective.',
          'decision_rule_amended_by':'User continuation request: recommend primarily using validation performance and reliability',
          'historical_training_only_rule_superseded':True,
          'ready_for_bounded_pilot':bool(eligible),'selection_uses_validation':True,
          'full_dataset_training_started':False,'generalization_improvement_established':False,
          'next_training_requires_approval':True,'validation_threshold':.5,'test_evaluated':False}
write('decision.json',decision)
for filename,values in [('validation_comparison.csv',validation_comparison)]:
    with (HERE/filename).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
write('validation_comparison.json',{'aggregation':'Primary metrics are per-image macro means; global metrics separately pool pixel confusion counts.',
     'foreground_percent':'Macro mean percentage of image pixels; all images have equal dimensions.',
     'undefined_precision_recall':'Zero when denominator is zero; empty/empty Dice and IoU equal one.',
     'variants':validation_comparison,'test_evaluated':False})

weak='mask_covid_279.png'
fig,axes=plt.subplots(3,4,figsize=(13,10))
weak_rows={}
for row_index,name in enumerate(NAMES):
    table=rows(HERE/name/'final_per_image.csv')
    i=next(i for i,r in enumerate(table) if r['image']==weak)
    row=table[i];weak_rows[name]=row
    for j,suffix in enumerate(['xray','ground_truth','prediction','overlay']):
        with Image.open(HERE/name/f'predictions/final/{i:02d}_{suffix}.png') as f:
            axes[row_index,j].imshow(np.asarray(f),cmap='gray' if j<3 else None)
        axes[row_index,j].axis('off')
        axes[row_index,j].set_title(f'{name}: {suffix}' if j==0 else
                (f'Dice {float(row["dice"]):.4f}' if suffix=='prediction' else suffix.replace('_',' ')),fontsize=10)
fig.suptitle('Same previously weakest training image: evaluation mode, fixed threshold 0.5')
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(HERE/'weakest_image_comparison.png',dpi=160);plt.close(fig)
write('weakest_image_comparison.json',weak_rows)

# Rehash originals and previous experiments after all ML subprocesses have exited.
before=read(HERE/'preservation_before.json');changed=[]
for i,(name,expected) in enumerate(before.items()):
    p=ROOT/name
    if not p.is_file() or p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']: changed.append(name)
    if i%50==0: print('PRESERVATION CHECK',i,'/',len(before),flush=True)
preservation={'checked_files':len(before),'all_protected_files_unchanged':not changed,'changed_or_missing':changed}
write('preservation_verification.json',preservation)
assert not changed,changed
write('independent_verification.json',{'CPU_only':True,'all_hyperparameters_except_normalization_matched':True,
          'all_starting_parameter_hashes_identical':True,'sample_orders_identical':True,
          'same_training_and_validation_manifests':True,'variants':checks,'all_checks_passed':True})
write('GPU_resource_summary.json',resources)

lines=['# Phase 1C: controlled normalization investigation','',
       'All three experiments completed their matched 2,000-microstep / 250-update budget. Only normalization differs. '
       'These are fitting diagnostics on eight training images, not a reproduction benchmark. The original notebooks, '
       'weights, datasets and earlier experiments were preserved. No test inference, threshold selection or full-dataset training occurred.','',
       '## Matched comparison','',
       '| Variant | Architecture changed | Mean training Dice (eval mode) | Minimum training Dice | Training IoU | Validation Dice | Validation IoU | Peak allocated MiB | Completion |',
       '|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in comparison:
    lines.append(f"| {r['variant']} | {r['architecture_modified']} | {r['mean_dice']:.6f} | {r['minimum_dice']:.6f} | {r['mean_iou']:.6f} | {r['validation32_mean_dice']:.6f} | {r['validation32_mean_iou']:.6f} | {r['peak_allocated_mib']:.2f} | Complete: 2000 / 250 |")
lines+=['','All rows use final matched-budget weights and threshold 0.5. Mean metrics are per-image macro averages. '
        'A fresh identical AdamW optimizer is used for each variant. No checkpoint was chosen by its validation score. '
        'The starting weights are original epoch 18, not the already overfit Phase 1B subset checkpoint. Thus the Phase 1B '
        '0.922282 result is contextual and not silently reused as a matched A result.','',
        'The first 320 sample positions match the prior Phase 1B run, but floating-point loss trajectories diverge '
        'from microstep 18 (initial difference 5.96e-8). See prior_phase1b_vs_A.json. The memory-efficient backward '
        'path is nondeterministic; the common Phase 1C runner also uses PyTorch default foreach selection for '
        'gradient clipping, whereas the previous runner explicitly disabled foreach clipping. Checkpoint/evaluation '
        'cadence differs from Phase 1B. These do not vary between A/B/C, but historical Phase 1B scores are not '
        'treated as a matched control. The exact contribution of each execution difference is not isolated.','',
        'A trains BN running statistics normally. B keeps epoch-18 running statistics fixed during optimization and inference, '
        'while affine weights remain trainable. C replaces only the measured BN layers with GroupNorm, copies their '
        'affine parameters exactly and retains all other parameters. C changes the released architecture; B changes '
        'training behavior. Any future GN result must be labeled a variant rather than an exact reproduction.','',
        'Shared settings: seed 2026; same image/text/mask manifest and exact shuffled sample order; no augmentation; '
        'FP32; microbatch 1; accumulation 8; logits BCE + FP32 sigmoid Dice; AdamW foreach=False; encoder LR 3e-5 / '
        'decoder LR 3e-4; weight decay .01; clip norm 1; frozen CXR-BERT eval; no attention-weight retention. '
        'See PROTOCOL.md and per-variant config.json.','',
        '## Layer inspection and batch-size mechanism','',
        '| Module | Type | Channels | Momentum | Epsilon |','|---|---|---:|---:|---:|']
for r in inventories[NAMES[0]]['original']:
    lines.append(f"| {r['name']} | {r['type']} | {r['channels']} | {r.get('momentum','')} | {r['eps']} |")
lines+=['','The measured segmentation output/subpixel head has no BatchNorm; the relevant BN layers are decoder '
        'convolution-block normalization. ConvNeXt/attention/text LayerNorm modules were preserved.','',
        'BatchNorm2d with batch size one is still mathematically defined because it reduces over B×H×W. However, '
        'spatial pixels are correlated, and each update reflects one image. Accumulation eight combines gradients, '
        'not BN statistics. Running means/variances may fail to match an individual small-lesion image as features '
        'change. GroupNorm normalizes within each image and uses the same rule in train/eval. Frozen BN uses the '
        'same fixed running statistics in both modes, which can also be stale. Neither choice guarantees generalization.','',
        'Initial/final probes record layer input shapes, BHW sample counts and mean/variance discrepancies under inference_mode. '
        'The probes restore BN buffers and RNG afterward and perform no optimizer updates. This avoids contaminating '
        'the experiment with a diagnostic recalibration.','',
        '## Training/evaluation consistency and original weakest image','',
        '| Variant | Optimization-mode mean Dice | Evaluation-mode mean Dice | Max probability difference | Weak image eval Dice | Weak image precision / recall |',
        '|---|---:|---:|---:|---:|---:|']
for r in comparison:
    w=weak_rows[r['variant']]
    lines.append(f"| {r['variant']} | {r['optimization_mode_mean_dice']:.6f} | {r['mean_dice']:.6f} | {r['mode_probability_max_difference']:.6g} | {float(w['dice']):.6f} | {float(w['precision']):.6f} / {float(w['recall']):.6f} |")
lines+=['','Optimization-mode scores are diagnostics and are never substituted for evaluation-mode results. '
        '`weakest_image_comparison.png` shows the same mask_covid_279 image, ground truth, prediction and overlay '
        'for all variants. Every sample also has separate exported images and probability arrays.','',
        '| Variant | Running-stat buffer tensors changed | Max mean / variance delta | Counter increment |',
        '|---|---:|---:|---:|']
for name in NAMES:
    n=summaries[name]['normalization_final']
    lines.append(f"| {name} | {n['changed_buffer_tensors']} | {n['maximum_running_mean_delta']:.5g} / {n['maximum_running_variance_delta']:.5g} | {n['maximum_counter_increment']} |")
lines+=['','Frozen BN buffer equality was asserted. GroupNorm has no running statistics. '
        'See initial/final_mode_consistency.json for per-image train/eval differences and normalization input measurements.','',
        '## Stability, resources and persistence','',
        '| Variant | Finite gradient updates | Gradient norm range (before clipping) | Updates clipped | GPU sampled peak MiB | Minimum free VRAM MiB | Compute seconds / microstep |',
        '|---|---:|---:|---:|---:|---:|---:|']
for name in NAMES:
    g=checks[name]['gradient_norm_range'];r=resources[name];s=summaries[name]
    lines.append(f"| {name} | 250 / 250 | {g[0]:.5f}–{g[1]:.5f} | {checks[name]['updates_with_preclip_norm_above_one']} | {r['whole_GPU_sampled_peak_mib']:.0f} | {r['minimum_GPU_free_mib']:.0f} | {s['compute_seconds_per_microstep']:.3f} |")
lines+=['','No OOM/nonfinite events occurred. The watched output parameter changed on every optimizer update. '
        'Microstep losses and updated weights were checked finite. `normalization_progression.png`, '
        '`loss_progression.png`, '
        'training_microsteps.csv/evaluation_progression.csv and comparison.csv contain progression and exact numbers.','',
        'Whole-GPU samples include other applications and occur every five seconds; PyTorch allocator peaks are '
        'separate measurements. The host RAM/commit minima are recorded in GPU_resource_summary.json. No unrelated '
        'process was stopped. Microbatch 1 is the measured configuration; a larger physical batch is not inferred.','',
        'Each variant has separate periodic/final model+optimizer+RNG checkpoints, atomic temporary writes, SHA-256 '
        'pointers and incremental histories. Strict reload at update 50 and update 250 reproduced all eight predictions '
        'exactly; optimizer moments remained finite and counters matched. A pre-replace interrupted-write simulation '
        'left each committed checkpoint unchanged. CPU optimizer scalars/RNG are cloned on reload to release the '
        'whole checkpoint mmap. Resume code verifies SHA/prediction parity, restores exact order/RNG and archives '
        'uncommitted history. No actual interruption was needed for these three completed runs. The prior usage limit '
        'interrupted the Codex conversation, but the controller continued and all three child processes exited '
        'successfully. Recovery performed no additional training and did not rerun any completed variant. '
        'This demonstrates continued optimization after in-process checkpoint reload at microstep 400 and '
        'interrupted-write safety; a killed-process/relaunched-process continuation was not newly exercised in Phase 1C.','',
        'The attention backward path may be nondeterministic under deterministic_algorithms(warn_only=True). Seeds '
        'and exact sample order are fixed, but bitwise-identical independent retraining is not promised. This limits '
        'interpretation of extremely small score differences; checkpoint inference reload parity is exact.','',
        'There is one seeded run per variant; across-seed robustness has not been measured.','',
        '## Independent validation context: fixed 32 images','',
        '| Variant | Initial Dice | Final Dice | Final IoU | Precision | Recall | Predicted foreground % | GT foreground % | Minimum final Dice |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name in NAMES:
    iv=read(HERE/name/'initial_validation_summary.json');v=summaries[name]['validation_32']
    vm=next(r for r in validation_comparison if r['variant']==name)
    lines.append(f"| {name} | {iv['mean_dice']:.6f} | {v['mean_dice']:.6f} | {v['mean_iou']:.6f} | {vm['mean_precision']:.6f} | {vm['mean_recall']:.6f} | {vm['predicted_foreground_percent']:.3f} | {vm['GT_foreground_percent']:.3f} | {v['minimum_dice']:.6f} |")
lines+=['','The same validation images were selected before any scoring, with seed 707, from the existing grouped '
        'validation partition. Its group keys do not overlap training group keys. Anonymous covid_ IDs are treated '
        'as unique by the existing splitter; this does not verify unknown real-world patient identity. Threshold stays '
        '0.5. These results measure changes on these 32 held-out images only. They are noisy, are not a full '
        'validation benchmark and do not establish broad generalization or reproduction. The validation partition '
        'was already used in previous project diagnostics; this is not a new untouched test cohort. Per the user continuation '
        'request, validation performance and reliability now guide normalization selection; the historical training-only '
        'decision rule is superseded in DECISION_AMENDMENT.md. No validation-based checkpoint or threshold selection '
        'occurred: all checkpoints are final equal-budget states and threshold is 0.5. Precision and recall in the table '
        'are macro means. validation_comparison.csv also provides separately labeled global pixel-pooled metrics and '
        'confusion counts. Foreground percentages are image-pixel proportions, not lesion count. Target empty-mask counts and foreground-area ranges '
        'are recorded in independent_verification.json; empty/empty Dice and IoU are defined as 1.','',
        '## Decision and bounded pilot recommendation','',
        f"Selected diagnostic configuration: **{candidate}**. Following the user continuation request, select the "
        'highest fixed-32 validation mean Dice among completed, numerically stable variants passing the final two '
        'all-image fitting checks (mean >=0.90 and minimum >=0.75). Train/eval consistency and memory provide '
        'additional reliability evidence. Training Dice does not determine ranking. '
        f"The raw highest-minimum variant is **{raw_best}**. "
        'If none pass, the highest-validation variant is only an unqualified candidate and pilot readiness remains false.','',
        f"Ready for a separately approved bounded pilot: **{decision['ready_for_bounded_pilot']}**.",'']
for name in NAMES:
    lines.append(f"- {name}: all-image criterion passed = {selection[name]['mean_minimum_criterion_passed']}.")
lines+=['',f"Frozen BN validation Dice exceeds baseline BN by {summaries['B_frozen_bn']['validation_32']['mean_dice']-summaries['A_current_bn']['validation_32']['mean_dice']:.6f} "
        f"and GroupNorm by {summaries['B_frozen_bn']['validation_32']['mean_dice']-summaries['C_groupnorm']['validation_32']['mean_dice']:.6f} on this subset. "
        'These are observed differences, not a significance claim. Frozen BN preserves the architecture and has exactly '
        'matching optimization/evaluation predictions. GroupNorm fits the eight images well but does not outperform '
        'frozen BN on this validation check. Baseline BN has a large train/evaluation discrepancy, including the '
        'small-lesion image; frozen BN removes this observed mismatch without claiming all remaining errors are normalization-related.']
if decision['ready_for_bounded_pilot']:
    lines+=['','Recommend two epochs initially, not five immediately, with the selected normalization configuration, '
            'the same stable FP32/microbatch-one settings, full grouped validation monitoring, streaming metrics/histograms '
            'and atomic checkpoints. Initialize the generalization pilot deliberately from original epoch 18 (perform '
            'the documented normalization conversion if GN is chosen), with a documented optimizer initialization. '
            'Do not silently reuse eight-image-overfit weights as the reproduction baseline. Extend only with validation '
            'and resource evidence. Any later threshold calibration uses validation only; test remains untouched.']
else:
    lines+=['','Do not start the full-dataset pilot yet. Resolve the remaining all-image fitting/consistency failure with '
            'a separately approved bounded diagnostic. No learning-rate/loss/augmentation sweep has been performed '
            'in this normalization comparison.']
lines+=['','Expected impact on held-out data is a **hypothesis**: consistent normalization may help microbatch-one '
        'training, but eight-image fitting cannot establish it. These results are not comparable to the paper’s '
        'Dice 0.8978/Jaccard 0.8145 benchmark.','',
        '## Preservation and outputs','',
        f"Post-run SHA-256 verification checked **{len(before)} protected files**, all unchanged. Inventory covers the "
        'six original notebooks, original source/checkpoints/evaluation outputs, all previous Phase 1/1B files and '
        'selected raw training image/mask files. Dataset operations are read only.','',
        'See comparison.csv, validation_comparison.csv, decision.json, independent_verification.json, preservation_verification.json, '
        'GPU_resource_summary.json, PROTOCOL.md and the per-variant source/config/log/checkpoint/prediction artifacts. '
        'Each controller log records real Python subprocess return codes.','',
        '**Phase 1C is stopped. Await user approval before any further training.**']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
write('comparison.json',{'variants':comparison,'selection':selection,'decision':decision})
print(json.dumps({'comparison':comparison,'decision':decision,'preservation':preservation},indent=2),flush=True)
