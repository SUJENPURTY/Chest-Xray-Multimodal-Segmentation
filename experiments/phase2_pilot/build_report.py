"""CPU-only validation/preservation audit and standalone figures. No training or test loading."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
os.environ['MPLCONFIGDIR']=str(HERE/'cache/matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(name): return json.loads((HERE/name).read_text(encoding='utf-8'))
def rows(name):
    with (HERE/name).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def write(name,obj): (HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024*1024): h.update(b)
    return h.hexdigest()
def table(name,values):
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)

assert (HERE/'summary.json').exists(),'Do not report interrupted training as complete'
s=read('summary.json');cfg=read('config.json');manifest=read('data_manifest.json')
parent=read('parent_verification.json')
assert parent['direct_GPU_construction_verified'] and parent['matches_phase1C_initial_parameters']
assert parent['baseline_probability_max_difference_vs_phase1C']<=2e-5
assert s['diagnostic_completed'] and s['completed_epochs']==2 and s['committed_microsteps']==11444 and s['optimizer_steps']==1432
assert not s['test_evaluated'] and not s['extended_beyond_two_epochs'] and s['BN_buffers_unchanged']
h=rows('training_microsteps.csv');orders=read('epoch_orders.json')
assert len(h)==11444 and [int(r['global_microstep']) for r in h]==list(range(1,11445))
assert all(r['image']==manifest['train'][int(r['train_index'])]['name'] for r in h)
updates=[r for r in h if int(r['update_committed'])]
assert len(updates)==1432 and [int(r['optimizer_steps']) for r in updates]==list(range(1,1433))
assert all(int(r['nonfinite_gradient_tensors'])==0 and int(r['gradient_tensors'])>0 for r in updates)
assert all(math.isfinite(float(r['loss'])) and math.isfinite(float(r['gradient_norm_before_clip'])) for r in h)
for epoch in [1,2]:
    hh=[r for r in h if int(r['epoch'])==epoch]
    assert len(hh)==5722 and [int(r['train_index']) for r in hh]==orders[str(epoch)]
    assert set(int(r['train_index']) for r in hh)==set(range(5722))
    assert int(hh[-1]['window_size'])==2 and int(hh[-1]['update_committed'])==1
    assert sum(int(r['update_committed']) for r in hh)==716
for i,r in enumerate(updates):
    factor=1-.5*i/1432
    assert abs(float(r['lr_vision'])-1e-5*factor)<1e-15
    assert abs(float(r['lr_decoder'])-1e-4*factor)<1e-15
assert s['skipped_updates']==0

comparison=[];checks={};labels=['baseline','epoch_1','epoch_2']
for label in labels:
    v=read(f'validation/{label}/summary.json');rr=rows(v['per_image_csv'])
    assert len(rr)==1423 and [r['image'] for r in rr]==[r['name'] for r in manifest['validation']]
    assert sha(HERE/v['per_image_csv'])==v['per_image_csv_sha256']
    for r in rr:
        tp,fp,fn,tn=[int(r[k]) for k in ['tp','fp','fn','tn']]
        pixels=tp+fp+fn+tn;assert pixels==224*224
        expected={'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'predicted_foreground_fraction':(tp+fp)/pixels,'true_foreground_fraction':(tp+fn)/pixels}
        assert all(abs(float(r[k])-value)<1e-12 for k,value in expected.items())
    for k in ['dice','iou','precision','recall','loss']:
        assert abs(statistics.mean(float(r[k]) for r in rr)-v[f'mean_{k}'])<1e-12
    verified_reps=0
    for index in v['representative_indices']:
        p=HERE/v['prediction_directory']/f'sample_{index:04d}'
        prob=np.load(p/'00_probability.npy')
        with Image.open(p/'00_ground_truth.png') as f: gt=np.asarray(f)>0
        with Image.open(p/'00_prediction.png') as f: pred=np.asarray(f)>0
        with Image.open(ROOT/manifest['validation'][index]['mask']) as f: raw=np.asarray(f)>0
        assert np.array_equal(gt,raw) and np.array_equal(pred,prob>=.5)
        assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
        count=[int(z.sum()) for z in [pred&gt,pred&~gt,~pred&gt,~pred&~gt]]
        assert count==[int(rr[index][k]) for k in ['tp','fp','fn','tn']]
        verified_reps+=1
    checks[label]={'all_per_image_confusion_metrics_verified':True,'representative_masks_verified':verified_reps,
        'validation_images':1423,'original_GT_orientation_verified':True}
    comparison.append({'label':label,**{k:v[k] for k in ['mean_dice','mean_iou','mean_precision','mean_recall',
        'predicted_foreground_percent','GT_foreground_percent','minimum_dice','weakest_image','mean_loss',
        'global_dice','global_iou','global_precision','global_recall','tp','fp','fn','tn','validation_seconds']}})
table('validation_comparison.csv',comparison);write('validation_comparison.json',comparison)

latest=read('checkpoints/latest.json');best=read('checkpoints/best.json')
for label,pointer in [('latest',latest),('best',best)]:
    path=HERE/pointer['path'];assert path.stat().st_size==pointer['bytes'] and sha(path)==pointer['sha256']
verifications=[read(p.name) for p in HERE.glob('checkpoint_verification_*.json')]
assert verifications and all(v['verified'] for v in verifications)
assert any(v['step_counters']==[1432.] for v in verifications)
resumes=[read(p.name) for p in HERE.glob('resume_*.json')]
controller=[json.loads(line) for line in (HERE/'controller_log.jsonl').read_text().splitlines()]
assert controller[-1]['returncode']==0
process_restart=any(r['returncode']==75 for r in controller) and bool(resumes) and all(v['verified'] for v in resumes)
assert process_restart,'The planned controlled process resume must be verified'

palette={'baseline':'#526A85','epoch_1':'#C48924','epoch_2':'#276B9A'}
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,metric in zip(axes,['mean_dice','mean_iou']):
    values=[r[metric] for r in comparison]
    bars=ax.bar(['Original\nepoch 18','Pilot\nepoch 1','Pilot\nepoch 2'],values,color=[palette[l] for l in labels],edgecolor='#30343B',linewidth=.7)
    ax.set(ylim=(0,1),ylabel=metric.replace('_',' ').title())
    ax.grid(axis='y',color='#E3E5E8',linewidth=.7);ax.set_axisbelow(True)
    for b,v in zip(bars,values): ax.text(b.get_x()+b.get_width()/2,v+.02,f'{v:.6f}',ha='center',fontsize=10)
fig.suptitle('Full validation: the same 1,423 images, threshold 0.5, per-image macro means')
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(HERE/'full_validation_dice_iou.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,8))
for epoch in [1,2]:
    hh=[r for r in h if int(r['epoch'])==epoch]
    bins=[hh[i:i+100] for i in range(0,len(hh),100)]
    x=[int(z[-1]['global_microstep']) for z in bins]
    for ax,key in zip(axes[0],['loss','dice']):
        ax.plot(x,[statistics.mean(float(r[key]) for r in z) for z in bins],label=f'Epoch {epoch}',
            color=palette[f'epoch_{epoch}'],linestyle='-' if epoch==1 else '--')
axes[0,0].set(ylabel='Mean BCE + soft Dice loss',xlabel='Committed training images')
axes[0,1].set(ylabel='Online pre-update mean Dice',xlabel='Committed training images',ylim=(0,1))
axes[1,0].plot([int(r['optimizer_steps']) for r in updates],[float(r['gradient_norm_before_clip']) for r in updates],color='#526A85')
axes[1,0].axhline(1,color='#30343B',linestyle='--',label='Clip threshold 1')
axes[1,0].set(yscale='log',ylabel='Pre-clipping gradient L2 norm (log scale)',xlabel='Optimizer updates')
axes[1,1].plot([int(r['optimizer_steps']) for r in updates],[float(r['lr_decoder']) for r in updates],color='#276B9A',label='Decoder / other')
axes[1,1].plot([int(r['optimizer_steps']) for r in updates],[float(r['lr_vision']) for r in updates],color='#C48924',linestyle='--',label='Vision encoder')
axes[1,1].set(ylabel='Learning rate',xlabel='Optimizer updates')
for ax in axes.flat: ax.grid(color='#E3E5E8',linewidth=.7);ax.legend(fontsize=9)
fig.suptitle('Two-epoch pilot training: 100-image bins; training Dice is not a validation score')
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(HERE/'training_progression.png',dpi=160);plt.close(fig)
selected=read('representative_manifest.json')['validation_indices'][:3]
for index in selected:
    fig,axes=plt.subplots(3,4,figsize=(12,9))
    for i,label in enumerate(labels):
        v=read(f'validation/{label}/summary.json');rr=rows(v['per_image_csv'])
        for j,suffix in enumerate(['xray','ground_truth','prediction','overlay']):
            p=HERE/v['prediction_directory']/f'sample_{index:04d}/00_{suffix}.png'
            with Image.open(p) as f: axes[i,j].imshow(np.asarray(f),cmap='gray' if j<3 else None)
            axes[i,j].axis('off');axes[i,j].set_title(label if j==0 else (f'Dice {float(rr[index]["dice"]):.4f}' if j==2 else suffix.replace('_',' ')))
    fig.suptitle('Fixed representative validation image: '+manifest['validation'][index]['name'],fontsize=10)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(HERE/f'representative_comparison_{index:04d}.png',dpi=140);plt.close(fig)

gpu=rows('gpu_monitor.csv')
resources={'peak_allocated_mib':max(float(r['peak_allocated_mib']) for r in h),
    'peak_reserved_mib':max(float(r['peak_reserved_mib']) for r in h),
    'whole_GPU_sampled_peak_mib':max(float(r['device_used_mib']) for r in gpu),
    'minimum_GPU_free_mib':min(float(r['device_free_mib']) for r in gpu),
    'minimum_available_RAM_mib':min(float(r['available_ram_mib']) for r in gpu),
    'minimum_available_commit_mib':min(float(r['available_commit_mib']) for r in gpu),
    'training_compute_seconds':sum(float(r['compute_seconds']) for r in h),
    'validation_seconds':sum(r['validation_seconds'] for r in comparison),
    'runner_wall_seconds_including_imports_and_controlled_restart':sum(r['wall_seconds'] for r in controller),
    'checkpoint_write_seconds':sum(float(r['write_seconds']) for r in rows('checkpoint_log.csv'))}
write('resource_summary.json',resources)

before=read('preservation_before.json');changed=[]
for i,(name,item) in enumerate(before.items()):
    p=ROOT/name
    if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']: changed.append(name)
    if i%100==0: print('POST-PILOT PROTECTED HASH CHECK',i,'/',len(before),flush=True)
metadata=read('dataset_metadata_before.json');data_changed=[];seen=set()
def scan_dataset(base):
    with os.scandir(base) as entries:
        for entry in entries:
            if entry.is_dir(): scan_dataset(entry.path)
            elif entry.is_file():
                stat=entry.stat();name=os.path.relpath(entry.path,ROOT);seen.add(name)
                expected=metadata.get(name)
                if expected is None or stat.st_size!=expected['bytes'] or stat.st_mtime_ns!=expected['mtime_ns']: data_changed.append(name)
for base in [ROOT/'_qata_compatible',ROOT/'Project/LanGuideMedSeg-MICCAI2023/Data',ROOT/'text_annotation']:
    if base.is_dir(): scan_dataset(base)
data_changed.extend(set(metadata)-seen)
preservation={'protected_files_hashed':len(before),'all_protected_files_unchanged':not changed,'changed_or_missing':changed,
    'dataset_files_metadata_checked':len(metadata),'dataset_metadata_unchanged':not data_changed,
    'dataset_changed_or_missing':data_changed,'dataset_pixel_bytes_all_newly_hashed':False}
write('preservation_verification.json',preservation);assert not changed and not data_changed
write('independent_verification.json',{'all_checks_passed':True,'CPU_only':True,'train_samples_once_per_epoch':True,
    'final_two_sample_window_verified':True,'scheduler_sequence_verified':True,'finite_gradient_updates':1432,
    'checkpoint_integrity_verified':True,'strict_reload_verified':True,'actual_process_restart_verified':process_restart,
    'watched_parameter_nonzero_updates':sum(float(r['watched_parameter_max_update'])>0 for r in updates),
    'validation':checks,'test_evaluated':False})

b,e1,e2=comparison
dice_consistent=e1['mean_dice']>b['mean_dice'] and e2['mean_dice']>e1['mean_dice']
iou_consistent=e1['mean_iou']>b['mean_iou'] and e2['mean_iou']>e1['mean_iou']
consistent=dice_consistent and iou_consistent
stable=s['OOM_events']==0 and s['nonfinite_events']==0
extend=consistent and stable
decision={'baseline_dice':b['mean_dice'],'epoch1_dice':e1['mean_dice'],'epoch2_dice':e2['mean_dice'],
    'consistent_validation_Dice_improvement':dice_consistent,'consistent_validation_IoU_improvement':iou_consistent,
    'numerically_stable':stable,'extension_justified_for_review':extend,
    'recommended_next_action':'Seek approval for a separately bounded continuation toward at most five total pilot epochs; monitor full validation.' if extend else
        'Investigate validation deterioration or instability before approving more training; retain best validated pilot and original baseline.',
    'best_pilot_epoch':best['epoch'],'baseline_still_better_than_best_pilot':b['mean_dice']>=best['validation_mean_dice'],
    'extension_started':False,'test_evaluated':False,'paper_reproduction_claimed':False,'further_training_requires_approval':True}
write('decision.json',decision)
tr=[read(f'training_epoch_{i}.json') for i in [1,2]]
lines=['# Phase 2: bounded Frozen-BatchNorm pilot','',
    '**Exactly two additional epochs completed.** All 5,722 training images were used once each per epoch. '
    'Full validation used the identical 1,423 images for original epoch 18 and both pilot epochs. '
    'No test inference, threshold calibration, PowerPoint generation or automatic extension occurred.','',
    '## Strictly comparable full-validation results','',
    '| Weights | Mean Dice | Mean IoU | Mean precision | Mean recall | Predicted foreground % | GT foreground % |',
    '|---|---:|---:|---:|---:|---:|---:|']
for r in comparison: lines.append(f"| {r['label']} | {r['mean_dice']:.6f} | {r['mean_iou']:.6f} | {r['mean_precision']:.6f} | {r['mean_recall']:.6f} | {r['predicted_foreground_percent']:.3f} | {r['GT_foreground_percent']:.3f} |")
lines+=['','Primary metrics are per-image macro means at threshold **0.5**, not global pixel-pooled Dice. '
    'validation_comparison.csv supplies both separately. Empty/empty Dice and IoU equal one; undefined precision/recall equal zero. '
    'The earlier fixed-32 scores are not substituted for these full-validation results.','',
    f"Observed mean-Dice changes: epoch 1 minus baseline = **{e1['mean_dice']-b['mean_dice']:+.6f}**; "
    f"epoch 2 minus baseline = **{e2['mean_dice']-b['mean_dice']:+.6f}**; epoch 2 minus epoch 1 = **{e2['mean_dice']-e1['mean_dice']:+.6f}**.",
    'These are observed validation differences for this single seeded run, not proof of population-level superiority.','',
    '## Training configuration and progression','',
    'Original epoch-18 best checkpoint strictly loaded; architecture unchanged. The configuration-only direct CUDA '
    'constructor avoids separate pretrained CPU weight loads. Its complete parameter hash and two baseline predictions '
    'were verified against Phase1C before training. Two earlier startup guards stopped before model construction; '
    'no training updates or OOM occurred in those attempts. All 18 BN buffers remain equal to their '
    'original values; affine weights train. Frozen text backbone stays in eval mode. Stable logits BCE + FP32 soft Dice, '
    'microbatch one, accumulation eight, no AMP/augmentation, zero loader workers and no whole-dataset tensor cache. '
    'Fresh AdamW foreach=False uses encoder 1e-5 / decoder-other 1e-4, one-third of Phase1C rates; clip norm one. '
    'A fresh per-update linear scheduler halves both rates over 1,432 updates. This is a pilot configuration, not a '
    'normalization-only causal comparison. Original image/mask orientation, resizing, normalization and pairing were retained.','',
    '| Epoch | Online mean loss | Online mean Dice | Online mean IoU | Updates | Skipped | Gradient norm min / max | Clipped updates |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for t in tr: lines.append(f"| {t['epoch']} | {t['mean_loss']:.6f} | {t['mean_dice']:.6f} | {t['mean_iou']:.6f} | {t['optimizer_steps']} | 0 | {t['gradient_norm_min']:.5f} / {t['gradient_norm_max']:.5f} | {t['clipped_updates']} |")
lines+=['','Training metrics are recorded at each pre-update forward pass; they do not represent a fixed-weight '
    'evaluation of the complete training set. Each final two-image accumulation window was averaged by two and '
    'updated once. There were no dropped/duplicated training images and no skipped updates. '
    'See training_microsteps.csv, training_epoch_1/2.json and training_progression.png.','',
    '## Resource measurements and reliability','',
    f"Peak training tensor allocation **{resources['peak_allocated_mib']:.2f} MiB**, reserved **{resources['peak_reserved_mib']:.2f} MiB**. "
    f"Sampled whole-GPU peak **{resources['whole_GPU_sampled_peak_mib']:.0f} MiB**, minimum free **{resources['minimum_GPU_free_mib']:.0f} MiB**. "
    'Whole-GPU samples include unrelated applications and occur every five seconds. Microbatch one is the measured setting.',
    f"Minimum host available RAM **{resources['minimum_available_RAM_mib']:.2f} MiB**; minimum available Windows commit "
    f"**{resources['minimum_available_commit_mib']:.2f} MiB**. No unrelated process was terminated.",
    f"Training compute/data-load time **{resources['training_compute_seconds']/60:.2f} minutes**; full-validation total "
    f"**{resources['validation_seconds']/60:.2f} minutes**; checkpoint write/hash total **{resources['checkpoint_write_seconds']/60:.2f} minutes**. "
    f"Runner wall time including imports and controlled restart **{resources['runner_wall_seconds_including_imports_and_controlled_restart']/60:.2f} minutes**. "
    'Preparation/preservation hashing and report generation are separate from runner wall time.',
    f"Recorded OOM events: **{s['OOM_events']}**; non-finite events: **{s['nonfinite_events']}**. All 1,432 logged gradient updates were finite.",
    'Atomic immutable periodic/epoch checkpoints preserve model, optimizer, scheduler, RNG, deterministic orders and committed cursors. '
    'At 800 images, the first checkpoint was strictly reloaded with exact probe predictions; the process exited deliberately '
    'and the controller launched --resume. The new process verified SHA/config/order/predictions and resumed at position 800. '
    'No budget restart or additional epochs were used. Final strict reload, optimizer counters, scheduler and simulated '
    'interrupted write also passed. CPU scalar states are cloned after mmap restore; no full CPU model copy is retained.','',
    f"Best pilot checkpoint is epoch **{best['epoch']}**, selected only using full-validation macro Dice. "
    f"Original baseline still better than the best pilot: **{decision['baseline_still_better_than_best_pilot']}**. "
    'Latest and best pointers reference immutable files under checkpoints/. The original checkpoints remain unchanged.','',
    '## Decision and remaining limitations','',decision['recommended_next_action'],'',
    'The validation partition has been used for diagnostics and selection. It is not an untouched test cohort. '
    'Group keys match the existing split and are disjoint for training/validation; anonymous covid_ IDs do not verify '
    'unknown real patient identity. No across-seed robustness or paper-protocol-equivalent test benchmark has been measured. '
    'Attention backward can be nondeterministic despite fixed seeds/order; exact checkpoint inference reload parity was verified. '
    'The paper score 0.8978 is not a reproduced result. Expected benefit from extending training remains a hypothesis.','']
if not consistent:
    lines+=['Validation did not improve monotonically. Before additional training, inspect per-image regressions, foreground/recall '
        'changes, clipped-gradient frequency and full-validation losses. Compare best pilot and baseline using the same protocol. '
        'Do not assume more epochs will repair deterioration; a later LR or normalization-statistics study must be explicitly controlled.','']
lines+=['## Preservation and artifacts','',
    f"Post-run SHA256 comparison verified **{len(before)} protected original/earlier-experiment files**, all unchanged. "
    f"Dataset size/mtime metadata checked **{len(metadata)} files**, unchanged; dataset pixels were not all newly hashed. "
    'Original/Phase1/1B pre-run hashes reuse the independently verified end-of-Phase1C inventory; Phase1C artifacts were freshly hashed before the pilot. '
    'The original split manifest and annotation hashes are in config.json and protected by the inventory.',
    'Inspect validation_comparison.csv, full_validation_dice_iou.png, training_progression.png, representative_comparison_*.png, '
    'validation/*/per_image.csv, predictions/, checkpoints/latest.json, checkpoints/best.json, resource_summary.json, '
    'independent_verification.json, preservation_verification.json and decision.json.','',
    '**Stopped after two epochs. Await user approval before any extension or PowerPoint.**']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'comparison':comparison,'resources':resources,'decision':decision,'preservation':preservation},indent=2),flush=True)
