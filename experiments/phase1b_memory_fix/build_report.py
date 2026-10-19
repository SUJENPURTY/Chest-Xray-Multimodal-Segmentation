"""CPU-only independent verification and report generation after the GPU runner exits."""
import csv
import json
import os
from pathlib import Path
import math
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']=str(HERE/'cache/matplotlib')
os.environ['TEMP']=os.environ['TMP']=str(HERE/'tmp')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8')) if (HERE/name).exists() else None


def rows(name):
    if not (HERE/name).exists():
        return []
    with (HERE/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def verify_prediction_set(directory,table):
    results=[]
    for i,row in enumerate(table):
        with Image.open(directory/f'{i:02d}_ground_truth.png') as f:
            y=np.asarray(f)>0
        with Image.open(directory/f'{i:02d}_prediction.png') as f:
            p=np.asarray(f)>0
        with Image.open(ROOT/'_qata_compatible/Train/GTs'/row['image']) as f:
            raw=np.asarray(f)
        assert raw.shape==(224,224) and np.array_equal(y,raw>0)
        probability=np.load(directory/f'{i:02d}_probability.npy')
        assert np.isfinite(probability).all() and (probability>=0).all() and (probability<=1).all()
        assert np.array_equal(p,probability>=.5)
        tp,fp,fn,tn=[int(np.count_nonzero(a)) for a in [p&y,p&~y,~p&y,~p&~y]]
        dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
        iou=tp/(tp+fp+fn) if tp+fp+fn else 1.
        assert all(int(row[k])==v for k,v in zip(['tp','fp','fn','tn'],[tp,fp,fn,tn]))
        assert abs(float(row['dice'])-dice)<1e-12 and abs(float(row['iou'])-iou)<1e-12
        assert abs(iou-dice/(2-dice))<1e-12
        results.append({'image':row['image'],'dice':dice,'iou':iou,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                        'raw_mask_values':np.unique(raw).tolist(),'export_orientation_matches_raw_mask':True})
    return results


def montage(label,table):
    fig,axes=plt.subplots(8,4,figsize=(12,22),squeeze=False)
    for i,row in enumerate(table):
        for j,(suffix,title) in enumerate([
            ('xray','Original X-ray'),('ground_truth','Ground truth'),
            ('prediction',f"Prediction: Dice {float(row['dice']):.3f}"),('overlay','GT green / prediction red')]):
            with Image.open(HERE/f'predictions/{label}/{i:02d}_{suffix}.png') as opened:
                axes[i,j].imshow(np.asarray(opened),cmap='gray' if j<3 else None)
            axes[i,j].set_title(title,fontsize=9)
            axes[i,j].axis('off')
        axes[i,0].text(0,-.07,row['image'].removeprefix('mask_')[:44],transform=axes[i,0].transAxes,fontsize=8)
    fig.suptitle(f'Phase 1B: {label}, eight training images only, threshold 0.5',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.975))
    fig.savefig(HERE/f'{label}_comparison.png',dpi=120)
    plt.close(fig)


summary=read('summary.json')
failure=read('failure.json')
gpu=rows('gpu_monitor.csv')
training=rows('training_microsteps.csv')
curve=rows('evaluation_progression.csv')
resume_config=read('resume_config.json')
if resume_config and (HERE/'resume_training_microsteps.csv').exists():
    resume_step=resume_config['resume_checkpoint']['microstep']
    training=[r for r in rows('attempt01/training_microsteps.csv') if int(r['microstep'])<=resume_step]+rows('resume_training_microsteps.csv')
    curve=[r for r in rows('attempt01/evaluation_progression.csv') if int(r['microstep'])<=resume_step]+rows('resume_evaluation_progression.csv')
    with (HERE/'combined_training_microsteps.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(training[0]))
        w.writeheader()
        w.writerows(training)
    with (HERE/'combined_evaluation_progression.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
progress=[]
verification={'verification_is_cpu_only':True,'no_new_validation_or_test_inference':True}
if summary:
    final_table=rows('final_per_image.csv')
    verified=verify_prediction_set(HERE/'predictions/final',final_table)
    verification['final_per_image']=verified
    verification['final_mean_dice']=statistics.mean(r['dice'] for r in verified)
    verification['final_mean_iou']=statistics.mean(r['iou'] for r in verified)
    verification['final_minimum_dice']=min(r['dice'] for r in verified)
    for a,b in [('final_mean_dice','mean_dice'),('final_mean_iou','mean_iou'),('final_minimum_dice','minimum_dice')]:
        assert abs(verification[a]-summary['final'][b])<1e-12
    verify_prediction_set(HERE/'predictions/initial',rows('initial_per_image.csv'))
    best_table=rows('best_per_image.csv') if summary['best_microstep'] else rows('initial_per_image.csv')
    best_folder=f"step_{summary['best_microstep']:04d}" if summary['best_microstep'] else 'initial'
    verification['best_per_image']=verify_prediction_set(HERE/'predictions'/best_folder,best_table)
    montage('final',final_table)
    montage(best_folder,best_table)
    weakest_position=min(range(8),key=lambda i:float(final_table[i]['dice']))
    weak_probability=np.load(HERE/f'predictions/final/{weakest_position:02d}_probability.npy')
    with Image.open(HERE/f'predictions/final/{weakest_position:02d}_ground_truth.png') as f:
        weak_target=np.asarray(f)>0
    distributions={}
    fig,ax=plt.subplots(figsize=(9,5))
    for region,label in [(weak_target,'GT foreground'),(~weak_target,'GT background')]:
        values=weak_probability[region]
        distributions[label]={'pixels':int(values.size),'mean':float(values.mean()),
                              'quantiles_10_50_90':np.quantile(values,[.1,.5,.9]).tolist(),
                              'fraction_at_or_above_0_5':float((values>=.5).mean())}
        ax.hist(values,bins=np.linspace(0,1,51),density=True,alpha=.55,label=label)
    ax.axvline(.5,color='black',linestyle='--',label='Fixed threshold 0.5')
    ax.set(xlabel='Sigmoid probability',ylabel='Density',title=final_table[weakest_position]['image'])
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE/'weakest_probability_distribution.png',dpi=160)
    plt.close(fig)
    write('weakest_probability_distribution.json',{'image':final_table[weakest_position]['image'],
                                                  'threshold_selection_performed':False,'regions':distributions})
    last_weak_training=next(r for r in reversed(training) if int(r['sample_position'])==weakest_position)
    write('weakest_loss_mode_comparison.json',{
        'image':final_table[weakest_position]['image'],'last_training_microstep':int(last_weak_training['microstep']),
        'training_mode_soft_dice':1-float(last_weak_training['dice_loss']),
        'training_mode_BCE':float(last_weak_training['bce_loss']),
        'evaluation_mode_soft_dice':1-float(final_table[weakest_position]['dice_loss']),
        'evaluation_mode_BCE':float(final_table[weakest_position]['bce_loss']),
        'note':'Training forward preceded its optimizer update; the fixed-weight normalization comparison in weakest_image_analysis.json is stronger causal evidence'})
    update_rows=[r for r in training if int(r['gradient_tensors'])]
    assert len(update_rows)==summary['optimizer_updates']
    assert all(int(r['nonfinite_gradient_tensors'])==0 for r in update_rows)
    assert all(math.isfinite(float(r['loss'])) for r in training)
    verification['verified_training_updates']=len(update_rows)
    verification['nonfinite_gradient_updates']=0
    verification['watched_parameter_updates_nonzero']=sum(float(r['watched_parameter_max_update'])>0 for r in update_rows)
    verification['gradient_norm_range']=[min(float(r['gradient_norm_before_clip']) for r in update_rows),
                                         max(float(r['gradient_norm_before_clip']) for r in update_rows)]
    verification['training_loss_range']=[min(float(r['loss']) for r in training),max(float(r['loss']) for r in training)]
    verification['compute_seconds_per_microstep']=statistics.mean(float(r['compute_seconds']) for r in training)
    if resume_config:
        all_actual_rows=rows('attempt01/training_microsteps.csv')+rows('resume_training_microsteps.csv')
        actual_updates=[r for r in all_actual_rows if int(r['gradient_tensors'])]
        assert len(actual_updates)==summary['actual_executed_optimizer_updates']
        assert all(int(r['nonfinite_gradient_tensors'])==0 for r in actual_updates)
        verification['verified_actual_updates_including_discarded_work']=len(actual_updates)
        discarded_rows={int(r['microstep']):r for r in rows('attempt01/training_microsteps.csv')
                        if int(r['microstep'])>resume_config['resume_checkpoint']['microstep']}
        replayed=[r for r in rows('resume_training_microsteps.csv') if int(r['microstep']) in discarded_rows]
        verification['rollback_replay']={
            'compared_microsteps':len(replayed),
            'sample_order_matches':all(r['sample_position']==discarded_rows[int(r['microstep'])]['sample_position'] for r in replayed),
            'maximum_microbatch_loss_difference':max(abs(float(r['loss'])-float(discarded_rows[int(r['microstep'])]['loss'])) for r in replayed),
            'note':'Continuation check on the discarded interval, not a separate performance experiment; SDPA backward can be nondeterministic'}
    weak_diagnostic=read('weakest_image_analysis.json')
    verification['weakest_batch_statistics']=verify_prediction_set(
        HERE/'predictions/weakest_batch_statistics',
        [{'image':weak_diagnostic['image'],**weak_diagnostic['single_image_batch_statistics']}])
    verification['all_metrics_verified']=True
write('independent_verification.json',verification)

resource={
    'sample_count':len(gpu),'scope':'Whole device includes desktop and unrelated applications; 5-second sampling',
    'maximum_device_used_mib':max(float(r['device_used_mib']) for r in gpu) if gpu else None,
    'minimum_device_free_mib':min(float(r['device_free_mib']) for r in gpu) if gpu else None,
    'maximum_temperature_c':max(float(r['temperature_c']) for r in gpu) if gpu else None,
    'minimum_host_available_ram_mib':min(float(r['available_ram_mib']) for r in gpu) if gpu else None,
    'minimum_host_available_commit_mib':min(float(r['available_commit_mib']) for r in gpu) if gpu else None,
}
write('gpu_monitor_summary.json',resource)
if curve:
    x=[int(r['microstep']) for r in curve]
    fig,ax=plt.subplots(figsize=(9,5))
    for key,label in [('mean_dice','Mean Dice'),('mean_iou','Mean IoU'),('minimum_dice','Minimum image Dice')]:
        ax.plot(x,[float(r[key]) for r in curve],marker='.',label=label)
    ax.axhline(.9,color='gray',linestyle=':',label='Mean sanity criterion 0.90')
    ax.set(xlabel='Microsteps (8 per optimizer update)',ylabel='Eight-training-image score at threshold 0.5',ylim=(0,1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE/'training_progression.png',dpi=160)
    plt.close(fig)
if training:
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    x=[int(r['microstep']) for r in training]
    axes[0].plot(x,[float(r['loss']) for r in training],linewidth=.5,label='Microbatch loss')
    axes[0].set_ylabel('BCEWithLogits + FP32 soft Dice')
    axes[0].legend()
    axes[1].plot(x,[float(r['allocated_mib']) for r in training],label='After-step allocated')
    axes[1].plot(x,[float(r['reserved_mib']) for r in training],label='Reserved')
    axes[1].plot(x,[float(r['peak_allocated_mib']) for r in training],label='Cumulative allocated peak',linestyle=':')
    axes[1].set(xlabel='Microstep',ylabel='PyTorch GPU MiB')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(HERE/'loss_and_gpu_memory.png',dpi=160)
    plt.close(fig)

old_failure=json.loads((ROOT/'experiments/phase1_diagnostics/failure.json').read_text())
old_host=json.loads((ROOT/'experiments/phase1_diagnostics/host_memory_after_failure.json').read_text())
oom_analysis={
    'confirmed_failure_stage':'Backward, run_phase1.py line 483, (loss / 8).backward()',
    'historical_kernel_localization':'Not established: asynchronous CUDA reporting warning in saved traceback',
    'saved_phase1_torch_memory':old_failure['memory'],
    'post_failure_host_ram_load_percent':old_host['load_percent'],
    'confirmed_unnecessary_retention':'Full CPU best_state clone at every improved check; previous clone stays alive while replacement is built',
    'autograd_graph_leak_confirmed':False,
    'evidence':'No retain_graph/create_graph in old runner; per-step hook references cleared; GPU allocated peak bounded in saved logs',
    'root_cause_hypothesis':'Combined WDDM/system RAM pressure, redundant model snapshots and external/runtime allocations; not proven by old logs',
    'phase1b_controls':'One ML process, no full CPU snapshots, plain tensors after element equality checks, tested attention outputs/gradients, inference_mode, explicit release, synchronous CUDA stage attribution',
}
write('oom_analysis.json',oom_analysis)

startup=read('attempt01/startup_resources.json') or read('startup_resources.json')
preservation=read('preservation_verification.json')
persistence=read('checkpoint_persistence_verification.json')
optimization=read('memory_optimization_verification.json')
weak=read('weakest_image_analysis.json')
ownership=read('checkpoint_mmap_ownership_fix.json')

lines=['# Phase 1B — GPU memory fix and eight-image FP32 diagnostic','',
       'This is a training-subset sanity experiment. It does not measure generalization or reproduce the paper. '
       'No validation or test inference, threshold search, or full-dataset training was performed. All new files are in this directory.','',
       '## Outcome','']
if summary:
    lines += [f"The bounded diagnostic completed: **{summary['completed_microsteps']} microsteps / {summary['optimizer_updates']} optimizer updates**, "
              f"with **zero OOM events and zero nonfinite gradient updates**. Stop reason: {summary['stop_reason']}.",'',
              '| Population / checkpoint | Mean Dice | Mean IoU | Minimum image Dice |',
              '|---|---:|---:|---:|',
              '| Historical original test (reference only; not rerun) | 0.185754 | 0.106600 | — |',
              '| Historical original validation, threshold 0.5 | 0.196050 | 0.113549 | — |',
              '| Historical validation, calibrated threshold 0.275 | 0.307530 | 0.194053 | — |',
              '| Phase 1 last persisted eight-image check (weights lost) | 0.869623 | 0.827750 | 0.242370 |']
    for label,data in [('Phase 1B initial epoch 18',summary['initial']),
                       (f"Phase 1B best persisted mean, step {summary['best_microstep']}",summary['best']),
                       ('Phase 1B final',summary['final'])]:
        lines.append(f"| {label} | {data['mean_dice']:.6f} | {data['mean_iou']:.6f} | {data['minimum_dice']:.6f} |")
    lines += ['', 'All Phase 1B scores are macro means over the same eight training images at fixed threshold 0.5. '
              'The validation-selected threshold 0.275 was not applied to changed weights. '
              'The published Dice 0.8978 / Jaccard 0.8145 are not comparable with eight-image fitting scores.','',
              'Phase 1B includes more optimizer updates than the interrupted Phase 1 run. Higher subset scores are real '
              'measurements, but this comparison does not isolate a segmentation benefit from the memory changes themselves. '
              'The memory controls enabled a longer stable diagnostic; held-out improvement remains unmeasured.','',
              '## Historical OOM diagnosis','',
              'Confirmed: the saved exception arose at backward propagation (`../phase1_diagnostics/run_phase1.py:483`). '
              'It did not arise at checkpoint export or visualization. The traceback warns that CUDA errors can surface asynchronously, '
              'so the first failing CUDA operation cannot be determined retrospectively.','',
              'At failure PyTorch reported 1,267.65 MiB allocated, 1,364.10 MiB peak allocated, and 1,406 MiB reserved. '
              'The old whole-device monitor peaked at 2,352 MiB. Windows physical RAM was 96% used after the failure '
              '(256.27 MiB free); this is a post-failure observation, not an exact failure-time measurement.','',
              'No retained autograd graph leak was established. The old runner used no `retain_graph=True`, cleared the output hook, '
              'and released step tensors. It did unnecessarily retain full CPU model copies: the original state plus `best_state` '
              'and a replacement clone being constructed on improved evaluations. Saving only at normal completion caused the loss '
              'of learned weights.','',
              '**Hypothesis, not a confirmed historical cause:** WDDM/system RAM pressure, redundant CPU snapshots and '
              'external/runtime GPU allocations contributed to the OOM. The controlled successful run supports the reliability of the '
              'new harness; it cannot isolate which historical factor caused that failure. See `oom_analysis.json`.','',
              '## Controlled checkpoint restart and retained mapping fix','',
              'The first Phase 1B process was deliberately stopped after verifying that checkpoint 240 was committed. '
              'Only this experiment process was stopped; unrelated processes were untouched. Its complete source/log evidence '
              'is preserved in `attempt01/`, and all committed checkpoint files remain intact. Up to 80 later microsteps were '
              'uncommitted and were rolled back, rather than claiming their weights survived.','',
              'A confirmed ownership problem was found in checkpoint reload: PyTorch AdamW returns CPU `step` tensors '
              'unchanged (`optimizer.py:756-760`), while mmap deserialization slices one overall file mapping '
              '(`serialization.py:1883-1885`). These small retained counters can therefore keep the whole ~979 MB checkpoint '
              'mapping alive. This is a checkpoint reload issue discovered in Phase 1B; it is not proof of the original '
              'Phase 1 backward OOM cause.','',
              'The resumed loader clones only CPU optimizer scalars and RNG tensors before releasing the loaded mapping; '
              'model and GPU optimizer moment tensors are not duplicated in full on the CPU. Independent storage is '
              'asserted. `checkpoint_mmap_ownership_fix.json` records the tensor count, bytes and before/after host resources. '
              '`resume_verification.json` checks exact saved prediction parity and restored optimizer counters.','',
              f"Only **{ownership['CPU_state_tensors_cloned']} CPU tensors / {ownership['CPU_state_bytes_cloned']} bytes** "
              f"were cloned. Available system commit rose from **{ownership['resources_mapping_alive']['available_commit_mib']:.1f} "
              f"to {ownership['resources_mapping_released']['available_commit_mib']:.1f} MiB** around mapping release; "
              'these whole-system observations can also fluctuate with other applications. The independent-storage assertion '
              'establishes the ownership fix even without attributing every resource change to it.','',
              f"Committed trajectory: **{summary.get('completed_microsteps')} microsteps / {summary['optimizer_updates']} updates**. "
              f"Actual executed work including discarded steps: **{summary.get('actual_executed_microsteps',summary['completed_microsteps'])} "
              f"microsteps / {summary.get('actual_executed_optimizer_updates',summary['optimizer_updates'])} updates**. "
              'The resumed cap subtracts discarded work from 2,000, so the total actual execution remains bounded. '
              'The resumed active loop has a 20-minute limit; the first attempt was also bounded.','',
              'After restart, checkpoints are saved every 160 microsteps and at completion. Best reported weights are '
              'selected among persisted states. All 80-step metric checks remain in the history, including any higher '
              'unsaved observation; a scalar metric is not presented as an available checkpoint.','',
              '## Memory controls and functional checks','',
              'The runner uses one ML process and one GPU model, memory-mapped epoch-18 loading, FP32 throughout, microbatch 1, '
              'eight-step gradient accumulation, and no AMP. Frozen CXR-BERT stays in evaluation mode; decoder BatchNorm retains '
              'the old training/evaluation behavior for this controlled comparison. Unused attention weights are disabled only '
              'after a matched full-gradient/output comparison. MONAI MetaTensors are converted to plain tensors after exact '
              'element equality assertions. The architecture and original source files are unchanged; all newly learned '
              'weights are stored only in the isolated experiment directory.','',
              f"Attention verification: output max absolute change **{optimization['output_max_abs_difference']:.3g}**, "
              f"loss change **{optimization['loss_abs_difference']:.3g}**, relative L2 gradient difference "
              f"**{optimization['all_gradient_relative_l2_difference']:.3g}** across {optimization['gradient_tensors']} gradient tensors. "
              f"Optimization adopted: **{optimization['optimization_adopted']}**. See `memory_optimization_verification.json`.",'',
              'PyTorch emitted a warning that the memory-efficient attention backward algorithm may be nondeterministic '
              'under `deterministic_algorithms(warn_only=True)`. Seeded sample order and checkpoint RNG state are preserved, '
              'but bitwise-identical retraining is not promised. The fixed-weight output/gradient comparison passed within '
              'the predeclared tolerances; checkpoint prediction reload parity is checked separately.','',
              'The PowerShell training wrapper returned status 1 after formatting that stderr warning as a '
              '`NativeCommandError`; a separate Python exit status was not captured. There is no saved failure '
              'traceback. The completion record, final checkpoint/reload checks, ending resource snapshot and '
              'preservation verification all exist, and CPU-only independent report verification exited successfully. '
              'Completion is established from those artifacts rather than claiming a clean wrapper exit.','',
              'Curve Dice/IoU and per-image metrics are measured in `model.eval()`; microstep losses use training-mode '
              'decoder BatchNorm with frozen BERT in eval mode. These modes can have very different losses on an outlier. '
              '`weakest_loss_mode_comparison.json` records this distinction; the final fixed-weight inference diagnostic '
              'provides the direct normalization comparison.','',
              'Evaluation and weakest-image forwards run under `torch.inference_mode()`. PNG creation uses detached CPU arrays '
              'after inference. Hook tensors and step tensors are cleared explicitly; gradients are zeroed after each update; '
              'allocator caches are released between independent stages. `CUDA_LAUNCH_BLOCKING=1` and synchronization provide '
              'stage attribution in any new failure. No unrelated process was terminated.','',
              '| Measured resource | Value |','|---|---:|',
              f"| GPU | GTX 1650, {startup['device_total_mib']:.0f} MiB |",
              f"| Before ML imports: device used / free | {startup['device_used_mib']:.0f} / {startup['device_free_mib']:.0f} MiB |",
              f"| Before imports: physical RAM free / commit available | {startup['available_ram_mib']:.1f} / {startup['available_commit_mib']:.1f} MiB |",
              f"| Training/evaluation/checkpoint PyTorch peak allocated | {summary['peak_allocated_mib']:.2f} MiB |",
              f"| PyTorch peak reserved | {summary['peak_reserved_mib']:.2f} MiB |",
              f"| Whole GPU sampled peak used / minimum free | {resource['maximum_device_used_mib']:.0f} / {resource['minimum_device_free_mib']:.0f} MiB |",
              f"| Minimum sampled RAM free / commit available | {resource['minimum_host_available_ram_mib']:.1f} / {resource['minimum_host_available_commit_mib']:.1f} MiB |",
              f"| Peak sampled temperature | {resource['maximum_temperature_c']:.0f} °C |",
              f"| Mean compute time per microstep | {summary['compute_seconds_per_microstep']:.3f} seconds |",
              f"| Resumed training wall time including exports/reload checks | {summary['training_wall_seconds_including_exports']:.1f} seconds |",'',
              'Whole-device values are sampled every five seconds and include desktop and other applications. PyTorch peaks '
              'are allocator measurements, not total physical VRAM. '
              'The startup process inventory lists Brave, Chrome, PowerToys, WhatsApp, Toolkit and Windows Explorer on the GPU. '
              'WDDM reports their individual memory as N/A, so their allocations cannot be assigned per process. '
              'Physical RAM was already severely constrained before launch; the run can page to the Windows commit pool. '
              'A subsequent pilot should start with more host RAM headroom and '
              'at least the observed GPU requirement plus margin. Do not infer that physical batch size 32 will fit.','',
              '## Checkpoint persistence','',
              'Model, optimizer, Python/NumPy/PyTorch/CUDA RNG states, exact sample manifest/configuration and update counters are '
              'saved at optimizer boundaries. Initial, improved, periodic and final checkpoints are immutable files. Writes go '
              'to a same-directory temporary file, are flushed/fsynced, then committed with `os.replace`. The latest/best pointer '
              'JSON is updated only after a valid checkpoint exists. A partial write leaves the prior committed checkpoint intact. '
              'Training microsteps are appended immediately and evaluations are appended every 80 microsteps. Predictions are '
              'exported at safely committed checkpoints. Full CPU model snapshots are avoided.','',
              f"Final checkpoint: `{summary['final_checkpoint']['path']}`; SHA-256 `{summary['final_checkpoint']['sha256']}`. "
              f"Best mean checkpoint: `{summary['best_checkpoint']['path']}`.",'',
              f"Strict reload and all-eight-image prediction parity verified: **{persistence['verified']}**. "
              f"Maximum prediction difference **{persistence['maximum_prediction_difference_after_reload']:.3g}**; "
              f"optimizer step counters **{persistence['optimizer_step_counters']}**; moments finite **{persistence['optimizer_state_tensors_finite']}**. "
              f"Simulated interrupted write preserved the committed checkpoint: **{persistence['interrupted_write_preserved_committed_checkpoint']}**. "
              'This checks a pre-replace interruption, not hardware power-loss durability. See `checkpoint_persistence_verification.json`.','',
              'The deliberately interrupted `periodic_step_0320.ckpt.tmp` remains as partial-write evidence. It is '
              'not a valid checkpoint and neither latest/best pointer references it. Resume used committed step 240.','',
              '## Small-sample learning and weakest image','',
              f"Final weakest image: `{weak['image']}`. GT foreground fraction "
              f"**{weak['eval_running_statistics']['true_foreground_fraction']:.4f}**; predicted fraction "
              f"**{weak['eval_running_statistics']['predicted_foreground_fraction']:.4f}**. Evaluation-mode Dice "
              f"**{weak['eval_running_statistics']['dice']:.6f}**, precision **{weak['eval_running_statistics']['precision']:.6f}**, "
              f"recall **{weak['eval_running_statistics']['recall']:.6f}**, FP **{weak['eval_running_statistics']['fp']}**, "
              f"FN **{weak['eval_running_statistics']['fn']}**.",'',
              f"With exactly the same weights but per-image BatchNorm statistics, Dice is "
              f"**{weak['single_image_batch_statistics']['dice']:.6f}** (difference "
              f"**{weak['batch_statistics_dice_difference']:+.6f}**). BatchNorm buffers were restored immediately; "
              'no weights or thresholds were changed. This diagnostic score is not used in the headline results.','']
    lines += ['Independent CPU verification confirmed that every exported ground-truth mask exactly equals '
              'the corresponding original 224×224 binary mask in its display orientation. Thus a transpose mismatch '
              'does not explain these eight-image scores. The initial `covid_279` overlay shows two small bilateral '
              'lower-lung annotations and scattered baseline false positives; its pairing agrees with the stored prompt. '
              'This visual check does not establish clinical correctness of the annotation.','']
    if abs(weak['batch_statistics_dice_difference'])>.15:
        lines += ['The large difference confirms that BatchNorm statistics materially affect this image. '
                  '**Hypothesis:** microbatch-one running-statistic mismatch contributes to poor evaluation on the outlier. '
                  'A later controlled normalization/statistics experiment should test this before assuming the image is unlearnable.','']
    else:
        lines += ['This comparison does not establish a large BatchNorm explanation for the weakest image. '
                  'Small lesion area, loss balance and optimization remain hypotheses; inspect the saved X-ray/mask/overlay '
                  'and per-image error counts before changing the model.','']
    lines += ['| Image | Final Dice | Final IoU | Precision | Recall |','|---|---:|---:|---:|---:|']
    for r in final_table:
        lines.append(f"| {r['image']} | {float(r['dice']):.6f} | {float(r['iou']):.6f} | {float(r['precision']):.6f} | {float(r['recall']):.6f} |")
    lines += ['', f"Finite-gradient checks covered **{verification['verified_training_updates']} updates**; pre-clip L2 norms "
              f"ranged from **{verification['gradient_norm_range'][0]:.5f} to {verification['gradient_norm_range'][1]:.5f}**. "
              f"The watched output parameter changed on **{verification['watched_parameter_updates_nonzero']} updates**. "
              'Losses and updated parameters were checked for finite values. See `training_microsteps.csv`.','',
              '## Readiness and next configuration','']
    reliable=summary['success_criterion_met'] and persistence['verified']
    if reliable:
        readiness='The tested implementation is conditionally ready for a separately approved 2-epoch pilot, subject to GPU/host memory headroom. No pilot has been started.'
    else:
        readiness='Checkpointing and FP32 memory stability are ready, but reliable fitting across all eight images has not met the predeclared criterion (mean >=0.90 and minimum >=0.75 twice). Do not start a full-dataset pilot until the weakest-image issue is resolved or an explicit scientific rationale is approved.'
    lines += [readiness,'',
              'For the next authorized diagnostic/pilot: retain microbatch 1, FP32 logits BCE + FP32 soft Dice, AMP disabled, '
              'accumulation 8, AdamW `foreach=False`, encoder LR 3e-5 / decoder LR 3e-4, weight decay 0.01, clip norm 1, '
              'workers 0, frozen text backbone in eval mode, tested no-attention-weight path, and atomic periodic model+optimizer '
              'checkpoints. Preserve threshold 0.5 for comparisons; any later threshold calibration must use validation only. '
              'An eight-step accumulated update does not make BatchNorm see eight images.','',
              'For a later pilot, use streaming validation metrics/histograms instead of retaining every probability map in RAM, '
              'keep pin-memory disabled initially, and choose an explicit checkpoint interval such as 100 optimizer updates '
              'plus each epoch end. Here each resumable checkpoint is about 979 MB and writes/hashing take roughly 55 seconds, '
              'so copying full checkpoints at every small metric improvement is expensive. Keep incremental histories and '
              'an emergency last-valid pointer. These are recommended changes, not an executed pilot.','',
              'After resolving the weakest-image limitation, begin with two epochs (not five immediately), monitor held-out '
              'validation per-image/global Dice, IoU, precision, recall and foreground fraction, and extend only if convergence '
              'and resource logs justify it. Any memory or normalization changes require matched correctness checks. '
              'No full training, new test scoring, or PowerPoint generation was performed.','']
    if weak['batch_statistics_dice_difference']>.15:
        lines += ['Before a full-data pilot, the next proposed experiment is another bounded eight-image branch with decoder '
                  'BatchNorm running statistics frozen during optimization (affine weights can still learn). This would make '
                  'the optimization forward match inference normalization. **Expected impact is a hypothesis**, not an '
                  'executed improvement: it may reduce the weak-image train/eval gap without changing the original architecture. '
                  'If that fails, compare a separately labeled GroupNorm/InstanceNorm variant or train-only BN recalibration. '
                  'Do not replace the reported evaluation scores with training-mode predictions.','']
    lines += ['Use the original epoch-18 checkpoint as the parent of a future generalization pilot, with a deliberately '
              'documented optimizer initialization. The Phase 1B weights were fit on eight images and are diagnostic artifacts; '
              'do not present their scores as held-out performance or silently use them as the baseline of a reproduction run.','']
    write('recommended_next_configuration.json',{'ready_for_pilot':reliable,'readiness':readiness,
          'configuration':read('config.json'),'pilot_parent_checkpoint':'outputs/run/checkpoints/best_dice.ckpt (epoch 18)',
          'subset_checkpoints_purpose':'Eight-image diagnostics only',
          'proposed_normalization_diagnostic':'Freeze decoder BatchNorm running statistics; hypothesis, not executed',
          'full_training_started':False,'next_action_requires_user_approval':True})
else:
    lines += ['The diagnostic did not complete. No final improvement is claimed.', '',
              f"Failure stage: {failure['stage'] if failure else 'unknown'}. "
              f"Error: {failure['message'] if failure else 'unknown'}. See `failure.json` and `checkpoints/latest.json`. ", '']

lines += ['## Preservation and evidence','',
          f"SHA-256 verification checked **{preservation['checked_files'] if preservation else 'unknown'} protected files**. "
          f"All checked originals and previous Phase 1 files unchanged: **{preservation['all_protected_files_unchanged'] if preservation else 'not verified'}**. "
          'The protected inventory includes the six notebooks, original source, original checkpoint/history/evaluation outputs, '
          'all prior Phase 1 experiment files, and the 16 selected raw image/mask files. The runner has no dataset writes.','',
          '- `run_phase1b.py`, `resume_phase1b.py`, `config.json`, `resume_config.json`, `sample_manifest.json`, `environment.json`: reproducible protocol and source.',
          '- `summary.json`, `initial_per_image.csv`, `best_per_image.csv`, `final_per_image.csv`: measured metrics.',
          '- `combined_training_microsteps.csv`, `combined_evaluation_progression.csv`, `training_progression.png`, `loss_and_gpu_memory.png`: committed progression; attempt-specific histories remain preserved.',
          '- `gpu_monitor.csv`, `gpu_monitor_summary.json`, `startup_resources.json`, `startup_nvidia_smi.txt`: measured resources and other processes.',
          '- `checkpoints/latest.json`, `checkpoints/best.json`, `checkpoint_log.csv`, `checkpoint_persistence_verification.json`: saved states and reload test.',
          '- `predictions/initial`, `predictions/final`, `predictions/step_*`: every sample has X-ray, GT, prediction, overlay and probability array.',
          '- `weakest_image_analysis.json`, `predictions/weakest_batch_statistics`: fixed-weight normalization diagnostic.',
          '- `independent_verification.json`: CPU recomputation from saved PNGs/probabilities and confusion counts; no new model inference.',
          '- `preservation_before.json`, `preservation_verification.json`: hashes and preservation evidence.','',
          'Phase 1B is stopped. Await approval before any further training.']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'summary':summary,'resource':resource,'independent_verification':verification},indent=2))
