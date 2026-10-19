"""Prepared, approval-gated two-epoch Frozen-BN fine-tuning. No test data."""
import argparse
import csv
import json
import math
from pathlib import Path
import time
import traceback
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--resume',action='store_true')
parser.add_argument('--execute', action='store_true', help='Run only after the user approves Phase 3')
args=parser.parse_args()
if not args.execute:
    print('Phase 3 is prepared only. No ML imports, model loading, or training performed.')
    print((HERE/'config.json').read_text(encoding='utf-8'))
    raise SystemExit(0)
if (HERE/'summary.json').exists(): raise SystemExit('Completed pilot preserved; refusing rerun')
if (HERE/'training_microsteps.csv').exists() and not args.resume: raise SystemExit('Use --resume for existing history')
if args.resume and not (HERE/'checkpoints/latest.json').exists(): raise SystemExit('No committed checkpoint available')
assert (HERE/'protected_sources.json').exists(), 'Run prepare.py before approved execution'
import phase3_utils as h
torch,np,nn=h.torch,h.np,h.nn
from src.data.dataset import group_key

CONFIG=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
assert CONFIG['phase']=='3' and CONFIG['max_epochs']==2
assert CONFIG['microbatch']==1 and CONFIG['accumulation']==8 and not CONFIG['AMP']
assert CONFIG['precision']=='FP32' and CONFIG['threshold']==.5 and CONFIG['comparison_threshold']==.525
h.CONFIG=CONFIG
SESSION=uuid.uuid4().hex[:8]
low_ram_checks=0
def ensure_json(name, value):
    if (HERE/name).exists():
        assert read(name) == value, 'Immutable preparation artifact changed: ' + name
    else:
        h.atomic_json(name, value)


def verify_protected_sources():
    inventory = read('protected_sources.json')
    changed = []
    for name, expected in inventory['files'].items():
        path = ROOT/name
        if not path.is_file():
            changed.append(name)
            continue
        stat = path.stat()
        if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
            changed.append(name)
        elif expected.get('sha256') and h.sha(path) != expected['sha256']:
            changed.append(name)
    if changed:
        raise RuntimeError('Protected source files changed: ' + str(changed))
    return {'checked_files': len(inventory['files']), 'unchanged': True,
            'coverage': inventory['coverage']}


def read(name): return json.loads((HERE/name).read_text(encoding='utf-8'))
def csv_rows(name):
    p=HERE/name
    if not p.exists(): return []
    with p.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def plain(item):
    (x,t),y=item
    return (x.as_tensor() if hasattr(x,'as_tensor') else x,t),y.as_tensor() if hasattr(y,'as_tensor') else y
def training_mode(model):
    model.train();model.text_encoder.model.eval()
    for name,module in model.named_modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):
            assert name.startswith(('decoder16.','decoder8.','decoder4.'))
            module.eval()
def bn_buffers(model):
    return {n:v.detach().cpu().clone() for n,v in model.named_buffers()
            if n.endswith(('running_mean','running_var','num_batches_tracked'))}
def construct_on_gpu(vision,text_source):
    from transformers import AutoConfig,AutoModel
    from unittest.mock import patch
    def configured(source,**kwargs):
        config=AutoConfig.from_pretrained(source,trust_remote_code=True,local_files_only=True)
        if 'output_hidden_states' in kwargs: config.output_hidden_states=kwargs['output_hidden_states']
        return AutoModel.from_config(config,trust_remote_code=True)
    # The original checkpoint contains all backbone parameters. Avoid loading
    # separate pretrained weight files or constructing a full CPU model copy.
    with patch.object(AutoModel,'from_pretrained',new=staticmethod(configured)):
        with torch.device('cuda'): model=h.create_model(vision,text_source).float()
    model=model.cuda()  # Also move any explicitly CPU-created nonpersistent buffers.
    assert all(p.device.type=='cuda' for p in model.parameters())
    return model
def parameter_sha(model):
    digest=h.hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(name.encode());digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
def verify_bn(model,initial):
    current=bn_buffers(model)
    assert len(current)==18 and all(torch.equal(initial[n],v) for n,v in current.items()),'Frozen BN drift'
def resource_guard():
    global low_ram_checks
    if h.RESOURCE_FAILURE: raise RuntimeError(h.RESOURCE_FAILURE)
    r={**h.host_memory(),**h.smi()}
    assert 'device_free_mib' in r,'GPU monitor unavailable'
    if r['device_free_mib']<256 or r['available_commit_mib']<512:
        raise RuntimeError('Resource headroom guard: '+str(r))
    low_ram_checks=low_ram_checks+1 if r['available_ram_mib']<32 else 0
    if low_ram_checks>=3: raise RuntimeError('Persistent available RAM below 32 MiB')
    if h.memory()['peak_allocated_mib']>2200: raise RuntimeError('Unexpected tensor allocation >2200 MiB')
    return r
def serial_records(dataset):
    return [{'index':i,'name':r['name'],'prompt':r['prompt'],'image':str(r['image'].relative_to(ROOT)),
             'mask':str(r['mask'].relative_to(ROOT)),'group':group_key(r['name'])} for i,r in enumerate(dataset.records)]
def metric_summary(rows):
    result={f'mean_{k}':float(np.mean([float(r[k]) for r in rows])) for k in ['dice','iou','precision','recall','loss']}
    result.update(images=len(rows),minimum_dice=min(float(r['dice']) for r in rows),
        weakest_image=min(rows,key=lambda r:float(r['dice']))['image'],
        predicted_foreground_percent=100*float(np.mean([float(r['predicted_foreground_fraction']) for r in rows])),
        GT_foreground_percent=100*float(np.mean([float(r['true_foreground_fraction']) for r in rows])))
    tp,fp,fn,tn=[sum(int(r[k]) for r in rows) for k in ['tp','fp','fn','tn']]
    result.update(tp=tp,fp=fp,fn=fn,tn=tn,global_dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
        global_iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,global_precision=tp/(tp+fp) if tp+fp else 0.,
        global_recall=tp/(tp+fn) if tp+fn else 0.,aggregation='Primary means: per-image macro; global: pooled pixel counts')
    dice_values=np.array([float(r['dice']) for r in rows])
    small=[r for r in rows if float(r['true_foreground_fraction'])<.01]
    result.update(dice_percentiles={str(p):float(np.percentile(dice_values,p)) for p in [5,10,25,50,75,90]},
        zero_dice_images=int(np.sum(dice_values==0)),below_05_images=int(np.sum(dice_values<.5)),
        below_07_images=int(np.sum(dice_values<.7)),
        small_GT_under_1pct={'images':len(small),'mean_dice':float(np.mean([float(r['dice']) for r in small])) if small else None})
    return result

def validate(model,capture,dataset,label,checkpoint_sha,representatives):
    existing=HERE/f'validation/{label}/summary.json'
    if existing.exists():
        saved=read(f'validation/{label}/summary.json')
        assert saved['checkpoint_sha256']==checkpoint_sha and saved['threshold']==.5 and saved['images']==len(dataset)
        assert saved['comparison']['threshold']==CONFIG['comparison_threshold']
        for key, hash_key in [('per_image_csv','per_image_csv_sha256'),
                              ('comparison_per_image_csv','comparison_per_image_csv_sha256')]:
            assert h.sha(HERE/saved[key])==saved[hash_key]
            assert [r['image'] for r in csv_rows(saved[key])]==[r['name'] for r in dataset.records]
        print('REUSE VERIFIED FULL VALIDATION',label,flush=True)
        return saved
    model.eval();capture.clear();h.release()
    started=time.perf_counter();table=[];comparison=[];attempt=f'{label}_{SESSION}';weak=None
    for i in range(len(dataset)):
        h.STAGE=f'{label} validation {i+1}/{len(dataset)}'
        if h.RESOURCE_FAILURE: raise RuntimeError(h.RESOURCE_FAILURE)
        item=plain(dataset[i])
        with torch.inference_mode():
            image,text,target=h.gpu_item(item)
            probability=model([image,text]);loss,dl,bl=h.loss_fn(capture.logits,target)
            assert torch.isfinite(probability).all() and torch.isfinite(loss),'Nonfinite validation forward'
            assert probability.shape==target.shape
            a,y=probability[0,0].cpu().numpy().copy(),target[0,0].cpu().numpy().copy()
            assert 0 <= a.min() <= a.max() <= 1
            shared={'validation_index':i,'image':dataset.records[i]['name'],
                    'loss':float(loss),'dice_loss':float(dl),'bce_loss':float(bl)}
            row={**shared,**h.scores(a,y,.5)}
            alternate={**shared,**h.scores(a,y,CONFIG['comparison_threshold'])}
            capture.clear();del image,text,target,probability,loss,dl,bl,item
        table.append(row);comparison.append(alternate)
        if i in representatives: h.save_images(f'{attempt}/sample_{i:04d}',[dataset.records[i]],[(a,y)])
        if weak is None or row['dice']<weak[0]: weak=(row['dice'],i,a.copy(),y.copy())
        del a,y
        if (i+1)%200==0:
            print('VALIDATION',label,i+1,'/',len(dataset),flush=True);resource_guard()
        if (HERE/'STOP_REQUESTED').exists(): raise InterruptedError('Stop file requested interruption during validation')
    result=metric_summary(table)
    assert result['images']==1423
    filename=f'validation/{label}/per_image.csv'
    alternate_filename=f'validation/{label}/per_image_threshold_0525.csv'
    h.csv_write(filename,table);h.csv_write(alternate_filename,comparison)
    h.save_images(f'{attempt}/weakest',[dataset.records[weak[1]]],[(weak[2],weak[3])])
    result.update(label=label,threshold=.5,checkpoint_sha256=checkpoint_sha,
        per_image_csv=filename,per_image_csv_sha256=h.sha(HERE/filename),
        comparison={**metric_summary(comparison),'threshold':CONFIG['comparison_threshold'],
                    'used_for_checkpoint_selection':False},
        comparison_per_image_csv=alternate_filename,comparison_per_image_csv_sha256=h.sha(HERE/alternate_filename),
        representative_indices=representatives,prediction_directory=f'predictions/{attempt}',weakest_index=weak[1],
        validation_seconds=time.perf_counter()-started,**h.memory())
    h.atomic_json(f'validation/{label}/summary.json',result)
    print('FULL VALIDATION COMPLETE',label,result,flush=True)
    del table,comparison,weak;h.release()
    return result

def restore_optimizer(optimizer,state):
    optimizer.state.clear();h.release();optimizer.load_state_dict(state)
    clones=0
    for values in optimizer.state.values():
        for k,v in list(values.items()):
            if isinstance(v,torch.Tensor) and v.device.type=='cpu': values[k]=v.clone();clones+=1
    assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor))
    return clones
def restore_rng(state):
    h.random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].clone());torch.cuda.set_rng_state_all([v.clone() for v in state['cuda_rng']])
def save_checkpoint(model,optimizer,scheduler,capture,probe_items,probe_records,cursor,parent_sha,manifest_sha,orders):
    resource_guard();optimizer.zero_grad(set_to_none=True);capture.clear();h.release()
    pr,pa=h.evaluate(model,capture,probe_items,probe_records)
    probe=[a[0].copy() for a in pa];del pr,pa
    path=HERE/'checkpoints'/f'e{cursor["epoch"]:02d}_p{cursor["position"]:05d}_u{cursor["optimizer_steps"]:05d}_{uuid.uuid4().hex[:8]}.ckpt'
    assert not path.exists()
    payload={'kind':'PHASE3_FROZEN_BN_FINETUNING_V1','model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'cursor':cursor.copy(),'config':CONFIG,'parent_checkpoint_sha256':parent_sha,
        'manifest_sha256':manifest_sha,'epoch_orders':orders,'probe_probabilities':probe,
        'python_rng':h.random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all()}
    started=time.perf_counter();h.atomic_torch(path,payload);del payload
    pointer={'path':str(path.relative_to(HERE)),'sha256':h.sha(path),'bytes':path.stat().st_size,
        **cursor,'write_seconds':time.perf_counter()-started}
    h.atomic_json('checkpoints/latest.json',pointer);h.append_csv('checkpoint_log.csv',pointer)
    print('COMMITTED CHECKPOINT',pointer,flush=True)
    return pointer

def verify_reload(pointer,model,optimizer,scheduler,capture,probe_items,probe_records):
    path=HERE/pointer['path'];assert h.sha(path)==pointer['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    loaded=model.load_state_dict(state['model'],strict=True);clones=restore_optimizer(optimizer,state['optimizer'])
    scheduler.load_state_dict(state['scheduler']);restore_rng(state)
    expected=[v.copy() for v in state['probe_probabilities']]
    del state;h.release()
    rr,aa=h.evaluate(model,capture,probe_items,probe_records)
    difference=max(float(np.max(np.abs(v[0]-old))) for v,old in zip(aa,expected))
    assert difference==0 and not loaded.missing_keys and not loaded.unexpected_keys
    assert scheduler.last_epoch==pointer['optimizer_steps']
    counters=sorted({float(s['step']) for s in optimizer.state.values()})
    assert counters==[float(pointer['optimizer_steps'])]
    temporary=path.with_name(path.name+'.interruption_test.tmp')
    with temporary.open('wb') as f: f.write(b'SIMULATED PARTIAL WRITE');f.flush();h.os.fsync(f.fileno())
    unchanged=h.sha(path)==pointer['sha256'];temporary.unlink();assert unchanged
    h.atomic_json(f'checkpoint_verification_u{pointer["optimizer_steps"]:05d}_{SESSION}.json',
        {'verified':True,'checkpoint':pointer,'maximum_prediction_difference':difference,
         'strict_missing_keys':loaded.missing_keys,'strict_unexpected_keys':loaded.unexpected_keys,
         'finite_optimizer_state':True,'step_counters':counters,'scheduler_steps':scheduler.last_epoch,
         'cloned_CPU_scalar_tensors':clones,'interrupted_write_preserved_checkpoint':unchanged,
         'process_restart_exercised':False})
    del rr,aa,expected;h.release()
    print('CHECKPOINT RELOAD VERIFIED',pointer['optimizer_steps'],flush=True)

def rollback_history(global_step):
    rows=csv_rows('training_microsteps.csv')
    if not rows: return 0
    keep=[r for r in rows if int(r['global_microstep'])<=global_step]
    discard=len(rows)-len(keep)
    if discard:
        archive=HERE/f'training_microsteps_before_resume_{SESSION}.csv'
        assert not archive.exists();archive.write_bytes((HERE/'training_microsteps.csv').read_bytes())
        if keep: h.csv_write('training_microsteps.csv',keep)
        else:
            with (HERE/'training_microsteps.csv').open('w',newline='',encoding='utf-8') as f:
                csv.DictWriter(f,fieldnames=list(rows[0])).writeheader();f.flush();h.os.fsync(f.fileno())
    assert len(keep)==global_step
    return discard

def epoch_training_summary(epoch,N,order):
    rows=[r for r in csv_rows('training_microsteps.csv') if int(r['epoch'])==epoch]
    assert len(rows)==N and [int(r['train_index']) for r in rows]==order
    updates=[r for r in rows if int(r['update_committed'])]
    result={f'mean_{k}':float(np.mean([float(r[k]) for r in rows])) for k in ['loss','dice','iou']}
    norms=[float(r['gradient_norm_before_clip']) for r in updates]
    result.update(epoch=epoch,training_images=N,optimizer_steps=len(updates),skipped_updates=0,
        finite_gradient_updates=len(updates),gradient_norm_min=min(norms),gradient_norm_max=max(norms),
        clipped_updates=sum(v>1 for v in norms),training_compute_seconds=sum(float(r['compute_seconds']) for r in rows),
        mean_compute_seconds_per_image=float(np.mean([float(r['compute_seconds']) for r in rows])),
        interpretation='Online pre-update macro training metrics, not fixed-weight reevaluation')
    h.atomic_json(f'training_epoch_{epoch}.json',result);return result

def main():
    model=optimizer=scheduler=capture=None;started=time.perf_counter();cursor=None
    try:
        preservation_start=verify_protected_sources()
        import shutil
        assert shutil.disk_usage(HERE).free >= CONFIG['minimum_disk_free_gib']*2**30, 'Need at least 20 GiB free disk'
        assert torch.cuda.is_available();assert h.smi()['device_free_mib']>=1800
        resource_wait=time.perf_counter()
        while h.host_memory()['available_commit_mib']<2500:
            h.STAGE='startup waiting for >=2500 MiB available Windows commit'
            if time.perf_counter()-resource_wait>300:
                raise RuntimeError('Startup headroom remained below 2500 MiB for five minutes: '+str(h.host_memory()))
            time.sleep(5)
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True);h.seed()
        # Functional test: accumulating a two-image tail must equal its batch-mean gradient.
        z=torch.tensor([[[[-20.,0.],[20.,2.]]],[[[2.,-2.],[0.,20.]]]],requires_grad=True)
        y=torch.tensor([[[[0.,1.],[1.,1.]]],[[[1.,0.],[0.,1.]]]])
        batch_loss=h.loss_fn(z,y)[0];batch_grad=torch.autograd.grad(batch_loss,z)[0]
        tail=z.detach().clone().requires_grad_(True)
        for i in range(2): (h.loss_fn(tail[i:i+1],y[i:i+1])[0]/2).backward()
        difference=float((tail.grad-batch_grad).abs().max())
        assert difference<1e-7 and torch.isfinite(tail.grad).all()
        h.atomic_json(f'loss_and_tail_sanity_{SESSION}.json',{'finite':True,'tail_vs_batch_gradient_max_difference':difference,
            'tested_window_size':2,'stable_logits_range':[-20.,20.]})
        del z,y,tail,batch_loss,batch_grad
        cfg=h.Config(data_root=ROOT/'_qata_compatible');vision,text_source=h._model_sources(cfg)
        tokenizer=h.AutoTokenizer.from_pretrained(text_source,trust_remote_code=True,local_files_only=True)
        train=h.QaTaDataset(cfg.train_csv,cfg.data_root,'train',tokenizer,augment=False,seed=2026)
        val=h.QaTaDataset(cfg.train_csv,cfg.data_root,'validation',tokenizer,augment=False,seed=2026)
        assert len(train)==5722 and len(val)==1423
        original=[]
        with (ROOT/'outputs/run/prepared_annotations/split_manifest.csv').open(encoding='utf-8-sig',newline='') as f: original=list(csv.DictReader(f))
        for dataset,split in [(train,'train'),(val,'validation')]:
            expected={r['Image']:r['Description'].strip() for r in original if r['Split']==split}
            assert {r['name']:r['prompt'] for r in dataset.records}==expected,'Existing split/prompt membership changed'
        assert not {group_key(r['name']) for r in train.records}&{group_key(r['name']) for r in val.records}
        manifests={'train':serial_records(train),'validation':serial_records(val)}
        phase2=ROOT/'experiments/phase2_pilot'
        assert h.sha(phase2/'data_manifest.json')==CONFIG['source_validation_manifest_sha256']
        assert json.loads((phase2/'data_manifest.json').read_text())==manifests
        ensure_json('data_manifest.json',manifests);manifest_sha=h.sha(HERE/'data_manifest.json')
        orders={}
        for epoch in [1,2]:
            order=list(range(len(train)));h.random.Random(CONFIG['order_seed_base']+epoch).shuffle(order);orders[str(epoch)]=order
        ensure_json('epoch_orders.json',orders)
        representatives=sorted(set(read_source_representatives()+[954,1383]))
        ensure_json('representative_manifest.json',{'selection':'Eight original Phase 2 representatives plus its two zero-Dice validation cases',
            'selected_before_phase3_scoring':True,'validation_indices':representatives,
            'records':[manifests['validation'][i] for i in representatives]})
        CONFIG.update(train_images=len(train),validation_images=len(val),updates_per_epoch=math.ceil(len(train)/8),
            planned_optimizer_updates=2*math.ceil(len(train)/8),train_csv_sha256=h.sha(cfg.train_csv),
            original_split_manifest_sha256=h.sha(ROOT/'outputs/run/prepared_annotations/split_manifest.csv'))
        if (HERE/'config.json').exists(): assert read('config.json')==CONFIG
        else: h.atomic_json('config.json',CONFIG)
        (HERE/f'executed_run_finetuning_{SESSION}.py').write_bytes(Path(__file__).read_bytes())
        (HERE/f'executed_phase3_utils_{SESSION}.py').write_bytes((HERE/'phase3_utils.py').read_bytes())
        h.atomic_json(f'environment_{SESSION}.json',{'python':h.sys.version,'torch':torch.__version__,
            'GPU':torch.cuda.get_device_name(),'deterministic_warn_only':True,'attention_backward_may_be_nondeterministic':True})
        h.STAGE='strict best Phase 2 Epoch 2 checkpoint loading'
        model=construct_on_gpu(vision,text_source);parent=ROOT/CONFIG['parent_checkpoint']
        assert parent.resolve().is_relative_to(ROOT/'experiments/phase2_pilot/checkpoints')
        parent_sha=h.sha(parent)
        assert parent_sha==CONFIG['parent_checkpoint_sha256']
        assert h.sha(phase2/'checkpoints/best.json')==CONFIG['source_phase2_best_metadata_sha256']
        assert h.sha(ROOT/'experiments/phase2_threshold_calibration/calibration_summary.json')==CONFIG['source_calibration_summary_sha256']
        state=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
        assert state['kind']=='PHASE2_FROZEN_BN_PILOT_V1' and state['cursor']['epoch']==2
        assert state['cursor']['position']==5722 and state['cursor']['optimizer_steps']==1432
        loaded=model.load_state_dict(state['model'],strict=True)
        del state;h.release()
        initial_bn=bn_buffers(model);assert len(initial_bn)==18
        for module in model.modules():
            if isinstance(module,nn.MultiheadAttention): module.forward=h.functools.partial(module.forward,need_weights=False)
        capture=h.Capture(model)
        probe_indices=[36,686];probe_items=[plain(val[i]) for i in probe_indices]
        probe_records=[{'name':val.records[i]['name']} for i in probe_indices]
        rr,aa=h.evaluate(model,capture,probe_items,probe_records)
        reference_summary=json.loads((phase2/'validation/epoch_2/summary.json').read_text())
        reference=phase2/reference_summary['prediction_directory']
        difference=max(float(np.max(np.abs(a[0].T-np.load(reference/f'sample_{index:04d}/00_probability.npy'))))
            for a,index in zip(aa,probe_indices))
        assert difference<=2e-5, 'Phase 2 parent prediction probes differ'
        del rr,aa;h.release();resource_guard()
        ensure_json('parent_verification.json',{'sha256':parent_sha,
            'strict_missing_keys':loaded.missing_keys,'strict_unexpected_keys':loaded.unexpected_keys,
            'BN_buffer_tensors':len(initial_bn),'architecture_changed':False,
            'probe_validation_indices':probe_indices,'baseline_probability_max_difference_vs_phase2':difference,
            'direct_GPU_construction_verified':True})
        baseline=validate(model,capture,val,'baseline',parent_sha,representatives)
        for metric, key in [('mean_dice','expected_parent_dice_05'),('mean_iou','expected_parent_iou_05')]:
            assert abs(baseline[metric]-CONFIG[key])<=CONFIG['baseline_tolerance'], 'Primary parent baseline mismatch'
        for metric, key in [('mean_dice','expected_parent_dice_0525'),('mean_iou','expected_parent_iou_0525')]:
            assert abs(baseline['comparison'][metric]-CONFIG[key])<=CONFIG['baseline_tolerance'], 'Comparison parent baseline mismatch'
        h.atomic_json(f'baseline_verification_{SESSION}.json',{'passed':True,'tolerance':CONFIG['baseline_tolerance'],
            'threshold_05_dice':baseline['mean_dice'],'threshold_05_iou':baseline['mean_iou'],
            'threshold_0525_dice':baseline['comparison']['mean_dice'],'threshold_0525_iou':baseline['comparison']['mean_iou']})
        verify_protected_sources()
        optimizer=torch.optim.AdamW([
            {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.model.')],'lr':CONFIG['vision_lr']},
            {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.model.')],'lr':CONFIG['decoder_lr']}],weight_decay=.01,foreach=False)
        total_updates=CONFIG['planned_optimizer_updates']
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1-.5*min(step,total_updates)/total_updates)
        cursor={'epoch':1,'position':0,'global_microstep':0,'optimizer_steps':0,'skipped_updates':0}
        if args.resume:
            pointer=read('checkpoints/latest.json');path=HERE/pointer['path']
            assert path.resolve().is_relative_to(HERE) and h.sha(path)==pointer['sha256']
            state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert state['kind']=='PHASE3_FROZEN_BN_FINETUNING_V1'
            assert state['config']==CONFIG and state['manifest_sha256']==manifest_sha and state['epoch_orders']==orders
            assert state['parent_checkpoint_sha256']==parent_sha
            model.load_state_dict(state['model'],strict=True);clones=restore_optimizer(optimizer,state['optimizer'])
            scheduler.load_state_dict(state['scheduler']);restore_rng(state)
            cursor=state['cursor'].copy();expected=[a.copy() for a in state['probe_probabilities']]
            assert all(cursor[k]==pointer[k] for k in cursor)
            assert cursor['position']%8==0 or cursor['position']==5722
            assert scheduler.last_epoch==cursor['optimizer_steps']
            counters={float(s['step']) for s in optimizer.state.values()}
            assert counters==({float(cursor['optimizer_steps'])} if cursor['optimizer_steps'] else set())
            del state;h.release()
            rr,aa=h.evaluate(model,capture,probe_items,probe_records)
            diff=max(float(np.max(np.abs(a[0]-b))) for a,b in zip(aa,expected));assert diff==0
            discarded=rollback_history(cursor['global_microstep'])
            h.atomic_json(f'resume_{SESSION}.json',{'checkpoint':pointer,'verified':True,'probe_max_difference':diff,
                'discarded_uncommitted_microsteps':discarded,'cloned_CPU_scalar_tensors':clones})
            del rr,aa,expected;verify_bn(model,initial_bn)
        else:
            h.seed()
            h.STAGE='initial committed model and fresh optimizer checkpoint'
            save_checkpoint(model,optimizer,scheduler,capture,probe_items,probe_records,cursor,parent_sha,manifest_sha,orders)
        torch.cuda.reset_peak_memory_stats();results={'baseline':baseline}
        training=[]
        first_reload_already=any(HERE.glob('checkpoint_verification_u00100_*.json'))
        for epoch in range(cursor['epoch'],3):
            cursor['epoch']=epoch
            order=orders[str(epoch)];N=len(order);epoch_start=time.perf_counter()
            for position in range(cursor['position'],N):
                if position%8==0: optimizer.zero_grad(set_to_none=True)
                window_size=min(8,N-(position//8)*8)
                if h.RESOURCE_FAILURE: raise RuntimeError(h.RESOURCE_FAILURE)
                if h.memory()['peak_allocated_mib']>CONFIG['max_tensor_allocation_mib']: raise RuntimeError('Tensor memory guard exceeded')
                training_mode(model);index=order[position];tick=time.perf_counter()
                h.STAGE=f'epoch {epoch} forward {position+1}/{N}'
                item=plain(train[index]);image,text,target=h.gpu_item(item)
                probability=model([image,text]);loss,dl,bl=h.loss_fn(capture.logits,target)
                if not torch.isfinite(loss) or not torch.isfinite(probability).all(): raise FloatingPointError('Nonfinite training forward/loss')
                a,y=probability[0,0].detach().cpu().numpy().copy(),target[0,0].cpu().numpy().copy()
                metric=h.scores(a,y);del a,y
                row={'global_microstep':cursor['global_microstep']+1,'epoch':epoch,'position':position+1,'train_index':index,
                    'image':train.records[index]['name'],'optimizer_steps':cursor['optimizer_steps'],'skipped_updates':0,
                    'window_size':window_size,'loss':float(loss.detach()),'dice_loss':float(dl.detach()),'bce_loss':float(bl.detach()),
                    **metric,'lr_vision':optimizer.param_groups[0]['lr'],'lr_decoder':optimizer.param_groups[1]['lr'],
                    'update_committed':0,'gradient_tensors':0,'nonfinite_gradient_tensors':0,
                    'gradient_norm_before_clip':0.,'watched_parameter_max_update':0.}
                h.STAGE=f'epoch {epoch} backward {position+1}/{N}'
                (loss/window_size).backward();capture.clear();del item,image,text,target,probability,loss,dl,bl
                boundary=(position+1)%8==0 or position+1==N
                if boundary:
                    count,bad,norm=h.gradients(model)
                    if bad: raise FloatingPointError('Nonfinite gradients: '+str(bad))
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
                    watched=next(p for n,p in model.named_parameters() if n.startswith('out.') and p.requires_grad)
                    before=watched.detach().clone();h.STAGE=f'epoch {epoch} optimizer {position+1}/{N}'
                    optimizer.step();torch.cuda.synchronize()
                    delta=float((watched.detach()-before).abs().max());del watched,before
                    if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad): raise FloatingPointError('Nonfinite updated parameters')
                    cursor['optimizer_steps']+=1;scheduler.step();optimizer.zero_grad(set_to_none=True)
                    row.update(optimizer_steps=cursor['optimizer_steps'],update_committed=1,gradient_tensors=count,
                        gradient_norm_before_clip=norm,watched_parameter_max_update=delta)
                row.update(compute_seconds=time.perf_counter()-tick,**h.memory())
                cursor.update(position=position+1,global_microstep=cursor['global_microstep']+1)
                h.append_csv('training_microsteps.csv',row)
                if boundary and (position+1)%64==0: resource_guard()
                if boundary and ((position+1)%800==0 or position+1==N):
                    verify_bn(model,initial_bn);h.STAGE=f'epoch {epoch} checkpoint {position+1}'
                    pointer=save_checkpoint(model,optimizer,scheduler,capture,probe_items,probe_records,cursor,parent_sha,manifest_sha,orders)
                    if cursor['optimizer_steps']==100 and not first_reload_already:
                        verify_reload(pointer,model,optimizer,scheduler,capture,probe_items,probe_records);first_reload_already=True
                if (position+1)%200==0:
                    print('TRAIN',epoch,position+1,'/',N,'updates',cursor['optimizer_steps'],'loss',row['loss'],
                        'Dice',row['dice'],'LR',optimizer.param_groups[1]['lr'],flush=True)
                    h.atomic_json('progress.json',{'status':'training','cursor':cursor,'session':SESSION,'last_loss':row['loss'],**h.memory()})
                if boundary and (HERE/'STOP_REQUESTED').exists():
                    save_checkpoint(model,optimizer,scheduler,capture,probe_items,probe_records,cursor,parent_sha,manifest_sha,orders)
                    raise InterruptedError('Stop file requested interruption at committed update')
            verify_bn(model,initial_bn)
            pointer=read('checkpoints/latest.json')
            assert pointer['epoch']==epoch and pointer['position']==N
            tr=epoch_training_summary(epoch,N,order);training.append(tr)
            result=validate(model,capture,val,f'epoch_{epoch}',pointer['sha256'],representatives)
            results[f'epoch_{epoch}']=result
            best=read('checkpoints/best.json') if (HERE/'checkpoints/best.json').exists() else None
            if best is None or result['mean_dice']>best['validation_mean_dice']:
                h.atomic_json('checkpoints/best.json',{**pointer,'validation_mean_dice':result['mean_dice'],
                    'validation_mean_iou':result['mean_iou'],'validation_summary':f'validation/epoch_{epoch}/summary.json'})
            h.atomic_json(f'epoch_{epoch}_completion.json',{'completed':True,'epoch':epoch,'training':tr,
                'validation':result,'checkpoint':pointer,'epoch_wall_seconds_this_session':time.perf_counter()-epoch_start})
            print('EPOCH COMPLETE',epoch,'primary Dice',result['mean_dice'],'IoU',result['mean_iou'],
                'comparison 0.525 Dice',result['comparison']['mean_dice'],flush=True)
            cursor['position']=0
        pointer=read('checkpoints/latest.json')
        assert pointer['epoch']==2 and pointer['position']==5722 and pointer['global_microstep']==11444 and pointer['optimizer_steps']==1432
        verify_reload(pointer,model,optimizer,scheduler,capture,probe_items,probe_records);verify_bn(model,initial_bn)
        results={label:read(f'validation/{label}/summary.json') for label in ['baseline','epoch_1','epoch_2']}
        failures=[read(p.name) for p in HERE.glob('failure_*.json')]
        resumes=[read(p.name) for p in HERE.glob('resume_*.json')]
        fine_best=read('checkpoints/best.json')
        improves_parent=fine_best['validation_mean_dice']>baseline['mean_dice']
        overall=({'source':'phase3','path':str((HERE/fine_best['path']).relative_to(ROOT)),
                  'sha256':fine_best['sha256'],'validation_mean_dice':fine_best['validation_mean_dice'],
                  'threshold':.5} if improves_parent else
                 {'source':'phase2_parent','path':CONFIG['parent_checkpoint'],'sha256':parent_sha,
                  'validation_mean_dice':baseline['mean_dice'],'threshold':.5})
        h.atomic_json('checkpoints/best_overall.json',overall)
        preservation_end=verify_protected_sources()
        summary={'phase':'3','finetuning_completed':True,'completed_epochs':2,'training_images_per_epoch':5722,
            'committed_microsteps':11444,'optimizer_steps':1432,'skipped_updates':0,
            'OOM_events':sum(bool(f['OOM']) for f in failures),'nonfinite_events':sum(bool(f['nonfinite']) for f in failures),
            'automatic_restart_attempts':0,
            'verified_process_resumes':len(resumes),'discarded_uncommitted_microsteps':sum(r['discarded_uncommitted_microsteps'] for r in resumes),
            'full_validation_images':1423,'results':results,'latest_checkpoint':pointer,'best_finetuned_checkpoint':fine_best,'best_overall_checkpoint':overall,
            'improves_phase2_parent':improves_parent,'comparison_threshold':CONFIG['comparison_threshold'],
            'protected_sources_before':preservation_start,'protected_sources_after':preservation_end,
            'session_wall_seconds':time.perf_counter()-started,'BN_buffers_unchanged':True,**h.memory(),
            'test_evaluated':False,'extended_beyond_two_epochs':False,'paper_reproduction_established':False}
        h.atomic_json('summary.json',summary);h.atomic_json('progress.json',{'status':'completed','completed_epochs':2,'optimizer_steps':1432})
        print('PHASE 3 TWO ADDITIONAL EPOCHS COMPLETED',json.dumps(summary),flush=True)
    except BaseException as exc:
        name=f'failure_{SESSION}.json'
        h.atomic_json(name,{'completed':False,'exception':repr(exc),'traceback':traceback.format_exc(),
            'cursor':cursor,'stage':h.STAGE,'OOM':isinstance(exc,torch.cuda.OutOfMemoryError),
            'nonfinite':isinstance(exc,FloatingPointError),'automatic_retry':False,
            'host':h.host_memory(),'GPU':h.smi(),
            'session_wall_seconds':time.perf_counter()-started})
        print('PHASE 3 STOPPED',repr(exc),traceback.format_exc(),flush=True)
        raise
    finally:
        if capture: capture.clear();capture.handle.remove()
        if optimizer: optimizer.zero_grad(set_to_none=True)
        del model,optimizer,scheduler,capture;h.release();h.STOP_MONITOR.set();h.thread.join(timeout=6)
        h.atomic_json(f'ending_resources_{SESSION}.json',{**h.host_memory(),**h.smi()})

def read_source_representatives():
    return json.loads((ROOT/'experiments/phase2_pilot/representative_manifest.json').read_text())['validation_indices']

if __name__=='__main__': main()
