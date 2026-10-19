"""Two-epoch streaming Frozen-BN pilot. Resume only from verified committed state."""
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
parser.add_argument('--verify-restart',action='store_true')
args=parser.parse_args()
if (HERE/'summary.json').exists(): raise SystemExit('Completed pilot preserved; refusing rerun')
if (HERE/'training_microsteps.csv').exists() and not args.resume: raise SystemExit('Use --resume for existing history')
if args.resume and not (HERE/'checkpoints/latest.json').exists(): raise SystemExit('No committed checkpoint available')
assert (HERE/'preservation_before.json').exists(),'Run prepare.py first'
import common as h
torch,np,nn=h.torch,h.np,h.nn
from src.data.dataset import group_key

CONFIG={'phase':'2','max_epochs':2,'seed':2026,'split_seed':2026,'precision':'FP32','AMP':False,
    'microbatch':1,'accumulation':8,'augmentation':False,'loader_workers':0,'dataset_tensor_cache':False,
    'normalization':'Frozen original epoch-18 decoder BatchNorm statistics; affine trainable',
    'architecture_modified':False,'vision_lr':1e-5,'decoder_lr':1e-4,'weight_decay':.01,
    'optimizer':'Fresh AdamW foreach=False','scheduler':'Fresh per-update linear decay to 0.5 of starting LR',
    'clip_l2':1.,'clip_foreach':False,'loss':'BCEWithLogits + FP32 sigmoid soft Dice, equal weights, smooth 1e-7',
    'threshold':.5,'image_size':224,'max_text_length':24,'partial_window':'average actual final window size',
    'periodic_checkpoint_every_samples':800,'frozen_text_backbone_eval':True,'attention_weights_retained':False,
    'test_evaluated':False,'selection':'Highest full-validation macro Dice among completed pilot epochs',
    'max_tensor_allocation_mib':2200,'minimum_free_GPU_mib':256,'minimum_available_commit_mib':512,
    'startup_minimum_free_GPU_mib':1800,'startup_minimum_available_commit_mib':2500,
    'model_construction':'Configuration-only direct CUDA initialization; all weights from strict original full checkpoint',
    'low_RAM_guard_mib':32,'low_RAM_guard_consecutive_checks':3,'stop_file':'STOP_REQUESTED',
    'no_automatic_OOM_retry':True,'checkpoint_CPU_full_copy':False,'full_dataset_training':True}
h.CONFIG=CONFIG
SESSION=uuid.uuid4().hex[:8]
low_ram_checks=0
class ControlledRestart(Exception): pass

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
    return result

def validate(model,capture,dataset,label,checkpoint_sha,representatives):
    existing=HERE/f'validation/{label}/summary.json'
    if existing.exists():
        saved=read(f'validation/{label}/summary.json')
        assert saved['checkpoint_sha256']==checkpoint_sha and saved['threshold']==.5 and saved['images']==len(dataset)
        table=csv_rows(saved['per_image_csv'])
        assert h.sha(HERE/saved['per_image_csv'])==saved['per_image_csv_sha256']
        assert [r['image'] for r in table]==[r['name'] for r in dataset.records]
        print('REUSE VERIFIED FULL VALIDATION',label,flush=True)
        return saved
    model.eval();capture.clear();h.release()
    started=time.perf_counter();table=[];attempt=f'{label}_{SESSION}';weak=None
    for i in range(len(dataset)):
        h.STAGE=f'{label} validation {i+1}/{len(dataset)}'
        item=plain(dataset[i])
        with torch.inference_mode():
            image,text,target=h.gpu_item(item)
            probability=model([image,text]);loss,dl,bl=h.loss_fn(capture.logits,target)
            assert torch.isfinite(probability).all() and torch.isfinite(loss),'Nonfinite validation forward'
            a,y=probability[0,0].cpu().numpy().copy(),target[0,0].cpu().numpy().copy()
            row={'validation_index':i,'image':dataset.records[i]['name'],**h.scores(a,y),
                 'loss':float(loss),'dice_loss':float(dl),'bce_loss':float(bl)}
            capture.clear();del image,text,target,probability,loss,dl,bl,item
        table.append(row)
        if i in representatives: h.save_images(f'{attempt}/sample_{i:04d}',[dataset.records[i]],[(a,y)])
        if weak is None or row['dice']<weak[0]: weak=(row['dice'],i,a.copy(),y.copy())
        del a,y
        if (i+1)%200==0:
            print('VALIDATION',label,i+1,'/',len(dataset),flush=True);resource_guard()
        if (HERE/'STOP_REQUESTED').exists(): raise InterruptedError('Stop file requested interruption during validation')
    result=metric_summary(table)
    assert result['images']==1423
    filename=f'validation/{label}/per_image.csv'
    h.csv_write(filename,table)
    h.save_images(f'{attempt}/weakest',[dataset.records[weak[1]]],[(weak[2],weak[3])])
    result.update(label=label,threshold=.5,checkpoint_sha256=checkpoint_sha,per_image_csv=filename,
        per_image_csv_sha256=h.sha(HERE/filename),representative_indices=representatives,
        prediction_directory=f'predictions/{attempt}',weakest_index=weak[1],
        validation_seconds=time.perf_counter()-started,**h.memory())
    h.atomic_json(f'validation/{label}/summary.json',result)
    print('FULL VALIDATION COMPLETE',label,result,flush=True)
    del table,weak;h.release()
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
    payload={'kind':'PHASE2_FROZEN_BN_PILOT_V1','model':model.state_dict(),'optimizer':optimizer.state_dict(),
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
        h.atomic_json('data_manifest.json',manifests);manifest_sha=h.sha(HERE/'data_manifest.json')
        orders={}
        for epoch in [1,2]:
            order=list(range(len(train)));h.random.Random(2026+epoch).shuffle(order);orders[str(epoch)]=order
        h.atomic_json('epoch_orders.json',orders)
        representatives=sorted(h.random.Random(707).sample(range(len(val)),8))
        h.atomic_json('representative_manifest.json',{'seed':707,'selected_before_scoring':True,'validation_indices':representatives,
            'records':[manifests['validation'][i] for i in representatives]})
        CONFIG.update(train_images=len(train),validation_images=len(val),updates_per_epoch=math.ceil(len(train)/8),
            planned_optimizer_updates=2*math.ceil(len(train)/8),train_csv_sha256=h.sha(cfg.train_csv),
            original_split_manifest_sha256=h.sha(ROOT/'outputs/run/prepared_annotations/split_manifest.csv'))
        if (HERE/'config.json').exists(): assert read('config.json')==CONFIG
        else: h.atomic_json('config.json',CONFIG)
        (HERE/f'executed_run_pilot_{SESSION}.py').write_bytes(Path(__file__).read_bytes())
        h.atomic_json(f'environment_{SESSION}.json',{'python':h.sys.version,'torch':torch.__version__,
            'GPU':torch.cuda.get_device_name(),'deterministic_warn_only':True,'attention_backward_may_be_nondeterministic':True})
        h.STAGE='strict original epoch-18 checkpoint loading'
        model=construct_on_gpu(vision,text_source);parent=ROOT/'outputs/run/checkpoints/best_dice.ckpt'
        parent_sha=h.sha(parent)
        assert parent_sha==read_prior_parent_sha()
        state=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
        key='state_dict' if 'state_dict' in state else 'model'
        loaded=model.load_state_dict({k.removeprefix('model.'):v for k,v in state[key].items()},strict=True)
        metadata={k:state.get(k) for k in ['kind','epoch','best_dice'] if isinstance(state.get(k),(str,int,float,type(None)))}
        del state;h.release()
        starting_sha=parameter_sha(model)
        prior=json.loads((ROOT/'experiments/phase1c_normalization/B_frozen_bn/normalization_inventory.json').read_text())
        assert starting_sha==prior['initial_parameter_sha256'],'Original parameter values differ'
        initial_bn=bn_buffers(model);assert len(initial_bn)==18
        for module in model.modules():
            if isinstance(module,nn.MultiheadAttention): module.forward=h.functools.partial(module.forward,need_weights=False)
        capture=h.Capture(model)
        probe_indices=[975,840];probe_items=[plain(train[i]) for i in probe_indices]
        probe_records=[{'name':train.records[i]['name']} for i in probe_indices]
        rr,aa=h.evaluate(model,capture,probe_items,probe_records)
        reference=ROOT/'experiments/phase1c_normalization/B_frozen_bn/predictions/initial'
        difference=max(float(np.max(np.abs(a[0].T-np.load(reference/f'{old:02d}_probability.npy'))))
            for a,old in zip(aa,[0,5]))
        assert difference<=2e-5,'Direct-GPU initialization changed baseline predictions'
        del rr,aa;h.release();resource_guard()
        h.atomic_json('parent_verification.json',{'sha256':parent_sha,'metadata':metadata,
            'strict_missing_keys':loaded.missing_keys,'strict_unexpected_keys':loaded.unexpected_keys,
            'BN_buffer_tensors':len(initial_bn),'architecture_changed':False,'probe_training_indices':probe_indices,
            'initial_parameter_sha256':starting_sha,'matches_phase1C_initial_parameters':True,
            'baseline_probability_max_difference_vs_phase1C':difference,'direct_GPU_construction_verified':True})
        baseline=validate(model,capture,val,'baseline',parent_sha,representatives)
        # The read-only metadata inventory can overlap model startup/baseline inference,
        # but must be committed before any optimizer update.
        wait_started=time.perf_counter()
        while not (HERE/'dataset_metadata_before.json').exists():
            h.STAGE='waiting for dataset preservation metadata before training'
            if time.perf_counter()-wait_started>1800: raise RuntimeError('Dataset metadata preparation did not complete within 30 minutes')
            time.sleep(5)
        optimizer=torch.optim.AdamW([
            {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.model.')],'lr':1e-5},
            {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.model.')],'lr':1e-4}],weight_decay=.01,foreach=False)
        total_updates=CONFIG['planned_optimizer_updates']
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1-.5*min(step,total_updates)/total_updates)
        cursor={'epoch':1,'position':0,'global_microstep':0,'optimizer_steps':0,'skipped_updates':0}
        if args.resume:
            pointer=read('checkpoints/latest.json');path=HERE/pointer['path']
            assert path.resolve().is_relative_to(HERE) and h.sha(path)==pointer['sha256']
            state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert state['config']==CONFIG and state['manifest_sha256']==manifest_sha and state['epoch_orders']==orders
            assert state['parent_checkpoint_sha256']==parent_sha
            model.load_state_dict(state['model'],strict=True);clones=restore_optimizer(optimizer,state['optimizer'])
            scheduler.load_state_dict(state['scheduler']);restore_rng(state)
            cursor=state['cursor'].copy();expected=[a.copy() for a in state['probe_probabilities']]
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
                    if args.verify_restart and not args.resume and cursor['global_microstep']==800:
                        raise ControlledRestart('Verified first checkpoint; controller must launch --resume')
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
            print('EPOCH COMPLETE',epoch,'full validation Dice',result['mean_dice'],'IoU',result['mean_iou'],flush=True)
            cursor['position']=0
        pointer=read('checkpoints/latest.json')
        assert pointer['epoch']==2 and pointer['position']==5722 and pointer['global_microstep']==11444 and pointer['optimizer_steps']==1432
        verify_reload(pointer,model,optimizer,scheduler,capture,probe_items,probe_records);verify_bn(model,initial_bn)
        results={label:read(f'validation/{label}/summary.json') for label in ['baseline','epoch_1','epoch_2']}
        failures=[read(p.name) for p in HERE.glob('failure_*.json')]
        resumes=[read(p.name) for p in HERE.glob('resume_*.json')]
        summary={'phase':'2','diagnostic_completed':True,'completed_epochs':2,'training_images_per_epoch':5722,
            'committed_microsteps':11444,'optimizer_steps':1432,'skipped_updates':0,
            'OOM_events':sum(bool(f['OOM']) for f in failures),'nonfinite_events':sum(bool(f['nonfinite']) for f in failures),
            'controlled_restart_tests':sum(bool(f.get('controlled_restart')) for f in failures),
            'verified_process_resumes':len(resumes),'discarded_uncommitted_microsteps':sum(r['discarded_uncommitted_microsteps'] for r in resumes),
            'full_validation_images':1423,'results':results,'latest_checkpoint':pointer,'best_pilot_checkpoint':read('checkpoints/best.json'),
            'session_wall_seconds':time.perf_counter()-started,'BN_buffers_unchanged':True,**h.memory(),
            'test_evaluated':False,'extended_beyond_two_epochs':False,'paper_reproduction_established':False}
        h.atomic_json('summary.json',summary);h.atomic_json('progress.json',{'status':'completed','completed_epochs':2,'optimizer_steps':1432})
        print('PILOT TWO EPOCHS COMPLETED',json.dumps(summary),flush=True)
    except BaseException as exc:
        name=f'failure_{SESSION}.json'
        h.atomic_json(name,{'completed':False,'exception':repr(exc),'traceback':traceback.format_exc(),
            'cursor':cursor,'stage':h.STAGE,'OOM':isinstance(exc,torch.cuda.OutOfMemoryError),
            'nonfinite':isinstance(exc,FloatingPointError),'controlled_restart':isinstance(exc,ControlledRestart),
            'host':h.host_memory(),'GPU':h.smi(),
            'session_wall_seconds':time.perf_counter()-started})
        print('PILOT STOPPED',repr(exc),traceback.format_exc(),flush=True)
        if isinstance(exc,ControlledRestart): raise SystemExit(75)
        raise
    finally:
        if capture: capture.clear();capture.handle.remove()
        if optimizer: optimizer.zero_grad(set_to_none=True)
        del model,optimizer,scheduler,capture;h.release();h.STOP_MONITOR.set();h.thread.join(timeout=6)
        h.atomic_json(f'ending_resources_{SESSION}.json',{**h.host_memory(),**h.smi()})

def read_prior_parent_sha():
    return json.loads((ROOT/'experiments/phase1c_normalization/B_frozen_bn/normalization_inventory.json').read_text())['parent_checkpoint_sha256']

if __name__=='__main__': main()
