"""One bounded, isolated normalization variant. No full-dataset training or test inputs.

Usage: python -B -u run_variant.py --variant A_current_bn
Resume: same command plus --resume (only the last committed checkpoint is used).
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
VARIANTS={'A_current_bn':'current','B_frozen_bn':'frozen','C_groupnorm':'groupnorm'}
parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=VARIANTS,required=True)
parser.add_argument('--resume',action='store_true')
args=parser.parse_args()
HERE=BASE/args.variant
if (HERE/'summary.json').exists():
    raise SystemExit('Completed variant preserved; refusing to run again')
if (HERE/'training_microsteps.csv').exists() and not args.resume:
    raise SystemExit('Existing history: use --resume; refusing to overwrite')
if args.resume and not (HERE/'checkpoints/latest.json').exists():
    raise SystemExit('No committed checkpoint to resume')
os.environ['PHASE1C_VARIANT_DIR']=str(HERE)
import common as h
torch,np,nn=h.torch,h.np,h.nn
MODE=VARIANTS[args.variant]
MAX_STEP=2000
CONFIG={**h.CONFIG,'phase':'1C','variant':args.variant,'normalization':MODE,
        'max_microsteps':MAX_STEP,'max_optimizer_updates':250,'max_training_wall_seconds':2700,
        'evaluation_every_microsteps':200,'periodic_checkpoint_every_microsteps':400,
        'augmentation':False,'validation_evaluated':True,'validation_samples':32,
        'validation_selection_seed':707,'threshold':.5,
        'architecture_modified':MODE=='groupnorm','groupnorm_max_groups':32,
        'selection':'Final matched budget; minimum evaluation-mode training Dice, then mean Dice',
        'early_success_stopping':False,
        'restart':'All variants start from original epoch-18 weights; fresh identical AdamW'}
h.CONFIG=CONFIG


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))


def training_mode(model):
    model.train()
    model.text_encoder.model.eval()
    if MODE=='frozen':
        for module in model.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()


def bn_buffers(model):
    return {n:v.detach().cpu().clone() for n,v in model.named_buffers()
            if n.endswith(('running_mean','running_var','num_batches_tracked'))}


def parameter_sha(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def inventory(model):
    rows=[]
    for name,module in model.named_modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):
            rows.append({'name':name,'type':type(module).__name__,'channels':module.num_features,
                         'eps':module.eps,'momentum':module.momentum,'affine':module.affine,
                         'track_running_stats':module.track_running_stats})
        elif isinstance(module,nn.GroupNorm):
            rows.append({'name':name,'type':'GroupNorm','channels':module.num_channels,
                         'groups':module.num_groups,'eps':module.eps,'affine':module.affine})
    return rows


def replace_bn(model):
    changes=[]
    for name,old in list(model.named_modules()):
        if not isinstance(old,nn.modules.batchnorm._BatchNorm): continue
        assert name.startswith(('decoder16.','decoder8.','decoder4.','decoder1.','out.')),name
        groups=min(32,old.num_features)
        while old.num_features%groups: groups-=1
        new=nn.GroupNorm(groups,old.num_features,eps=old.eps,affine=old.affine).to(
            device=old.weight.device if old.affine else 'cuda',dtype=torch.float32)
        with torch.no_grad():
            if old.affine:
                new.weight.copy_(old.weight)
                new.bias.copy_(old.bias)
                assert torch.equal(new.weight,old.weight) and torch.equal(new.bias,old.bias)
        parent,key=name.rsplit('.',1)
        setattr(model.get_submodule(parent),key,new)
        changes.append({'name':name,'old':type(old).__name__,'new':'GroupNorm',
                        'channels':old.num_features,'groups':groups,'affine_copied_exactly':True,
                        'removed_buffers':['running_mean','running_var','num_batches_tracked']})
    assert changes
    return changes


def normalization_snapshot(model,initial,step):
    current=bn_buffers(model)
    changed=[n for n in initial if not torch.equal(initial[n],current[n])]
    if MODE=='frozen': assert not changed,'Frozen running statistics changed'
    summary={'microstep':step,'BN_buffer_tensors':len(current),'changed_buffer_tensors':len(changed),
             'maximum_running_mean_delta':max([float((current[n]-initial[n]).abs().max())
                 for n in current if n.endswith('running_mean')]+[0.]),
             'maximum_running_variance_delta':max([float((current[n]-initial[n]).abs().max())
                 for n in current if n.endswith('running_var')]+[0.]),
             'maximum_counter_increment':max([int(current[n]-initial[n])
                 for n in current if n.endswith('num_batches_tracked')]+[0])}
    h.append_csv('normalization_progression.csv',summary)
    return summary


def mode_comparison(model,capture,items,manifest,records,label,eval_rows,eval_arrays):
    """No optimizer update. Restore running statistics and all generator states afterward."""
    saved_buffers=bn_buffers(model)
    rng=torch.get_rng_state().clone()
    cuda_rng=[v.clone() for v in torch.cuda.get_rng_state_all()]
    handles=[]
    inputs=[]
    current_image={'name':None}
    def make_hook(name):
        def hook(module,values):
            x=values[0].detach().float()
            dims=(0,)+tuple(range(2,x.ndim))
            mean=x.mean(dims)
            variance=x.var(dims,unbiased=False)
            effective=x.numel()//x.shape[1]
            row={'image':current_image['name'],'module':name,'input_shape':list(x.shape),
                 'physical_batch':int(x.shape[0]),'samples_per_channel_BHW':effective,
                 'input_channel_mean_rms':float(mean.square().mean().sqrt()),
                 'input_channel_variance_mean':float(variance.mean())}
            if isinstance(module,nn.modules.batchnorm._BatchNorm):
                row.update(running_mean_RMSE=float((mean-module.running_mean).square().mean().sqrt()),
                           variance_to_running_ratio_median=float((variance/(module.running_var+module.eps)).median()),
                           running_counter=int(module.num_batches_tracked))
            inputs.append(row)
        return hook
    for name,module in model.named_modules():
        if isinstance(module,(nn.modules.batchnorm._BatchNorm,nn.GroupNorm)):
            handles.append(module.register_forward_pre_hook(make_hook(name)))
    training_mode(model)
    rows,arrays=[],[]
    try:
        with torch.inference_mode():
            for item,record in zip(items,manifest):
                current_image['name']=record['name']
                image,text,target=h.gpu_item(item)
                probability=model([image,text])
                loss,dl,bl=h.loss_fn(capture.logits,target)
                a,y=probability[0,0].cpu().numpy().copy(),target[0,0].cpu().numpy().copy()
                assert np.isfinite(a).all()
                rows.append({'image':record['name'],**h.scores(a,y),'loss':float(loss),
                             'dice_loss':float(dl),'bce_loss':float(bl)})
                arrays.append((a,y))
                capture.clear()
                del image,text,target,probability,loss,dl,bl
    finally:
        for handle in handles: handle.remove()
        with torch.no_grad():
            for name,v in model.named_buffers():
                if name in saved_buffers: v.copy_(saved_buffers[name])
        assert all(torch.equal(v,saved_buffers[n]) for n,v in bn_buffers(model).items())
        torch.set_rng_state(rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        model.eval()
    differences=[float(np.max(np.abs(a[0]-b[0]))) for a,b in zip(arrays,eval_arrays)]
    details=[{'image':r['image'],'evaluation_dice':e['dice'],'optimization_mode_dice':r['dice'],
              'dice_difference':r['dice']-e['dice'],'probability_max_abs_difference':d}
             for r,e,d in zip(rows,eval_rows,differences)]
    if MODE in ('frozen','groupnorm'): assert max(differences)<=2e-5,details
    h.csv_write(f'{label}_optimization_mode_per_image.csv',rows)
    h.save_images(f'{label}_optimization_mode',records,arrays)
    result={'label':label,'evaluation':h.aggregate(eval_rows),'optimization_mode':h.aggregate(rows),
            'per_image':details,'running_statistics_restored':True,'no_optimizer_updates':True,
            'maximum_probability_difference':max(differences),
            'warning':'Optimization-mode predictions are diagnostic, not evaluation replacements',
            'normalization_inputs':inputs}
    h.atomic_json(f'{label}_mode_consistency.json',result)
    del arrays,rows,saved_buffers
    h.release()
    return result


def load_items(dataset,indices):
    items,manifest,records=[],[],[]
    for index in indices:
        record=dataset.records[index]
        (image,text),target=dataset[index]
        x=image.as_tensor() if hasattr(image,'as_tensor') else image
        y=target.as_tensor() if hasattr(target,'as_tensor') else target
        assert torch.equal(x,image) and torch.equal(y,target)
        items.append(((x,text),y))
        records.append(record)
        manifest.append({'index':index,'name':record['name'],'prompt':record['prompt'],
                         'group':h.QaTaDataset.__module__})
    return items,manifest,records


def restore_optimizer(optimizer,saved):
    optimizer.state.clear()
    h.release()
    optimizer.load_state_dict(saved)
    clones=0
    for state in optimizer.state.values():
        for key in list(state):
            value=state[key]
            if isinstance(value,torch.Tensor) and value.device.type=='cpu':
                state[key]=value.clone()
                assert state[key].data_ptr()!=value.data_ptr()
                clones+=1
    return clones


def resume_history(step):
    stamp=time.strftime('%Y%m%d_%H%M%S')
    discarded=0
    for name in ['training_microsteps.csv','evaluation_progression.csv','normalization_progression.csv']:
        path=HERE/name
        if not path.exists(): continue
        with path.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
        if name=='training_microsteps.csv' and rows:
            discarded=max(0,max(int(r['microstep']) for r in rows)-step)
        if any(int(r['microstep'])>step for r in rows):
            archive=HERE/f'interrupted_{stamp}_{name}'
            archive.write_bytes(path.read_bytes())
            h.csv_write(name,[r for r in rows if int(r['microstep'])<=step])
    return discarded


def main():
    model=capture=optimizer=None
    try:
        source_snapshot=HERE/f'executed_run_variant{"_resume" if args.resume else ""}.py'
        if not source_snapshot.exists(): source_snapshot.write_bytes(Path(__file__).read_bytes())
        assert torch.cuda.is_available()
        assert h.smi()['device_free_mib']>=1800,'Insufficient GPU headroom before model construction'
        assert h.host_memory()['available_commit_mib']>=3500,'Insufficient system commit headroom'
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True)
        h.seed()
        h.atomic_json('config.json',CONFIG)
        h.atomic_json('environment.json',{'python':h.sys.version,'torch':torch.__version__,
                       'GPU':torch.cuda.get_device_name(),'one_ML_runtime':True,
                       'deterministic_algorithms_warn_only':True})
        h.STAGE='construct model and same eight training inputs'
        cfg=h.Config(data_root=ROOT/'_qata_compatible')
        vision,text_source=h._model_sources(cfg)
        tokenizer=h.AutoTokenizer.from_pretrained(text_source,trust_remote_code=True,local_files_only=True)
        dataset=h.QaTaDataset(ROOT/'outputs/run/prepared_annotations/train.csv',ROOT/'_qata_compatible',
                             'train',tokenizer,augment=False,seed=2026)
        previous=json.loads((ROOT/'experiments/phase1b_memory_fix/sample_manifest.json').read_text())
        indices=[r['training_index'] for r in previous]
        items,manifest,records=load_items(dataset,indices)
        for actual,prior in zip(manifest,previous):
            assert actual['name']==prior['name'] and actual['prompt']==prior['prompt']
        h.atomic_json('sample_manifest.json',previous)
        from src.data.dataset import group_key
        train_groups={group_key(r['name']) for r in dataset.records}
        val_dataset=h.QaTaDataset(ROOT/'outputs/run/prepared_annotations/train.csv',ROOT/'_qata_compatible',
                                 'validation',tokenizer,augment=False,seed=2026)
        assert not train_groups&{group_key(r['name']) for r in val_dataset.records}
        val_indices=sorted(h.random.Random(707).sample(range(len(val_dataset)),32))
        val_items,val_manifest,val_records=load_items(val_dataset,val_indices)
        for row in val_manifest: row['group']=group_key(row['name'])
        h.atomic_json('validation_manifest.json',{'indices':val_indices,'records':val_manifest,
                       'split_size':len(val_dataset),'group_leakage':False,'seed':707,
                       'selection_before_any_model_scoring':True,'threshold':.5})
        del dataset,val_dataset,tokenizer
        model=h.create_model(vision,text_source).float().cuda()
        h.STAGE='strict original epoch-18 loading'
        parent=ROOT/'outputs/run/checkpoints/best_dice.ckpt'
        parent_sha=h.sha(parent)
        state=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
        state_key='state_dict' if 'state_dict' in state else 'model'
        original_state=state[state_key]
        original_state={k.removeprefix('model.'):v for k,v in original_state.items()}
        loaded=model.load_state_dict(original_state,strict=True)
        del state,original_state
        h.release()
        original_inventory=inventory(model)
        starting_hash=parameter_sha(model)
        changes=replace_bn(model) if MODE=='groupnorm' else []
        assert parameter_sha(model)==starting_hash,'Only normalization rules/buffers may differ at initialization'
        initial_bn=bn_buffers(model)
        h.atomic_json('normalization_inventory.json',{'original':original_inventory,'variant':inventory(model),
                       'changes':changes,'architecture_modified':MODE=='groupnorm',
                       'output_head_normalization':[r for r in original_inventory if r['name'].startswith(('out.','decoder1.'))],
                       'initial_parameter_sha256':starting_hash,'all_affine_and_other_parameters_identical':True,
                       'parent_checkpoint_sha256':parent_sha,'strict_missing_keys':loaded.missing_keys,
                       'strict_unexpected_keys':loaded.unexpected_keys})
        for module in model.modules():
            if isinstance(module,nn.MultiheadAttention):
                module.forward=h.functools.partial(module.forward,need_weights=False)
        capture=h.Capture(model)
        vision_params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.model.')]
        other_params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.model.')]
        optimizer=torch.optim.AdamW([{'params':vision_params,'lr':3e-5},{'params':other_params,'lr':3e-4}],
                                    weight_decay=.01,foreach=False)
        del vision_params,other_params
        order=list(range(8)); updates=completed=0
        if args.resume:
            pointer=read('checkpoints/latest.json')
            path=HERE/pointer['path']
            assert path.resolve().is_relative_to(HERE) and h.sha(path)==pointer['sha256']
            state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert state['config']['variant']==args.variant and state['parent_checkpoint_sha256']==parent_sha
            model.load_state_dict(state['model'],strict=True)
            clones=restore_optimizer(optimizer,state['optimizer'])
            completed=state['microstep'];updates=state['optimizer_updates'];order=state['sample_order']
            h.random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'].clone())
            torch.cuda.set_rng_state_all([v.clone() for v in state['cuda_rng']])
            expected=state['evaluation'];del state;h.release()
            rr,aa=h.evaluate(model,capture,items,manifest)
            assert h.aggregate(rr)==expected
            maxdiff=max(float(np.max(np.abs(a[0].T-np.load(HERE/f'predictions/step_{completed:04d}/{i:02d}_probability.npy'))))
                        for i,a in enumerate(aa))
            assert maxdiff==0
            discarded=resume_history(completed)
            h.atomic_json(f'resume_{completed:04d}_{time.strftime("%Y%m%d_%H%M%S")}.json',{'checkpoint':pointer,'verified':True,
                          'prediction_max_difference':maxdiff,'cloned_CPU_counters':clones,
                          'discarded_uncommitted_microsteps':discarded})
            del rr,aa
        else:
            h.STAGE='initial normalization and fixed-weight consistency probe'
            rr,aa=h.evaluate(model,capture,items,manifest)
            initial=h.aggregate(rr)
            h.csv_write('initial_per_image.csv',rr)
            h.save_images('initial',records,aa)
            h.append_csv('evaluation_progression.csv',{'microstep':0,'optimizer_updates':0,**initial,**h.memory()})
            mode_comparison(model,capture,items,manifest,records,'initial',rr,aa)
            vv,va=h.evaluate(model,capture,val_items,val_manifest)
            h.csv_write('initial_validation_per_image.csv',vv)
            h.atomic_json('initial_validation_summary.json',h.aggregate(vv))
            h.atomic_json('initial_summary.json',initial)
            del rr,aa,vv,va
            if MODE!='groupnorm':
                rows,arrays=h.evaluate(model,capture,items,manifest)
                reference=ROOT/'experiments/phase1b_memory_fix/predictions/initial'
                diff=max(float(np.max(np.abs(a[0].T-np.load(reference/f'{i:02d}_probability.npy'))))
                         for i,a in enumerate(arrays))
                assert diff<=2e-5
                h.atomic_json('baseline_parity.json',{'max_probability_difference_vs_phase1b':diff,'verified':True})
                del rows,arrays
            h.seed()
        initial=read('initial_summary.json')
        compute_seconds=0.
        if args.resume:
            with (HERE/'training_microsteps.csv').open(newline='',encoding='utf-8') as f:
                compute_seconds=sum(float(r['compute_seconds']) for r in csv.DictReader(f))
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();stop='Matched 2000-microstep diagnostic budget completed'
        for step in range(completed+1,MAX_STEP+1):
            tick=time.perf_counter()
            if (step-1)%8==0:
                h.random.shuffle(order)
                optimizer.zero_grad(set_to_none=True)
            training_mode(model)
            h.STAGE=f'{args.variant} forward {step}'
            sample=order[(step-1)%8]
            image,text,target=h.gpu_item(items[sample])
            output=model([image,text])
            loss,dl,bl=h.loss_fn(capture.logits,target)
            if not torch.isfinite(loss) or not torch.isfinite(output).all():
                raise FloatingPointError('Nonfinite forward/loss')
            row={'microstep':step,'sample_position':sample,'optimizer_updates':updates,'loss':float(loss.detach()),
                 'dice_loss':float(dl.detach()),'bce_loss':float(bl.detach()),'gradient_tensors':0,
                 'nonfinite_gradient_tensors':0,'gradient_norm_before_clip':0.,'watched_parameter_max_update':0.}
            h.STAGE=f'{args.variant} backward {step}'
            (loss/8).backward()
            capture.clear()
            del image,text,target,output,loss,dl,bl
            if step%8==0:
                count,bad,norm=h.gradients(model)
                row.update(gradient_tensors=count,nonfinite_gradient_tensors=len(bad),gradient_norm_before_clip=norm)
                if bad: raise FloatingPointError('Nonfinite gradients: '+str(bad))
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                watched=next(p for n,p in model.named_parameters() if n.startswith('out.') and p.requires_grad)
                prior=watched.detach().clone()
                h.STAGE=f'{args.variant} optimizer update {step}'
                optimizer.step();torch.cuda.synchronize()
                row['watched_parameter_max_update']=float((watched.detach()-prior).abs().max())
                del watched,prior
                updates+=1;row['optimizer_updates']=updates
                if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                    raise FloatingPointError('Nonfinite updated parameters')
                optimizer.zero_grad(set_to_none=True)
            duration=time.perf_counter()-tick
            compute_seconds+=duration;row['compute_seconds']=duration;row.update(h.memory())
            h.append_csv('training_microsteps.csv',row);completed=step
            if row['peak_allocated_mib']>2200: raise RuntimeError('Unexpected CUDA allocation >2200 MiB')
            if step%200==0:
                h.STAGE=f'{args.variant} evaluation {step}'
                rows,arrays=h.evaluate(model,capture,items,manifest)
                result=h.aggregate(rows)
                h.append_csv('evaluation_progression.csv',{'microstep':step,'optimizer_updates':updates,**result,**h.memory()})
                normalization_snapshot(model,initial_bn,step)
                print('CHECK',args.variant,step,result,'HOST',h.host_memory(),flush=True)
                if step%400==0 and step<MAX_STEP:
                    h.STAGE=f'{args.variant} checkpoint {step}'
                    periodic_kind=f'periodic_resume_{time.strftime("%Y%m%d_%H%M%S")}' if args.resume else 'periodic'
                    pointer=h.checkpoint(model,optimizer,step,updates,parent_sha,order,result,periodic_kind)
                    h.save_images(f'step_{step:04d}',records,arrays)
                    if step==400: h.persistence_test(pointer,model,optimizer,capture,items,manifest)
                del rows,arrays;h.release()
                resource={**h.host_memory(),**h.smi()}
                if resource['device_free_mib']<256 or resource['available_commit_mib']<512:
                    stop='Resource headroom safety guard';break
                if time.perf_counter()-started>=CONFIG['max_training_wall_seconds']:
                    stop='45-minute per-variant training wall safety limit';break
        h.STAGE=f'{args.variant} final evaluation and checkpoint'
        rows,arrays=h.evaluate(model,capture,items,manifest)
        final=h.aggregate(rows)
        h.csv_write('final_per_image.csv',rows)
        h.save_images('final',records,arrays)
        final_kind=f'final_resume_{time.strftime("%Y%m%d_%H%M%S")}' if args.resume else 'final'
        pointer=h.checkpoint(model,optimizer,completed,updates,parent_sha,order,final,final_kind)
        h.save_images(f'step_{completed:04d}',records,arrays)
        h.persistence_test(pointer,model,optimizer,capture,items,manifest)
        final_norm=normalization_snapshot(model,initial_bn,completed)
        consistency=mode_comparison(model,capture,items,manifest,records,'final',rows,arrays)
        h.STAGE=f'{args.variant} independent 32-image validation check'
        val_rows,val_arrays=h.evaluate(model,capture,val_items,val_manifest)
        val_result=h.aggregate(val_rows)
        h.csv_write('final_validation_per_image.csv',val_rows)
        h.save_images('validation_final',val_records,val_arrays)
        h.atomic_json('final_validation_summary.json',val_result)
        del arrays,val_arrays,rows,val_rows
        h.release()
        prior_failures=[json.loads(p.read_text(encoding='utf-8')) for p in HERE.glob('failure_*.json')]
        discarded_work=sum(json.loads(p.read_text(encoding='utf-8')).get('discarded_uncommitted_microsteps',0)
                           for p in HERE.glob('resume_*.json'))
        summary={'phase':'1C','variant':args.variant,'architecture_modified':MODE=='groupnorm',
                 'diagnostic_completed':completed==MAX_STEP,'completed_microsteps':completed,
                 'optimizer_updates':updates,'initial':initial,'final':final,'validation_32':val_result,
                 'training_evaluation_consistency':{k:v for k,v in consistency.items() if k!='normalization_inputs'},
                 'normalization_final':final_norm,'OOM_events':sum(bool(r.get('OOM')) for r in prior_failures),
                 'nonfinite_events':sum('nonfinite' in r['message'].lower() for r in prior_failures),
                 'nonfinite_gradient_updates':sum('nonfinite' in r['message'].lower() and
                     any(stage in r['stage'] for stage in ['backward','optimizer']) for r in prior_failures),
                 'discarded_uncommitted_microsteps':discarded_work,
                 'actual_executed_microsteps':completed+discarded_work,
                 'stop_reason':stop,'final_checkpoint':pointer,'training_compute_seconds':compute_seconds,
                 'compute_seconds_per_microstep':compute_seconds/completed,
                 'training_wall_seconds_this_session':time.perf_counter()-started,**h.memory(),
                 'threshold':.5,'test_evaluated':False,'full_dataset_training':False,
                 'generalization_improvement_established':False}
        h.atomic_json('summary.json',summary)
        print('VARIANT COMPLETED',args.variant,summary,flush=True)
    except BaseException as exc:
        h.atomic_json(f'failure_{time.strftime("%Y%m%d_%H%M%S")}.json',{'type':type(exc).__name__,
                      'message':str(exc),'stage':h.STAGE,'traceback':traceback.format_exc(),
                      'host_memory':h.host_memory(),'device':h.smi(),
                      'OOM':isinstance(exc,torch.cuda.OutOfMemoryError)})
        raise
    finally:
        h.STAGE=f'{args.variant} cleanup'
        if capture:
            capture.clear();capture.handle.remove()
        if optimizer: optimizer.zero_grad(set_to_none=True)
        del model,capture,optimizer
        h.release()
        h.STOP_MONITOR.set();h.thread.join(timeout=20)
        h.atomic_json('ending_resources.json',{**h.host_memory(),**h.smi()})


if __name__=='__main__': main()
