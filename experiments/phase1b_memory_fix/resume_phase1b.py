"""Continue only the same eight-image diagnostic from its last committed state.

The first process was deliberately stopped after checkpoint 240 to fix CPU
optimizer scalar ownership. First-attempt evidence is preserved in attempt01/.
No test/validation inputs. This is not a full-dataset training runner.
"""
import csv
import json
import time
import traceback
import run_phase1b as h

torch,np,nn=h.torch,h.np,h.nn
HERE,ROOT=h.HERE,h.ROOT


def table(name):
    with (HERE/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    model=capture=optimizer=None
    before=json.loads((HERE/'attempt01/preservation_before.json').read_text())
    try:
        assert not (HERE/'resume_training_microsteps.csv').exists(), 'Refusing to overwrite resumed training'
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True,warn_only=True)
        interruption=json.loads((HERE/'controlled_interruption.json').read_text(encoding='utf-8-sig'))
        pointer=interruption['resume_checkpoint']
        resume_path=HERE/pointer['path']
        assert resume_path.resolve().is_relative_to(HERE.resolve())
        assert h.sha(resume_path)==pointer['sha256']
        start_step=pointer['microstep']
        original_rows=table('attempt01/training_microsteps.csv')
        executed_first=max(int(r['microstep']) for r in original_rows)
        discarded=executed_first-start_step
        # Count discarded work against the total cap: never exceed 2,000 actual microsteps.
        max_step=2000-discarded
        max_step-=max_step%8
        h.CONFIG.update(max_microsteps=max_step,max_optimizer_updates=max_step//8,
                        max_training_wall_seconds=1200,periodic_checkpoint_every_microsteps=160,
                        restart='Resume committed Phase 1B state after controlled CPU mmap ownership fix',
                        checkpoint_selection='Best mean among safely persisted checks; every 160 microsteps and final')
        h.atomic_json('resume_config.json',{**h.CONFIG,'resume_checkpoint':pointer,
                      'executed_first_attempt_microsteps':executed_first,
                      'discarded_uncommitted_microsteps':discarded,'total_actual_microstep_cap':2000})
        h.seed()
        manifest=json.loads((HERE/'sample_manifest.json').read_text())
        cfg=h.Config(data_root=ROOT/'_qata_compatible')
        h.STAGE='resume model construction'
        vision,text_source=h._model_sources(cfg)
        tokenizer=h.AutoTokenizer.from_pretrained(text_source,trust_remote_code=True,local_files_only=True)
        dataset=h.QaTaDataset(ROOT/'outputs/run/prepared_annotations/train.csv',ROOT/'_qata_compatible',
                             'train',tokenizer,augment=False,seed=2026)
        items=[]
        records=[]
        for record in manifest:
            index=record['training_index']
            actual=dataset.records[index]
            assert actual['name']==record['name'] and actual['prompt']==record['prompt']
            (image,text),target=dataset[index]
            plain_image=image.as_tensor() if hasattr(image,'as_tensor') else image
            plain_target=target.as_tensor() if hasattr(target,'as_tensor') else target
            assert torch.equal(image,plain_image) and torch.equal(target,plain_target)
            items.append(((plain_image,text),plain_target))
            records.append(actual)
        h.dataset_records=records
        model=h.create_model(vision,text_source).float().cuda()
        del dataset,tokenizer,image,target,plain_image,plain_target
        h.release()
        optimization=json.loads((HERE/'memory_optimization_verification.json').read_text())
        if optimization['optimization_adopted']:
            for module in model.modules():
                if isinstance(module,nn.MultiheadAttention):
                    module.forward=h.functools.partial(module.forward,need_weights=False)
        capture=h.Capture(model)
        vision_params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.model.')]
        other_params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.model.')]
        optimizer=torch.optim.AdamW([{'params':vision_params,'lr':3e-5},{'params':other_params,'lr':3e-4}],
                                    weight_decay=.01,foreach=False)
        del vision_params,other_params
        h.STAGE='resume checkpoint loading and mmap ownership correction'
        resources_before_load=h.host_memory()
        saved=torch.load(resume_path,map_location='cpu',weights_only=False,mmap=True)
        loaded=model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        clone_count,clone_bytes=0,0
        for optimizer_state in optimizer.state.values():
            for key in list(optimizer_state):
                value=optimizer_state[key]
                if isinstance(value,torch.Tensor) and value.device.type=='cpu':
                    clone_bytes+=value.numel()*value.element_size()
                    optimizer_state[key]=value.clone()
                    assert optimizer_state[key].data_ptr()!=value.data_ptr()
                    clone_count+=1
        del optimizer_state,value
        # CUDA/CPU generators copy state internally; explicitly pass owned copies.
        h.random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].clone())
        torch.cuda.set_rng_state_all([v.clone() for v in saved['cuda_rng']])
        order=saved['sample_order']
        updates=saved['optimizer_updates']
        parent_sha=saved['parent_checkpoint_sha256']
        expected=saved['evaluation']
        resources_mapping_alive=h.host_memory()
        del saved
        h.release()
        resources_mapping_released=h.host_memory()
        h.atomic_json('checkpoint_mmap_ownership_fix.json',{
            'checkpoint':pointer,'strict_missing_keys':loaded.missing_keys,
            'strict_unexpected_keys':loaded.unexpected_keys,
            'CPU_state_tensors_cloned':clone_count,'CPU_state_bytes_cloned':clone_bytes,
            'independent_storage_asserted':True,'resources_before_load':resources_before_load,
            'resources_mapping_alive':resources_mapping_alive,'resources_mapping_released':resources_mapping_released,
            'source_evidence':{'optimizer_policy':'.venv/Lib/site-packages/torch/optim/optimizer.py:756-760',
                               'shared_mmap':'.venv/Lib/site-packages/torch/serialization.py:1883-1885'}})
        print('MMAP OWNERSHIP FIX',clone_count,clone_bytes,resources_mapping_released,flush=True)
        h.STAGE='resumed checkpoint prediction verification'
        rows,arrays=h.evaluate(model,capture,items,manifest)
        resumed=h.aggregate(rows)
        prior_directory=HERE/'predictions'/f'step_{start_step:04d}'
        difference=max(float(np.max(np.abs(a[0].T-np.load(prior_directory/f'{i:02d}_probability.npy'))))
                       for i,a in enumerate(arrays))
        counters=sorted({float(state['step']) for state in optimizer.state.values() if 'step' in state})
        assert difference==0 and counters==[float(updates)] and resumed==expected
        h.atomic_json('resume_verification.json',{'checkpoint':pointer,'prediction_max_abs_difference':difference,
                      'optimizer_step_counters':counters,'evaluation_matches_saved':True,
                      'CPU_steps_own_storage':True,'verified':True})
        del rows,arrays
        initial=json.loads((HERE/'baseline_consistency.json').read_text())['initial']
        best_pointer=json.loads((HERE/'checkpoints/best.json').read_text())
        best=expected
        best_step=start_step
        successes=0
        completed=start_step
        stop='microstep limit reached'
        first_compute=sum(float(r['compute_seconds']) for r in original_rows if int(r['microstep'])<=start_step)
        compute_seconds=first_compute
        resumed_compute=0.
        first_peak=max(float(r['peak_allocated_mib']) for r in original_rows)
        first_reserved=max(float(r['peak_reserved_mib']) for r in original_rows)
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        last_point=None
        for step in range(start_step+1,max_step+1):
            tick=time.perf_counter()
            if (step-1)%8==0:
                h.random.shuffle(order)
                optimizer.zero_grad(set_to_none=True)
            model.train()
            model.text_encoder.model.eval()
            h.STAGE=f'resumed forward microstep {step}'
            image,text,target=h.gpu_item(items[order[(step-1)%8]])
            output=model([image,text])
            loss,dl,bl=h.loss_fn(capture.logits,target)
            if not torch.isfinite(loss) or not torch.isfinite(output).all():
                raise FloatingPointError('Nonfinite forward/loss')
            row={'microstep':step,'sample_position':order[(step-1)%8],'loss':float(loss.detach()),
                 'dice_loss':float(dl.detach()),'bce_loss':float(bl.detach()),'optimizer_updates':updates,
                 'gradient_tensors':0,'nonfinite_gradient_tensors':0,'gradient_norm_before_clip':0.,
                 'watched_parameter_max_update':0.,'compute_seconds':0.}
            h.STAGE=f'resumed backward microstep {step}'
            (loss/8).backward()
            torch.cuda.synchronize()
            capture.clear()
            del output,loss,dl,bl,image,text,target
            if step%8==0:
                h.STAGE=f'resumed optimizer microstep {step}'
                count,bad,norm=h.gradients(model)
                row.update(gradient_tensors=count,nonfinite_gradient_tensors=len(bad),gradient_norm_before_clip=norm)
                if bad:
                    raise FloatingPointError('Nonfinite gradients: '+str(bad))
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,
                                        error_if_nonfinite=True,foreach=False)
                watched=next(model.out.parameters())
                prior=watched.detach().clone()
                optimizer.step()
                torch.cuda.synchronize()
                row['watched_parameter_max_update']=float((watched.detach()-prior).abs().max())
                del prior,watched
                updates+=1
                row['optimizer_updates']=updates
                if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                    raise FloatingPointError('Nonfinite updated parameters')
                optimizer.zero_grad(set_to_none=True)
            duration=time.perf_counter()-tick
            compute_seconds+=duration
            resumed_compute+=duration
            row['compute_seconds']=duration
            row.update(h.memory())
            h.append_csv('resume_training_microsteps.csv',row)
            completed=step
            if row['peak_allocated_mib']>2200:
                raise RuntimeError('Unexpected tensor allocation >2200 MiB')
            if step%80==0 or step==max_step:
                h.STAGE=f'resumed evaluation microstep {step}'
                rows,arrays=h.evaluate(model,capture,items,manifest)
                result=h.aggregate(rows)
                point={'microstep':step,'optimizer_updates':updates,'elapsed_seconds':time.perf_counter()-started,
                       **result,**h.memory()}
                h.append_csv('resume_evaluation_progression.csv',point)
                h.csv_write('latest_per_image.csv',rows)
                print('RESUMED TRAINING CHECK',point,'HOST',h.host_memory(),flush=True)
                if step%160==0 or step==max_step:
                    h.STAGE=f'resumed checkpoint export microstep {step}'
                    latest=h.checkpoint(model,optimizer,step,updates,parent_sha,order,result,'resumed')
                    h.save_images(f'step_{step:04d}',records,arrays)
                    if result['mean_dice']>best['mean_dice']:
                        best,best_step,best_pointer=result,step,latest
                        h.csv_write('best_per_image.csv',rows)
                        h.atomic_json('checkpoints/best.json',best_pointer)
                last_point=point
                del rows,arrays
                h.release()
                resource={**h.host_memory(),**h.smi()}
                if resource.get('device_free_mib',4096)<256 or resource['available_commit_mib']<512:
                    stop='Resource headroom safety guard'
                    break
                successes=successes+1 if result['mean_dice']>=.90 and result['minimum_dice']>=.75 else 0
                if successes>=2:
                    stop='Mean >=0.90 and minimum >=0.75 on two successive checks'
                    break
                if time.perf_counter()-started>=1200:
                    stop='20-minute resumed training wall limit'
                    break
        h.STAGE='resumed final checkpoint and evaluation'
        rows,arrays=h.evaluate(model,capture,items,manifest)
        final=h.aggregate(rows)
        h.csv_write('final_per_image.csv',rows)
        h.save_images('final',records,arrays)
        final_pointer=h.checkpoint(model,optimizer,completed,updates,parent_sha,order,final,'final')
        if final['mean_dice']>best['mean_dice']:
            best,best_step,best_pointer=final,completed,final_pointer
            h.csv_write('best_per_image.csv',rows)
            h.save_images(f'step_{completed:04d}',records,arrays)
            h.atomic_json('checkpoints/best.json',final_pointer)
        h.persistence_test(final_pointer,model,optimizer,capture,items,manifest)
        memory=h.memory()
        memory['peak_allocated_mib']=max(first_peak,memory['peak_allocated_mib'])
        memory['peak_reserved_mib']=max(first_reserved,memory['peak_reserved_mib'])
        h.atomic_json('summary.json',{'diagnostic_completed':True,'stop_reason':stop,
             'initial':initial,'best':best,'best_microstep':best_step,'best_checkpoint':best_pointer,
             'best_selection':'Best mean among persisted checkpoints, not every observed check',
             'final':final,'final_checkpoint':final_pointer,'completed_microsteps':completed,
             'actual_executed_microsteps':completed+discarded,'discarded_uncommitted_microsteps':discarded,
             'optimizer_updates':updates,'actual_executed_optimizer_updates':updates+discarded//8,
             'nonfinite_gradient_updates':0,'OOM_events':0,'controlled_restart_count':1,
             'training_compute_seconds':compute_seconds,'compute_seconds_per_microstep':compute_seconds/completed,
             'resumed_compute_seconds_per_microstep':resumed_compute/(completed-start_step),
             'training_wall_seconds_including_exports':time.perf_counter()-started,
             'wall_seconds_scope':'Resumed stage only; first attempt logged separately; excludes imports/setup',
             'success_criterion_met':successes>=2,'memory_optimization':optimization,**memory,
             'AMP_used':False,'training_samples':8,'threshold':.5,'validation_or_test_evaluated':False})
        h.STAGE='weakest-image inference diagnostic'
        h.weak_image_diagnostic(model,capture,items,manifest,rows)
        print('PHASE 1B COMPLETED',final,'stop',stop,flush=True)
    except BaseException as exc:
        h.atomic_json('failure.json',{'type':type(exc).__name__,'message':str(exc),'stage':h.STAGE,
                      'traceback':traceback.format_exc(),'host_memory':h.host_memory(),'device':h.smi()})
        raise
    finally:
        h.STAGE='resume cleanup and preservation verification'
        if capture:
            capture.clear()
            capture.handle.remove()
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        del model,capture,optimizer
        h.release()
        h.preservation_end(before)
        h.STOP_MONITOR.set()
        h.thread.join(timeout=20)
        h.atomic_json('ending_resources.json',{**h.host_memory(),**h.smi()})


if __name__=='__main__':
    main()
