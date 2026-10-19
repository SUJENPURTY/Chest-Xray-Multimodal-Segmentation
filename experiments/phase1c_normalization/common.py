"""Isolated, bounded eight-training-image FP32 diagnostic; never opens test data.

Utilities copied into Phase 1C; never imports the previous experiment.
All artifacts and runtime caches are confined to this experiment directory.
"""
from __future__ import annotations
import csv
import ctypes
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(os.environ['PHASE1C_VARIANT_DIR']).resolve()
ROOT = Path(__file__).resolve().parents[2]
assert HERE.is_relative_to(ROOT/'experiments/phase1c_normalization')
for key, value in {
    'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1', 'HF_MODULES_CACHE': str(HERE/'cache/hf_modules'),
    'MPLCONFIGDIR': str(HERE/'cache/matplotlib'), 'TEMP': str(HERE/'tmp'),
    'TMP': str(HERE/'tmp'), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'CUDA_LAUNCH_BLOCKING': '1',
}.items():
    os.environ[key] = value
for directory in ['tmp', 'checkpoints', 'predictions', 'cache']:
    (HERE/directory).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


def atomic_json(name, value):
    path = HERE/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(4*1024*1024):
            h.update(block)
    return h.hexdigest()


def host_memory():
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('total_physical', ctypes.c_ulonglong), ('available_physical', ctypes.c_ulonglong),
                    ('total_page', ctypes.c_ulonglong), ('available_page', ctypes.c_ulonglong),
                    ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong),
                    ('available_extended', ctypes.c_ulonglong)]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return {'ram_load_percent': status.load,
            'available_ram_mib': status.available_physical/2**20,
            'total_ram_mib': status.total_physical/2**20,
            'available_commit_mib': status.available_page/2**20}


def smi():
    fields = 'memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu'
    result = subprocess.run(['nvidia-smi', '--query-gpu='+fields, '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15, creationflags=0x08000000)
    if result.returncode:
        return {'smi_error': result.stderr.strip()}
    values = [float(v.strip()) for v in result.stdout.strip().splitlines()[0].split(',')]
    return dict(zip(['device_total_mib', 'device_used_mib', 'device_free_mib',
                     'gpu_utilization_percent', 'temperature_c'], values))


def append_csv(name, row):
    path = HERE/name
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()


def csv_write(name, rows):
    if not rows:
        return
    path = HERE/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        f.flush()
    os.replace(temp, path)


STAGE = 'startup'
STOP_MONITOR = threading.Event()


def monitor():
    while not STOP_MONITOR.is_set():
        try:
            append_csv('gpu_monitor.csv', {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                          'stage': STAGE, **smi(), **host_memory()})
        except Exception as exc:
            with (HERE/'monitor_errors.txt').open('a', encoding='utf-8') as f:
                f.write(str(exc)+'\n')
        STOP_MONITOR.wait(5)


def preservation_start():
    protected = set(ROOT.glob('notebooks/*.ipynb'))
    protected.update(ROOT.glob('src/**/*.py'))
    protected.update(ROOT.glob('scripts/*.py'))
    for folder in ['outputs/run', 'experiments/phase1_diagnostics']:
        protected.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    inventory = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(protected)}
    atomic_json('preservation_before.json', inventory)
    return inventory


def preservation_end(before):
    changed = [name for name, item in before.items()
               if not (ROOT/name).is_file() or sha(ROOT/name) != item['sha256']]
    atomic_json('preservation_verification.json', {'checked_files': len(before),
                'all_protected_files_unchanged': not changed, 'changed_or_missing': changed})
    if changed:
        raise RuntimeError('Protected files changed: '+str(changed))


print('STARTUP resource snapshot', host_memory(), smi(), flush=True)
atomic_json('startup_resources.json', {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                    **host_memory(), **smi()})
subprocess_result = subprocess.run(['nvidia-smi'], capture_output=True, text=True,
                                  timeout=15, creationflags=0x08000000)
(HERE/'startup_nvidia_smi.txt').write_text(subprocess_result.stdout+subprocess_result.stderr, encoding='utf-8')
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
print('IMPORTING one ML runtime', flush=True)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from transformers import AutoTokenizer
from src.config import Config
from src.data.dataset import QaTaDataset
from src.inference.predictor import _model_sources
from src.models.languide_medseg import create_model

PARENT = ROOT/'outputs/run/checkpoints/best_dice.ckpt'
CONFIG = {
    'seed': 2026, 'precision': 'FP32', 'AMP': False, 'microbatch': 1,
    'accumulation': 8, 'max_microsteps': 2000, 'max_optimizer_updates': 250,
    'max_training_wall_seconds': 1800, 'evaluation_every_microsteps': 80,
    'periodic_checkpoint_every_microsteps': 160, 'augmentation': False,
    'vision_lr': 3e-5, 'decoder_lr': 3e-4, 'weight_decay': .01,
    'clip_l2': 1., 'threshold': .5, 'frozen_text_backbone_eval': True,
    'decoder_batchnorm': 'training during updates; running statistics during evaluation',
    'success_mean_dice': .90, 'success_minimum_dice': .75,
    'success_checks_required': 2, 'full_dataset_training': False,
    'validation_evaluated': False, 'test_evaluated': False,
    'restart': 'Original epoch-18 checkpoint; Phase 1 subset weights were never persisted',
}


def seed():
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)


def memory():
    return {'allocated_mib': torch.cuda.memory_allocated()/2**20,
            'reserved_mib': torch.cuda.memory_reserved()/2**20,
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}


def release():
    gc.collect()
    torch.cuda.empty_cache()


def gpu_item(item):
    (image, text), target = item
    return image.unsqueeze(0).cuda(), {k:v.unsqueeze(0).cuda() for k,v in text.items()}, target.unsqueeze(0).cuda()


class Capture:
    def __init__(self, model):
        self.logits = None
        self.handle = model.out.register_forward_hook(self.hook)
    def hook(self, module, inputs, output):
        self.logits = output
    def clear(self):
        self.logits = None


def loss_fn(logits, target):
    z, y = logits.float(), target.float()
    p = z.sigmoid()
    intersection = (p*y).sum((1,2,3))
    dice = 1-((2*intersection+1e-7)/(p.sum((1,2,3))+y.sum((1,2,3))+1e-7)).mean()
    bce = F.binary_cross_entropy_with_logits(z,y)
    return dice+bce, dice, bce


def scores(p,y):
    pred, truth = p>=.5, y>.5
    tp,fp,fn,tn = [int(np.count_nonzero(a)) for a in
                   [pred&truth,pred&~truth,~pred&truth,~pred&~truth]]
    return {'dice': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'iou': tp/(tp+fp+fn) if tp+fp+fn else 1.,
            'precision': tp/(tp+fp) if tp+fp else 0., 'recall': tp/(tp+fn) if tp+fn else 0.,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'predicted_foreground_fraction':float(pred.mean()), 'true_foreground_fraction':float(truth.mean())}


def aggregate(rows):
    return {'mean_dice':float(np.mean([r['dice'] for r in rows])),
            'mean_iou':float(np.mean([r['iou'] for r in rows])),
            'minimum_dice':min(r['dice'] for r in rows),
            'weakest_image':min(rows,key=lambda r:r['dice'])['image'],
            'mean_loss':float(np.mean([r['loss'] for r in rows]))}


def evaluate(model, capture, items, manifest):
    model.eval()
    rows, arrays = [], []
    with torch.inference_mode():
        for item, record in zip(items,manifest):
            image,text,target = gpu_item(item)
            prob = model([image,text])
            loss,dl,bl = loss_fn(capture.logits,target)
            a,y = prob[0,0].cpu().numpy().copy(), target[0,0].cpu().numpy().copy()
            if not np.isfinite(a).all():
                raise FloatingPointError('Nonfinite evaluation probability')
            rows.append({'image':record['name'],**scores(a,y),'loss':float(loss),
                         'dice_loss':float(dl),'bce_loss':float(bl)})
            arrays.append((a,y))
            capture.clear()
            del image,text,target,prob,loss,dl,bl
    return rows, arrays


def save_images(label, records, arrays):
    # CPU arrays only; GPU inference is completed before visualization.
    directory = HERE/'predictions'/label
    directory.mkdir(parents=True,exist_ok=True)
    for i,(record,(a,y)) in enumerate(zip(records,arrays)):
        with Image.open(record['image']) as opened:
            original = np.asarray(opened.convert('L')).copy()
        if original.shape != (224,224):
            original = np.asarray(Image.fromarray(original).resize((224,224),Image.Resampling.BICUBIC))
        truth,pred = (y>.5).T,(a>=.5).T
        overlay = np.repeat(original[:,:,None],3,axis=2).astype(np.float32)/255
        overlay[truth] = .6*overlay[truth]+.4*np.array([0,1,0])
        overlay[pred] = .6*overlay[pred]+.4*np.array([1,0,0])
        Image.fromarray(original).save(directory/f'{i:02d}_xray.png')
        Image.fromarray((truth*255).astype('uint8')).save(directory/f'{i:02d}_ground_truth.png')
        Image.fromarray((pred*255).astype('uint8')).save(directory/f'{i:02d}_prediction.png')
        Image.fromarray((overlay*255).astype('uint8')).save(directory/f'{i:02d}_overlay.png')
        np.save(directory/f'{i:02d}_probability.npy',a.T)


def atomic_torch(path,payload):
    temp = path.with_name(path.name+'.tmp')
    # No full CPU model clone: torch.save transfers individual storages as needed.
    with temp.open('wb') as f:
        torch.save(payload,f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)


def checkpoint(model,optimizer,step,updates,parent_sha,order,result,kind='periodic'):
    path = HERE/'checkpoints'/f'{kind}_step_{step:04d}.ckpt'
    payload = {'kind':'PHASE1C_NORMALIZATION_V1','model':model.state_dict(),
               'optimizer':optimizer.state_dict(),'microstep':step,'optimizer_updates':updates,
               'parent_checkpoint_sha256':parent_sha,'config':CONFIG,'sample_order':order,
               'evaluation':result,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    started=time.perf_counter()
    assert not path.exists(), str(path)
    atomic_torch(path,payload)
    del payload
    pointer={'path':str(path.relative_to(HERE)),'sha256':sha(path),'bytes':path.stat().st_size,
             'microstep':step,'optimizer_updates':updates,'write_seconds':time.perf_counter()-started}
    atomic_json('checkpoints/latest.json',pointer)
    append_csv('checkpoint_log.csv',pointer)
    print('CHECKPOINT',pointer,flush=True)
    return pointer


def gradients(model):
    count,bad,squared = 0,[],0.
    for name,p in model.named_parameters():
        if p.grad is None:
            continue
        count+=1
        if not torch.isfinite(p.grad).all():
            bad.append(name)
        else:
            squared+=float(p.grad.detach().double().square().sum())
    return count,bad,squared**.5


def verify_memory_changes(model,capture,item):
    global STAGE
    STAGE='attention correctness comparison'
    model.eval()
    model.zero_grad(set_to_none=True)
    image,text,target=gpu_item(item)
    output=model([image,text])
    loss,dl,bl=loss_fn(capture.logits,target)
    reference=output.detach().cpu().numpy().copy()
    reference_loss=float(loss.detach())
    loss.backward()
    grad_path=HERE/'attention_reference_gradients.pt'
    atomic_torch(grad_path,{n:p.grad.detach() for n,p in model.named_parameters() if p.grad is not None})
    capture.clear()
    del output,loss,dl,bl
    model.zero_grad(set_to_none=True)
    release()
    originals=[]
    for module in model.modules():
        if isinstance(module,nn.MultiheadAttention):
            originals.append((module,module.forward))
            module.forward=functools.partial(module.forward,need_weights=False)
    output=model([image,text])
    loss,dl,bl=loss_fn(capture.logits,target)
    probability_difference=float(np.max(np.abs(reference-output.detach().cpu().numpy())))
    loss_difference=abs(reference_loss-float(loss.detach()))
    loss.backward()
    saved=torch.load(grad_path,map_location='cpu',weights_only=True,mmap=True)
    numerator,denominator=0.,0.
    worst_absolute=0.
    for name,p in model.named_parameters():
        if p.grad is None:
            if name in saved:
                raise AssertionError('Gradient disappeared: '+name)
            continue
        old=saved.pop(name).to(p.device)
        delta=p.grad.detach()-old
        numerator+=float(delta.double().square().sum())
        denominator+=float(old.double().square().sum())
        worst_absolute=max(worst_absolute,float(delta.abs().max()))
        del old,delta
    assert not saved
    relative=(numerator/max(denominator,1e-30))**.5
    count,bad,norm=gradients(model)
    accepted=probability_difference<=2e-5 and loss_difference<=2e-5 and relative<=1e-3 and not bad
    if not accepted:
        for module,forward in originals:
            module.forward=forward
    result={'attention_modules':len(originals),'output_max_abs_difference':probability_difference,
            'loss_abs_difference':loss_difference,'all_gradient_relative_l2_difference':relative,
            'gradient_max_abs_difference':worst_absolute,'gradient_tensors':count,
            'nonfinite_gradients':bad,'optimized_gradient_l2':norm,'optimization_adopted':accepted,
            'acceptance': 'output/loss abs <=2e-5; all-gradient relative L2 <=1e-3; all finite',
            'MetaTensor_conversion': 'as_tensor preserves every input element; asserted before comparison'}
    atomic_json('memory_optimization_verification.json',result)
    print('MEMORY CHANGE VERIFICATION',result,flush=True)
    capture.clear()
    del output,loss,dl,bl,image,text,target,saved,originals
    model.zero_grad(set_to_none=True)
    release()
    return result


def persistence_test(pointer,model,optimizer,capture,items,manifest):
    global STAGE
    STAGE='checkpoint reload verification'
    path=HERE/pointer['path']
    before_rows,before_arrays=evaluate(model,capture,items,manifest)
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    result=model.load_state_dict(saved['model'],strict=True)
    optimizer.state.clear()
    release()
    optimizer.load_state_dict(saved['optimizer'])
    # AdamW keeps CPU step tensors as-is. Clone these tiny tensors so they do
    # not keep the entire checkpoint mmap alive after `saved` is released.
    cloned_cpu_state_tensors=0
    for optimizer_state in optimizer.state.values():
        for key in list(optimizer_state):
            value=optimizer_state[key]
            if isinstance(value,torch.Tensor) and value.device.type=='cpu':
                optimizer_state[key]=value.clone()
                cloned_cpu_state_tensors+=1
    del optimizer_state,value
    moment_finite=all(torch.isfinite(v).all().item() for state in optimizer.state.values()
                      for key,v in state.items() if isinstance(v,torch.Tensor))
    steps=sorted({float(state['step']) for state in optimizer.state.values() if 'step' in state})
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'].clone())
    torch.cuda.set_rng_state_all([v.clone() for v in saved['cuda_rng']])
    del saved
    release()
    after_rows,after_arrays=evaluate(model,capture,items,manifest)
    difference=max(float(np.max(np.abs(a[0]-b[0]))) for a,b in zip(before_arrays,after_arrays))
    # Simulate failure before replace: last committed file must survive partial temporary data.
    interrupted=path.with_name(path.name+'.interrupted.tmp')
    try:
        with interrupted.open('wb') as f:
            f.write(b'SIMULATED INTERRUPTED WRITE')
            f.flush()
            os.fsync(f.fileno())
        raise OSError('Deliberate pre-replace failure for persistence verification')
    except OSError:
        unchanged=sha(path)==pointer['sha256']
    finally:
        interrupted.unlink()
    verification={'checkpoint':pointer['path'],'strict_missing_keys':result.missing_keys,
                  'strict_unexpected_keys':result.unexpected_keys,'optimizer_state_tensors_finite':moment_finite,
                  'optimizer_step_counters':steps,'maximum_prediction_difference_after_reload':difference,
                  'cloned_cpu_state_tensors_to_release_mmap':cloned_cpu_state_tensors,
                  'interrupted_write_preserved_committed_checkpoint':unchanged,
                  'verified':not result.missing_keys and not result.unexpected_keys and moment_finite
                             and difference==0 and unchanged and steps==[float(pointer['optimizer_updates'])]}
    atomic_json('checkpoint_persistence_verification.json',verification)
    if not verification['verified']:
        raise AssertionError(verification)
    print('PERSISTENCE VERIFIED',verification,flush=True)
    del before_arrays,after_arrays
    release()

