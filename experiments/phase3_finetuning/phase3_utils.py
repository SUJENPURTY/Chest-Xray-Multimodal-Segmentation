"""Isolated Phase 3 FP32 fine-tuning utilities; never opens test data.

Adapted from verified Phase 2 utilities; no imports with Phase 2 side effects.
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

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
assert HERE == ROOT/'experiments/phase3_finetuning'
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
RESOURCE_FAILURE = None
STOP_MONITOR = threading.Event()


def monitor():
    global RESOURCE_FAILURE
    low_ram_checks = 0
    while not STOP_MONITOR.is_set():
        try:
            snapshot = {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                        'stage': STAGE, **smi(), **host_memory(), **memory()}
            append_csv('gpu_monitor.csv', snapshot)
            low_ram_checks = low_ram_checks + 1 if snapshot['available_ram_mib'] < 32 else 0
            if 'device_free_mib' not in snapshot:
                raise RuntimeError('GPU monitor unavailable')
            if (snapshot['device_free_mib'] < 256 or snapshot['available_commit_mib'] < 512
                    or snapshot['peak_allocated_mib'] > 2200 or low_ram_checks >= 3):
                raise RuntimeError('Resource headroom exceeded: ' + str(snapshot))
        except Exception as exc:
            RESOURCE_FAILURE = str(exc)
            append_csv('monitor_errors.csv', {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'), 'error': str(exc)})
            return
        STOP_MONITOR.wait(5)








print('STARTUP resource snapshot', host_memory(), smi(), flush=True)
atomic_json('startup_resources.json', {'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                    **host_memory(), **smi()})
subprocess_result = subprocess.run(['nvidia-smi'], capture_output=True, text=True,
                                  timeout=15, creationflags=0x08000000)
(HERE/'startup_nvidia_smi.txt').write_text(subprocess_result.stdout+subprocess_result.stderr, encoding='utf-8')
thread = threading.Thread(target=monitor, daemon=True)
# Start monitoring only after the ML imports at the end of this module.
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

PARENT = None  # The runner resolves the immutable Phase 2 parent from config.json.
CONFIG = {}


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


def scores(p,y,threshold=.5):
    pred, truth = p>=threshold, y>.5
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
    assert not path.exists(), 'Immutable checkpoint already exists'
    with temp.open('xb') as f:
        torch.save(payload,f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp,path)


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



thread.start()
