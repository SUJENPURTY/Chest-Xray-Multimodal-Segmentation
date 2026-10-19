"""Read-only inventory and isolated helper preparation. No ML runtime."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4*1024*1024): h.update(b)
    return h.hexdigest()
def atomic(name,obj):
    p=HERE/name;t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

if (HERE/'training_microsteps.csv').exists(): raise SystemExit('Existing pilot: do not prepare again')
source=(ROOT/'experiments/phase1c_normalization/common.py').read_text(encoding='utf-8')
source=source.split('\ndef checkpoint(')[0]+source[source.index('\ndef gradients('):source.index('\ndef verify_memory_changes(')]
source=source.replace("HERE = Path(os.environ['PHASE1C_VARIANT_DIR']).resolve()",'HERE = Path(__file__).resolve().parent')
source=source.replace("assert HERE.is_relative_to(ROOT/'experiments/phase1c_normalization')","assert HERE == ROOT/'experiments/phase2_pilot'")
source=source.replace('Isolated, bounded eight-training-image FP32 diagnostic; never opens test data.','Isolated Phase 2 FP32 pilot utilities; never opens test data.')
ast.parse(source)
assert shutil.disk_usage(ROOT).free>50*2**30
helper=HERE/'common.py'
if not helper.exists(): helper.write_text(source,encoding='utf-8')
else: assert helper.read_text(encoding='utf-8')==source
inventory=HERE/'preservation_before.json'
if not inventory.exists():
    # Original/Phase1/1B hashes were independently verified at the end of Phase1C,
    # immediately before this authorized continuation. Retain that provenance.
    previous=ROOT/'experiments/phase1c_normalization'
    proof=json.loads((previous/'preservation_verification.json').read_text())
    assert proof['all_protected_files_unchanged'] and proof['checked_files']==939
    protected=json.loads((previous/'preservation_before.json').read_text())
    for name,v in protected.items():
        assert (ROOT/name).stat().st_size==v['bytes']
        v['mtime_ns']=(ROOT/name).stat().st_mtime_ns
        v['initial_hash_provenance']='Verified end-of-Phase1C inventory'
    files=sorted(p for p in previous.rglob('*') if p.is_file())
    for i,p in enumerate(files):
        protected[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns,
            'initial_hash_provenance':'Fresh Phase2 pre-run SHA256'}
        if i%100==0: print('PHASE1C INVENTORY',i,'/',len(files),flush=True)
    atomic('preservation_before.json',protected)
    print('Protected files',len(protected),flush=True)
if not (HERE/'dataset_metadata_before.json').exists():
    roots=[ROOT/'_qata_compatible',ROOT/'Project/LanGuideMedSeg-MICCAI2023/Data',ROOT/'text_annotation']
    metadata={}
    def scan(base):
        with os.scandir(base) as entries:
            for entry in entries:
                if entry.is_dir(): scan(entry.path)
                elif entry.is_file():
                    stat=entry.stat()
                    name=os.path.relpath(entry.path,ROOT)
                    metadata[name]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
                    if len(metadata)%5000==0: print('DATASET METADATA',len(metadata),flush=True)
    for base in roots:
        if base.is_dir(): scan(base)
    atomic('dataset_metadata_before.json',metadata)
    print('Read-only dataset file metadata',len(metadata),flush=True)
print('Preparation complete; no training performed',flush=True)
